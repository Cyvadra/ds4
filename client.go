package ds4

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"
)

// DefaultBaseURL is the root URL for all DeepSeek API calls.
const DefaultBaseURL = "https://api.deepseek.com"

// DefaultSystemPrompt is included as the first message of every new chat.
const DefaultSystemPrompt = "You are a helpful assistant."

// Client is the entry point for all DeepSeek API interactions.
// Create one with New and then call Chat() or FIM() to start building requests.
type Client struct {
	apiKey       string
	baseURL      string
	localKey     string
	localURL     string
	localFIM     bool
	httpClient   *http.Client
	maxRetries   int
	systemPrompt string

	modelsMu      sync.Mutex
	modelsCond    *sync.Cond
	modelsLoading bool
	modelsLoaded  bool
	models        map[string]backendKind
	officialErr   error
	localErr      error
}

type backendKind uint8

const (
	backendOfficial backendKind = iota
	backendLocal
)

type backend struct {
	kind    backendKind
	baseURL string
	apiKey  string
}

// New creates a new Client authenticated with apiKey. An optional defaultSystemPrompt
// replaces DefaultSystemPrompt for chats created by this client.
//
//	client := ds4.New(os.Getenv("DEEPSEEK_API_KEY"))
//
// The client sets no overall request deadline, because that would also cap
// reading of streamed responses. Bound individual calls with the
// ...WithContext variants instead.
func New(apiKey string, defaultSystemPrompt ...string) *Client {
	systemPrompt := DefaultSystemPrompt
	if len(defaultSystemPrompt) > 0 && defaultSystemPrompt[0] != "" {
		systemPrompt = defaultSystemPrompt[0]
	}
	client := &Client{
		apiKey:       apiKey,
		baseURL:      DefaultBaseURL,
		systemPrompt: systemPrompt,
		httpClient: &http.Client{
			Transport: &http.Transport{
				Proxy: http.ProxyFromEnvironment,
				DialContext: (&net.Dialer{
					Timeout:   30 * time.Second,
					KeepAlive: 30 * time.Second,
				}).DialContext,
				TLSHandshakeTimeout:   10 * time.Second,
				ResponseHeaderTimeout: 120 * time.Second,
			},
		},
	}
	client.modelsCond = sync.NewCond(&client.modelsMu)
	return client
}

// WithBaseURL overrides the default API base URL (useful for proxies or testing).
func (c *Client) WithBaseURL(url string) *Client {
	c.baseURL = strings.TrimRight(url, "/")
	return c
}

// WithLocalKey sets the API key sent to the local OpenAI-compatible server.
// An empty key is allowed for servers that do not require authentication.
func (c *Client) WithLocalKey(key string) *Client {
	c.localKey = key
	return c
}

// WithLocalURL adds a local OpenAI-compatible server to model routing.
func (c *Client) WithLocalURL(url string) *Client {
	c.localURL = strings.TrimRight(url, "/")
	return c
}

// WithLocalFIM enables adapting local FIM requests to /completions.
func (c *Client) WithLocalFIM(enabled bool) *Client {
	c.localFIM = enabled
	return c
}

// WithHTTPClient replaces the default net/http client. Avoid setting Timeout on
// the supplied client: it also bounds streaming response reads.
func (c *Client) WithHTTPClient(httpClient *http.Client) *Client {
	c.httpClient = httpClient
	return c
}

// WithRetry retries requests that fail with a retryable status (429, 500, 503)
// up to n additional attempts, using exponential backoff with jitter and
// honouring any Retry-After header. Retries are disabled by default.
func (c *Client) WithRetry(n int) *Client {
	c.maxRetries = n
	return c
}

// Chat returns a new ChatBuilder for composing chat completion requests.
func (c *Client) Chat() *ChatBuilder {
	return newChatBuilder(c)
}

// FIM returns a new FIMBuilder for Fill-In-the-Middle (beta) completion requests.
func (c *Client) FIM() *FIMBuilder {
	return newFIMBuilder(c)
}

type modelList struct {
	Data []struct {
		ID      string   `json:"id"`
		Aliases []string `json:"aliases"`
	} `json:"data"`
	Models []struct {
		Name  string `json:"name"`
		Model string `json:"model"`
	} `json:"models"`
}

// Ping refreshes the official and local model lists. Discovery failures are
// reported independently; models from a backend that succeeded remain usable.
func (c *Client) Ping() error {
	return c.refreshModels(context.Background(), true)
}

func (c *Client) resolveBackend(ctx context.Context, model string) (backend, error) {
	c.modelsMu.Lock()
	hadLoaded := c.modelsLoaded
	c.modelsMu.Unlock()
	c.refreshModels(ctx, false)

	c.modelsMu.Lock()
	kind, ok := c.models[model]
	officialErr, localErr := c.officialErr, c.localErr
	c.modelsMu.Unlock()
	if !ok && hadLoaded && (officialErr != nil || localErr != nil) {
		c.refreshModels(ctx, true)
		c.modelsMu.Lock()
		kind, ok = c.models[model]
		officialErr, localErr = c.officialErr, c.localErr
		c.modelsMu.Unlock()
	}
	if !ok {
		if officialErr != nil || localErr != nil {
			return backend{}, fmt.Errorf("ds4: unknown model %q (model discovery incomplete: %w)", model, errorsJoin(officialErr, localErr))
		}
		return backend{}, fmt.Errorf("ds4: unknown model %q", model)
	}
	if kind == backendLocal {
		return backend{kind: kind, baseURL: c.localURL, apiKey: c.localKey}, nil
	}
	return backend{kind: kind, baseURL: c.baseURL, apiKey: c.apiKey}, nil
}

func (c *Client) refreshModels(ctx context.Context, force bool) error {
	c.modelsMu.Lock()
	for c.modelsLoading {
		c.modelsCond.Wait()
	}
	if c.modelsLoaded && !force {
		c.modelsMu.Unlock()
		return nil
	}
	c.modelsLoading = true
	c.modelsMu.Unlock()

	models := make(map[string]backendKind)
	officialErr := c.loadModels(ctx, backend{kind: backendOfficial, baseURL: c.baseURL, apiKey: c.apiKey}, models)
	var localErr error
	if c.localURL != "" {
		localErr = c.loadModels(ctx, backend{kind: backendLocal, baseURL: c.localURL, apiKey: c.localKey}, models)
	}

	c.modelsMu.Lock()
	c.models = models
	c.officialErr = officialErr
	c.localErr = localErr
	c.modelsLoaded = true
	c.modelsLoading = false
	c.modelsCond.Broadcast()
	c.modelsMu.Unlock()
	return errorsJoin(officialErr, localErr)
}

func (c *Client) loadModels(ctx context.Context, target backend, models map[string]backendKind) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.baseURL+"/models", nil)
	if err != nil {
		return fmt.Errorf("%s models: create request: %w", target.name(), err)
	}
	if target.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+target.apiKey)
	}
	req.Header.Set("Accept", "application/json")
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("%s models: %w", target.name(), err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("%s models: read response: %w", target.name(), err)
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s models: %w", target.name(), parseAPIError(resp.StatusCode, resp.Header, data))
	}
	var list modelList
	if err := json.Unmarshal(data, &list); err != nil {
		return fmt.Errorf("%s models: unmarshal response: %w", target.name(), err)
	}
	for _, model := range list.Data {
		if model.ID != "" {
			models[model.ID] = target.kind
		}
		for _, alias := range model.Aliases {
			if alias != "" {
				models[alias] = target.kind
			}
		}
	}
	for _, model := range list.Models {
		name := model.Name
		if name == "" {
			name = model.Model
		}
		if name != "" {
			models[name] = target.kind
		}
	}
	return nil
}

func (b backend) name() string {
	if b.kind == backendLocal {
		return "local"
	}
	return "official"
}

func errorsJoin(officialErr, localErr error) error {
	switch {
	case officialErr != nil && localErr != nil:
		return fmt.Errorf("official: %v; local: %w", officialErr, localErr)
	case officialErr != nil:
		return officialErr
	default:
		return localErr
	}
}

// send builds and dispatches a single HTTP attempt.
func (c *Client) send(ctx context.Context, target backend, method, path string, data []byte, accept string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, target.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("ds4: create request: %w", err)
	}
	if target.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+target.apiKey)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", accept)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ds4: http: %w", err)
	}
	return resp, nil
}

// doRequest marshals body as JSON, sends a POST to path, and returns the raw
// response bytes on success (HTTP 200). Any non-200 status is converted to *APIError.
func (c *Client) doRequest(ctx context.Context, target backend, method, path string, body any) ([]byte, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ds4: marshal request: %w", err)
	}

	for attempt := 0; ; attempt++ {
		resp, err := c.send(ctx, target, method, path, data, "application/json")
		if err != nil {
			return nil, err
		}

		respData, readErr := io.ReadAll(resp.Body)
		resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("ds4: read response: %w", readErr)
		}
		if resp.StatusCode == http.StatusOK {
			return respData, nil
		}

		apiErr := parseAPIError(resp.StatusCode, resp.Header, respData)
		if attempt >= c.maxRetries || !apiErr.IsRetryable() {
			return nil, apiErr
		}
		if err := waitBackoff(ctx, attempt, apiErr.RetryAfter); err != nil {
			return nil, err
		}
	}
}

// doStreamRequest is like doRequest but leaves the response body open for
// SSE reading. The caller must close the returned ReadCloser.
func (c *Client) doStreamRequest(ctx context.Context, target backend, method, path string, body any) (io.ReadCloser, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ds4: marshal request: %w", err)
	}

	for attempt := 0; ; attempt++ {
		resp, err := c.send(ctx, target, method, path, data, "text/event-stream")
		if err != nil {
			return nil, err
		}
		if resp.StatusCode == http.StatusOK {
			return resp.Body, nil
		}

		respData, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		apiErr := parseAPIError(resp.StatusCode, resp.Header, respData)
		if attempt >= c.maxRetries || !apiErr.IsRetryable() {
			return nil, apiErr
		}
		if err := waitBackoff(ctx, attempt, apiErr.RetryAfter); err != nil {
			return nil, err
		}
	}
}

// waitBackoff sleeps before the next attempt, preferring the server's
// Retry-After hint over exponential backoff with jitter.
func waitBackoff(ctx context.Context, attempt int, retryAfter time.Duration) error {
	const (
		base     = 500 * time.Millisecond
		maxDelay = 30 * time.Second
	)

	delay := retryAfter
	if delay <= 0 {
		delay = base << attempt
		if delay > maxDelay || delay <= 0 {
			delay = maxDelay
		}
		delay += time.Duration(rand.Int63n(int64(delay / 2)))
	}
	if delay > maxDelay {
		delay = maxDelay
	}

	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// APIError is returned when the DeepSeek API responds with a non-200 status.
type APIError struct {
	StatusCode int
	Message    string `json:"message"`
	Type       string `json:"type"`
	Param      string `json:"param"`
	Code       any    `json:"code"`
	// RetryAfter carries the server's Retry-After hint, when present.
	RetryAfter time.Duration
}

func (e *APIError) Error() string {
	return fmt.Sprintf("ds4: HTTP %d: %s", e.StatusCode, e.Message)
}

// IsRetryable reports whether the request may succeed if retried after a wait.
func (e *APIError) IsRetryable() bool {
	switch e.StatusCode {
	case http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusServiceUnavailable:
		return true
	default:
		return false
	}
}

// errorEnvelope matches the {"error": {...}} wrapper used by the DeepSeek API.
type errorEnvelope struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Param   string `json:"param"`
		Code    any    `json:"code"`
	} `json:"error"`
}

func parseAPIError(statusCode int, header http.Header, data []byte) *APIError {
	apiErr := &APIError{StatusCode: statusCode}
	var env errorEnvelope
	if err := json.Unmarshal(data, &env); err == nil && env.Error.Message != "" {
		apiErr.Message = env.Error.Message
		apiErr.Type = env.Error.Type
		apiErr.Param = env.Error.Param
		apiErr.Code = env.Error.Code
	} else {
		apiErr.Message = string(data)
	}
	if seconds, err := strconv.Atoi(header.Get("Retry-After")); err == nil && seconds > 0 {
		apiErr.RetryAfter = time.Duration(seconds) * time.Second
	}
	return apiErr
}
