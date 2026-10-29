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

// DefaultAnthropicVersion is the anthropic-version header sent to the
// Anthropic Messages backend unless WithAnthropicVersion overrides it.
const DefaultAnthropicVersion = "2023-06-01"

// DefaultSystemPrompt is included as the first message of every new chat.
const DefaultSystemPrompt = "You are a helpful assistant."

// Client is the entry point for all DeepSeek API interactions.
// Create one with New and then call Chat() or FIM() to start building requests.
type Client struct {
	apiKey             string
	baseURL            string
	localKey           string
	localURL           string
	localFIM           bool
	anthropicURL       string
	anthropicKey       string
	anthropicVersion   string
	anthropicStatic    []string
	anthropicWorkspace string
	httpClient         *http.Client
	maxRetries         int
	systemPrompt       string

	modelsMu           sync.Mutex
	modelsCond         *sync.Cond
	modelsLoading      bool
	modelsLoaded       bool
	officialModels     map[string]struct{}
	localModels        map[string]struct{}
	anthropicModels    map[string]struct{}
	officialErr        error
	localErr           error
	anthropicErr       error
	lastForceRefreshAt time.Time
}

// unknownModelRefreshCooldown bounds how often an unrecognized model name can
// trigger a forced re-discovery, preventing repeated bad model names from
// hammering both backends' /models endpoints.
const unknownModelRefreshCooldown = 5 * time.Second

type backendKind uint8

const (
	backendOfficial backendKind = iota
	backendLocal
	backendAnthropic
)

type backend struct {
	kind      backendKind
	baseURL   string
	apiKey    string
	version   string
	workspace string
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
		apiKey:           apiKey,
		baseURL:          DefaultBaseURL,
		anthropicVersion: DefaultAnthropicVersion,
		systemPrompt:     systemPrompt,
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

// WithAnthropicURL adds a server speaking the Anthropic Messages API to model
// routing, such as https://api.anthropic.com or DeepSeek's
// https://api.deepseek.com/anthropic. The URL excludes the /v1 segment.
func (c *Client) WithAnthropicURL(url string) *Client {
	c.anthropicURL = strings.TrimRight(url, "/")
	return c
}

// WithAnthropicKey sets the x-api-key credential for the Anthropic backend.
// An empty key omits the header.
func (c *Client) WithAnthropicKey(key string) *Client {
	c.anthropicKey = key
	return c
}

// WithAnthropicVersion overrides the anthropic-version header
// (default DefaultAnthropicVersion).
func (c *Client) WithAnthropicVersion(version string) *Client {
	c.anthropicVersion = version
	return c
}

// WithAnthropicWorkspace sends the anthropic-workspace-id header, which the
// Anthropic API requires for keys that are not scoped to a single workspace.
func (c *Client) WithAnthropicWorkspace(id string) *Client {
	c.anthropicWorkspace = id
	return c
}

// WithAnthropicModels registers model names for the Anthropic backend instead
// of discovering them through GET /v1/models. Use it for compatible servers
// that do not expose a models endpoint.
func (c *Client) WithAnthropicModels(models ...string) *Client {
	c.anthropicStatic = append([]string(nil), models...)
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

type modelEntry struct {
	ID      string   `json:"id"`
	Name    string   `json:"name"`
	Model   string   `json:"model"`
	Aliases []string `json:"aliases"`
}

type modelList struct {
	Data   []modelEntry `json:"data"`
	Models []modelEntry `json:"models"`
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
	kind, ok := c.lookupModel(model)
	discoveryErr := c.discoveryErrLocked()
	c.modelsMu.Unlock()
	if !ok && hadLoaded && c.shouldForceRefresh(discoveryErr) {
		c.refreshModels(ctx, true)
		c.modelsMu.Lock()
		kind, ok = c.lookupModel(model)
		discoveryErr = c.discoveryErrLocked()
		c.modelsMu.Unlock()
	}
	if !ok {
		if discoveryErr != nil {
			return backend{}, fmt.Errorf("ds4: unknown model %q (model discovery incomplete: %w)", model, discoveryErr)
		}
		return backend{}, fmt.Errorf("ds4: unknown model %q", model)
	}
	return c.backendFor(kind), nil
}

func (c *Client) backendFor(kind backendKind) backend {
	switch kind {
	case backendLocal:
		return backend{kind: kind, baseURL: c.localURL, apiKey: c.localKey}
	case backendAnthropic:
		return backend{kind: kind, baseURL: c.anthropicURL, apiKey: c.anthropicKey, version: c.anthropicVersion, workspace: c.anthropicWorkspace}
	default:
		return backend{kind: kind, baseURL: c.baseURL, apiKey: c.apiKey}
	}
}

// discoveryErrLocked must be called with modelsMu held.
func (c *Client) discoveryErrLocked() error {
	return errorsJoin(c.officialErr, c.localErr, c.anthropicErr)
}

// lookupModel must be called with modelsMu held. Local discovery wins when
// several sources advertise the same model, then the Anthropic backend, per
// docs/api.md.
func (c *Client) lookupModel(model string) (backendKind, bool) {
	if _, ok := c.localModels[model]; ok {
		return backendLocal, true
	}
	if _, ok := c.anthropicModels[model]; ok {
		return backendAnthropic, true
	}
	if _, ok := c.officialModels[model]; ok {
		return backendOfficial, true
	}
	return 0, false
}

// shouldForceRefresh reports whether an unrecognized model justifies a forced
// re-discovery: always after a prior discovery error, otherwise at most once
// per unknownModelRefreshCooldown so unknown model names can't be used to
// force repeated network round-trips.
func (c *Client) shouldForceRefresh(discoveryErr error) bool {
	if discoveryErr != nil {
		return true
	}
	c.modelsMu.Lock()
	defer c.modelsMu.Unlock()
	if time.Since(c.lastForceRefreshAt) < unknownModelRefreshCooldown {
		return false
	}
	c.lastForceRefreshAt = time.Now()
	return true
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

	officialModels := make(map[string]struct{})
	localModels := make(map[string]struct{})
	officialErr := c.loadModels(ctx, backend{kind: backendOfficial, baseURL: c.baseURL, apiKey: c.apiKey}, officialModels)
	var localErr error
	if c.localURL != "" {
		localErr = c.loadModels(ctx, c.backendFor(backendLocal), localModels)
	}
	anthropicModels := make(map[string]struct{})
	var anthropicErr error
	if c.anthropicURL != "" {
		if len(c.anthropicStatic) > 0 {
			addModelNames(anthropicModels, c.anthropicStatic...)
		} else {
			anthropicErr = c.loadModels(ctx, c.backendFor(backendAnthropic), anthropicModels)
		}
	}

	c.modelsMu.Lock()
	c.officialModels = officialModels
	c.localModels = localModels
	c.officialErr = officialErr
	c.localErr = localErr
	c.anthropicModels = anthropicModels
	c.anthropicErr = anthropicErr
	c.modelsLoaded = true
	c.modelsLoading = false
	c.modelsCond.Broadcast()
	c.modelsMu.Unlock()
	return errorsJoin(officialErr, localErr, anthropicErr)
}

func (c *Client) loadModels(ctx context.Context, target backend, models map[string]struct{}) error {
	path := "/models"
	if target.kind == backendAnthropic {
		path = "/v1/models?limit=1000"
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.baseURL+path, nil)
	if err != nil {
		return fmt.Errorf("%s models: create request: %w", target.name(), err)
	}
	target.setAuth(req)
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
	for _, model := range append(list.Data, list.Models...) {
		addModelNames(models, model.ID, model.Name, model.Model)
		addModelNames(models, model.Aliases...)
		if target.kind == backendAnthropic {
			addModelNames(models, undatedModelID(model.ID))
		}
	}
	return nil
}

// undatedModelID returns the short alias of a dated Anthropic model ID, such as
// claude-haiku-4-5 for claude-haiku-4-5-20251001, or "" when id has no date.
// The Anthropic API accepts both forms but lists only the dated one.
func undatedModelID(id string) string {
	cut := strings.LastIndexByte(id, '-')
	if cut < 0 || len(id)-cut-1 != 8 {
		return ""
	}
	for _, r := range id[cut+1:] {
		if r < '0' || r > '9' {
			return ""
		}
	}
	return id[:cut]
}

func addModelNames(models map[string]struct{}, names ...string) {
	for _, name := range names {
		if name != "" {
			models[name] = struct{}{}
		}
	}
}

func (b backend) name() string {
	switch b.kind {
	case backendLocal:
		return "local"
	case backendAnthropic:
		return "anthropic"
	default:
		return "official"
	}
}

// setAuth applies the credential headers expected by the backend's protocol.
func (b backend) setAuth(req *http.Request) {
	if b.kind == backendAnthropic {
		if b.apiKey != "" {
			req.Header.Set("x-api-key", b.apiKey)
		}
		req.Header.Set("anthropic-version", b.version)
		if b.workspace != "" {
			req.Header.Set("anthropic-workspace-id", b.workspace)
		}
		return
	}
	if b.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.apiKey)
	}
}

// errorsJoin combines per-backend discovery errors in official, local,
// anthropic order. A single error is returned unchanged.
func errorsJoin(errs ...error) error {
	names := []string{"official", "local", "anthropic"}
	var present []int
	for index, err := range errs {
		if err != nil {
			present = append(present, index)
		}
	}
	switch len(present) {
	case 0:
		return nil
	case 1:
		return errs[present[0]]
	}
	parts := make([]string, 0, len(present)-1)
	for _, index := range present[:len(present)-1] {
		parts = append(parts, fmt.Sprintf("%s: %v", names[index], errs[index]))
	}
	last := present[len(present)-1]
	return fmt.Errorf("%s; %s: %w", strings.Join(parts, "; "), names[last], errs[last])
}

// send builds and dispatches a single HTTP attempt.
func (c *Client) send(ctx context.Context, target backend, method, path string, data []byte, accept string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, target.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("ds4: create request: %w", err)
	}
	target.setAuth(req)
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

// statusOverloaded is the Anthropic API's non-standard "overloaded" status.
const statusOverloaded = 529

// IsRetryable reports whether the request may succeed if retried after a wait.
func (e *APIError) IsRetryable() bool {
	switch e.StatusCode {
	case http.StatusTooManyRequests, http.StatusInternalServerError, http.StatusServiceUnavailable, statusOverloaded:
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
