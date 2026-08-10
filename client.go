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
	httpClient   *http.Client
	maxRetries   int
	systemPrompt string
	// local, when non-nil, targets a llama.cpp-compatible local server instead
	// of the DeepSeek API and translates DeepSeek-private fields and endpoints.
	local *LocalBackend
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
	return &Client{
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
}

// WithBaseURL overrides the default API base URL (useful for proxies or testing).
func (c *Client) WithBaseURL(url string) *Client {
	c.baseURL = url
	return c
}

// LocalBackend configures the client for a llama.cpp-compatible local server.
//
// llama.cpp exposes an OpenAI-compatible surface but lacks DeepSeek's private
// /beta route, thinking parameters, and separated reasoning_content field.
// With a local backend the client:
//
//   - never routes to /beta/... — strict tools and chat-prefix requests use the
//     ordinary /chat/completions endpoint,
//   - drops the thinking and reasoning_effort fields (local servers ignore
//     them),
//   - restores the reasoning/answer split on the client side from inline
//     markers such as <reasoning>...</reasoning> (see DefaultReasoningMarkers
//     and SplitReasoning),
//   - maps Fill-In-the-Middle requests to the server's OpenAI-compatible
//     /completions endpoint using input_prefix/input_suffix.
type LocalBackend struct {
	// FIM, when true, adapts FIMBuilder requests to the local server's
	// /completions endpoint. Leave it false when the loaded model has no
	// Fill-In-the-Middle support.
	FIM bool
}

// WithLocalBackend switches the client to a llama.cpp-compatible local server
// instead of the DeepSeek API. Point WithBaseURL at the server's
// OpenAI-compatible prefix, for example http://localhost:8080/v1.
//
//	client := ds4.New("any-key").
//	    WithBaseURL("http://localhost:8080/v1").
//	    WithLocalBackend(ds4.LocalBackend{FIM: true})
func (c *Client) WithLocalBackend(b LocalBackend) *Client {
	c.local = &b
	return c
}

// isLocal reports whether the client is configured for a local server.
func (c *Client) isLocal() bool { return c.local != nil }

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

// send builds and dispatches a single HTTP attempt.
func (c *Client) send(ctx context.Context, method, path string, data []byte, accept string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("ds4: create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
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
func (c *Client) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ds4: marshal request: %w", err)
	}

	for attempt := 0; ; attempt++ {
		resp, err := c.send(ctx, method, path, data, "application/json")
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
func (c *Client) doStreamRequest(ctx context.Context, method, path string, body any) (io.ReadCloser, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ds4: marshal request: %w", err)
	}

	for attempt := 0; ; attempt++ {
		resp, err := c.send(ctx, method, path, data, "text/event-stream")
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
