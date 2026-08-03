package ds4

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// DefaultBaseURL is the root URL for all DeepSeek API calls.
const DefaultBaseURL = "https://api.deepseek.com"

// Client is the entry point for all DeepSeek API interactions.
// Create one with New and then call Chat() or FIM() to start building requests.
type Client struct {
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

// New creates a new Client authenticated with apiKey.
//
//	client := ds4.New(os.Getenv("DEEPSEEK_API_KEY"))
func New(apiKey string) *Client {
	return &Client{
		apiKey:  apiKey,
		baseURL: DefaultBaseURL,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

// WithBaseURL overrides the default API base URL (useful for proxies or testing).
func (c *Client) WithBaseURL(url string) *Client {
	c.baseURL = url
	return c
}

// WithHTTPClient replaces the default net/http client.
func (c *Client) WithHTTPClient(httpClient *http.Client) *Client {
	c.httpClient = httpClient
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

// doRequest marshals body as JSON, sends a POST to path, and returns the raw
// response bytes on success (HTTP 200). Any non-200 status is converted to *APIError.
func (c *Client) doRequest(ctx context.Context, method, path string, body any) ([]byte, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ds4: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("ds4: create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ds4: http: %w", err)
	}
	defer resp.Body.Close()

	respData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ds4: read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, parseAPIError(resp.StatusCode, respData)
	}
	return respData, nil
}

// doStreamRequest is like doRequest but leaves the response body open for
// SSE reading. The caller must close the returned ReadCloser.
func (c *Client) doStreamRequest(ctx context.Context, method, path string, body any) (io.ReadCloser, error) {
	data, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("ds4: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, method, c.baseURL+path, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("ds4: create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ds4: http: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respData, _ := io.ReadAll(resp.Body)
		return nil, parseAPIError(resp.StatusCode, respData)
	}
	return resp.Body, nil
}

// APIError is returned when the DeepSeek API responds with a non-200 status.
type APIError struct {
	StatusCode int
	Message    string `json:"message"`
	Type       string `json:"type"`
	Param      string `json:"param"`
	Code       any    `json:"code"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("ds4: HTTP %d: %s", e.StatusCode, e.Message)
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

func parseAPIError(statusCode int, data []byte) *APIError {
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
	return apiErr
}
