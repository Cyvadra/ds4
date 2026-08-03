package ds4

import (
	"context"
	"encoding/json"
	"fmt"
)

// FIMRequest is the JSON body sent to POST /beta/completions (Fill-In-the-Middle).
type FIMRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	Suffix      string   `json:"suffix,omitempty"`
	MaxTokens   int      `json:"max_tokens,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	Stream      bool     `json:"stream,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	Echo        bool     `json:"echo,omitempty"`
}

// FIMResponse is the JSON body returned from POST /beta/completions.
type FIMResponse struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []FIMChoice `json:"choices"`
	Usage   Usage       `json:"usage"`
}

// FIMChoice is one completion alternative inside a FIMResponse.
type FIMChoice struct {
	Index        int    `json:"index"`
	Text         string `json:"text"`
	FinishReason string `json:"finish_reason"`
}

// Text returns the generated text of the first choice.
func (r *FIMResponse) Text() string {
	if len(r.Choices) == 0 {
		return ""
	}
	return r.Choices[0].Text
}

// FinishReason returns the finish reason of the first choice.
func (r *FIMResponse) FinishReason() string {
	if len(r.Choices) == 0 {
		return ""
	}
	return r.Choices[0].FinishReason
}

// ---------------------------------------------------------------------------
// FIMBuilder
// ---------------------------------------------------------------------------

// FIMBuilder composes a Fill-In-the-Middle completion request using a fluent interface.
// Create one via Client.FIM(). FIM is a beta feature and uses the /beta/completions endpoint.
//
//	resp, err := client.FIM().
//	    Prompt("func greet(name string) string {\n    return ").
//	    Suffix("\n}").
//	    Do()
//	fmt.Println(resp.Text())
type FIMBuilder struct {
	client      *Client
	model       string
	prompt      string
	suffix      string
	maxTokens   int
	temperature *float64
	topP        *float64
	stop        []string
	echo        bool
}

func newFIMBuilder(c *Client) *FIMBuilder {
	return &FIMBuilder{
		client: c,
		model:  ModelDeepSeekV4Flash,
	}
}

// Model overrides the model (default: ModelDeepSeekV4Flash).
func (b *FIMBuilder) Model(model string) *FIMBuilder {
	b.model = model
	return b
}

// Prompt sets the text before the cursor (the prefix).
func (b *FIMBuilder) Prompt(prompt string) *FIMBuilder {
	b.prompt = prompt
	return b
}

// Suffix sets the text after the cursor. The model will fill in the middle.
func (b *FIMBuilder) Suffix(suffix string) *FIMBuilder {
	b.suffix = suffix
	return b
}

// MaxTokens sets the upper bound on generated tokens.
func (b *FIMBuilder) MaxTokens(n int) *FIMBuilder {
	b.maxTokens = n
	return b
}

// Temperature sets the sampling temperature (0.0–2.0).
func (b *FIMBuilder) Temperature(t float64) *FIMBuilder {
	b.temperature = &t
	return b
}

// TopP sets nucleus-sampling probability (0.0–1.0).
func (b *FIMBuilder) TopP(p float64) *FIMBuilder {
	b.topP = &p
	return b
}

// Stop sets sequences at which the model will stop generating.
func (b *FIMBuilder) Stop(sequences ...string) *FIMBuilder {
	b.stop = sequences
	return b
}

// Echo includes the prompt in the response text.
func (b *FIMBuilder) Echo() *FIMBuilder {
	b.echo = true
	return b
}

// Build constructs the FIMRequest without sending it.
func (b *FIMBuilder) Build() (*FIMRequest, error) {
	if b.prompt == "" {
		return nil, fmt.Errorf("ds4: fim: Prompt is required")
	}
	return &FIMRequest{
		Model:       b.model,
		Prompt:      b.prompt,
		Suffix:      b.suffix,
		MaxTokens:   b.maxTokens,
		Temperature: b.temperature,
		TopP:        b.topP,
		Stop:        b.stop,
		Echo:        b.echo,
	}, nil
}

// Do sends the FIM completion request and returns the full response.
func (b *FIMBuilder) Do() (*FIMResponse, error) {
	return b.DoWithContext(context.Background())
}

// DoWithContext is like Do but honours ctx for cancellation and deadlines.
func (b *FIMBuilder) DoWithContext(ctx context.Context) (*FIMResponse, error) {
	req, err := b.Build()
	if err != nil {
		return nil, err
	}

	data, err := b.client.doRequest(ctx, "POST", "/beta/completions", req)
	if err != nil {
		return nil, err
	}

	var resp FIMResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("ds4: unmarshal response: %w", err)
	}
	return &resp, nil
}

// Stream sends the FIM request with streaming enabled and calls handler for
// each FIMStreamChunk. Return a non-nil error from handler to abort early.
func (b *FIMBuilder) Stream(handler func(FIMStreamChunk) error) error {
	return b.StreamWithContext(context.Background(), handler)
}

// StreamWithContext is like Stream but honours ctx for cancellation and deadlines.
func (b *FIMBuilder) StreamWithContext(ctx context.Context, handler func(FIMStreamChunk) error) error {
	req, err := b.Build()
	if err != nil {
		return err
	}
	req.Stream = true

	body, err := b.client.doStreamRequest(ctx, "POST", "/beta/completions", req)
	if err != nil {
		return err
	}
	defer body.Close()

	return parseSSE(body, func(data []byte) error {
		var chunk FIMStreamChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			return fmt.Errorf("ds4: unmarshal stream chunk: %w", err)
		}
		return handler(chunk)
	})
}
