package ds4

import (
	"context"
	"encoding/json"
	"fmt"
)

// ChatRequest is the JSON body sent to POST /chat/completions.
type ChatRequest struct {
	Model            string          `json:"model"`
	Messages         []Message       `json:"messages"`
	MaxTokens        int             `json:"max_tokens,omitempty"`
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"top_p,omitempty"`
	Stream           bool            `json:"stream,omitempty"`
	StreamOptions    *StreamOptions  `json:"stream_options,omitempty"`
	Stop             []string        `json:"stop,omitempty"`
	FrequencyPenalty *float64        `json:"frequency_penalty,omitempty"`
	PresencePenalty  *float64        `json:"presence_penalty,omitempty"`
	ResponseFormat   *ResponseFormat `json:"response_format,omitempty"`
	Tools            []Tool          `json:"tools,omitempty"`
	ToolChoice       any             `json:"tool_choice,omitempty"`
	LogProbs         bool            `json:"logprobs,omitempty"`
	TopLogProbs      int             `json:"top_logprobs,omitempty"`
}

// ChatResponse is the JSON body returned from POST /chat/completions.
type ChatResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage"`
}

// Content returns the text content of the first choice's message.
func (r *ChatResponse) Content() string {
	if len(r.Choices) == 0 {
		return ""
	}
	return r.Choices[0].Message.Content
}

// ReasoningContent returns the chain-of-thought content from the deepseek-reasoner model.
func (r *ChatResponse) ReasoningContent() string {
	if len(r.Choices) == 0 {
		return ""
	}
	return r.Choices[0].Message.ReasoningContent
}

// ToolCalls returns any tool invocations requested by the model in the first choice.
func (r *ChatResponse) ToolCalls() []ToolCall {
	if len(r.Choices) == 0 {
		return nil
	}
	return r.Choices[0].Message.ToolCalls
}

// FinishReason returns why the model stopped generating (e.g. "stop", "tool_calls").
func (r *ChatResponse) FinishReason() string {
	if len(r.Choices) == 0 {
		return ""
	}
	return r.Choices[0].FinishReason
}

// FirstMessage returns a pointer to the first choice's Message, or nil when empty.
func (r *ChatResponse) FirstMessage() *Message {
	if len(r.Choices) == 0 {
		return nil
	}
	return &r.Choices[0].Message
}

// Choice is one completion alternative inside a ChatResponse.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// ---------------------------------------------------------------------------
// ChatBuilder
// ---------------------------------------------------------------------------

// ChatBuilder composes a chat completion request using a fluent interface.
// Create one via Client.Chat(); each method mutates and returns the same builder.
//
//	resp, err := client.Chat().
//	    System("You are a helpful assistant.").
//	    User("Explain goroutines.").
//	    Do()
type ChatBuilder struct {
	client *Client

	model            string
	messages         []Message
	maxTokens        int
	temperature      *float64
	topP             *float64
	stop             []string
	frequencyPenalty *float64
	presencePenalty  *float64
	responseFormat   *ResponseFormat
	tools            []Tool
	toolChoice       any
	logProbs         bool
	topLogProbs      int
	includeUsage     bool
}

func newChatBuilder(c *Client) *ChatBuilder {
	return &ChatBuilder{
		client: c,
		model:  ModelDeepSeekV4Flash,
	}
}

// Model overrides the model used for this request (default: ModelDeepSeekV4Flash).
func (b *ChatBuilder) Model(model string) *ChatBuilder {
	b.model = model
	return b
}

// System appends a system message to the conversation.
func (b *ChatBuilder) System(content string) *ChatBuilder {
	b.messages = append(b.messages, Message{Role: RoleSystem, Content: content})
	return b
}

// User appends a user message to the conversation.
func (b *ChatBuilder) User(content string) *ChatBuilder {
	b.messages = append(b.messages, Message{Role: RoleUser, Content: content})
	return b
}

// Assistant appends an assistant message. Use this when building multi-round
// conversations to include a previous assistant reply in the history.
//
//	resp1, _ := builder.User("Q1").Do()
//	resp2, _ := builder.Assistant(resp1.Content()).User("Q2").Do()
func (b *ChatBuilder) Assistant(content string) *ChatBuilder {
	b.messages = append(b.messages, Message{Role: RoleAssistant, Content: content})
	return b
}

// ToolResult appends a tool result message. toolCallID must match the ID from
// the model's ToolCall.
func (b *ChatBuilder) ToolResult(toolCallID, content string) *ChatBuilder {
	b.messages = append(b.messages, Message{
		Role:       RoleTool,
		Content:    content,
		ToolCallID: toolCallID,
	})
	return b
}

// AddMessage appends a raw Message to the conversation.
func (b *ChatBuilder) AddMessage(msg Message) *ChatBuilder {
	b.messages = append(b.messages, msg)
	return b
}

// WithMessages replaces the entire message history.
func (b *ChatBuilder) WithMessages(msgs []Message) *ChatBuilder {
	b.messages = msgs
	return b
}

// Prefix enables chat prefix completion (beta) by appending an assistant message
// with Prefix:true. The model will continue generating from this prefix.
//
//	resp, err := client.Chat().
//	    User("Write a Go function").
//	    Prefix("func hello() {\n").
//	    Do()
func (b *ChatBuilder) Prefix(prefix string) *ChatBuilder {
	b.messages = append(b.messages, Message{
		Role:    RoleAssistant,
		Content: prefix,
		Prefix:  true,
	})
	return b
}

// ThinkingMode switches the model to deepseek-reasoner, which returns both a
// chain-of-thought in ReasoningContent and a final answer in Content.
func (b *ChatBuilder) ThinkingMode() *ChatBuilder {
	b.model = ModelDeepSeekReasoner
	return b
}

// JSONMode sets response_format to {"type":"json_object"} so the model returns
// a valid JSON object.
func (b *ChatBuilder) JSONMode() *ChatBuilder {
	b.responseFormat = &ResponseFormat{Type: "json_object"}
	return b
}

// MaxTokens sets the upper bound on generated tokens.
func (b *ChatBuilder) MaxTokens(n int) *ChatBuilder {
	b.maxTokens = n
	return b
}

// Temperature sets the sampling temperature (0.0–2.0).
func (b *ChatBuilder) Temperature(t float64) *ChatBuilder {
	b.temperature = &t
	return b
}

// TopP sets nucleus-sampling probability (0.0–1.0).
func (b *ChatBuilder) TopP(p float64) *ChatBuilder {
	b.topP = &p
	return b
}

// Stop sets sequences at which the model will stop generating.
func (b *ChatBuilder) Stop(sequences ...string) *ChatBuilder {
	b.stop = sequences
	return b
}

// FrequencyPenalty reduces repetition by penalising frequently used tokens (−2.0–2.0).
func (b *ChatBuilder) FrequencyPenalty(p float64) *ChatBuilder {
	b.frequencyPenalty = &p
	return b
}

// PresencePenalty encourages topic diversity by penalising tokens already present (−2.0–2.0).
func (b *ChatBuilder) PresencePenalty(p float64) *ChatBuilder {
	b.presencePenalty = &p
	return b
}

// Tool appends a tool definition that the model may call.
func (b *ChatBuilder) Tool(tool Tool) *ChatBuilder {
	b.tools = append(b.tools, tool)
	return b
}

// WithTools replaces all tool definitions.
func (b *ChatBuilder) WithTools(tools []Tool) *ChatBuilder {
	b.tools = tools
	return b
}

// ToolChoice controls how the model selects tools:
// "none" disables tools, "auto" lets the model decide, or pass a specific
// tool selector map such as map[string]any{"type":"function","function":map[string]any{"name":"fn"}}.
func (b *ChatBuilder) ToolChoice(choice any) *ChatBuilder {
	b.toolChoice = choice
	return b
}

// LogProbs enables log-probability output for the topN most likely tokens at
// each generated position.
func (b *ChatBuilder) LogProbs(topN int) *ChatBuilder {
	b.logProbs = true
	b.topLogProbs = topN
	return b
}

// IncludeStreamUsage requests token usage statistics in the final streaming chunk.
func (b *ChatBuilder) IncludeStreamUsage() *ChatBuilder {
	b.includeUsage = true
	return b
}

// Clone returns a deep copy of the builder, allowing you to fork a conversation
// or try different parameters without mutating the original.
func (b *ChatBuilder) Clone() *ChatBuilder {
	msgs := make([]Message, len(b.messages))
	copy(msgs, b.messages)
	tools := make([]Tool, len(b.tools))
	copy(tools, b.tools)

	cp := *b
	cp.messages = msgs
	cp.tools = tools
	return &cp
}

// Build constructs the ChatRequest from the builder state without sending it.
// Returns an error when the request would be invalid (e.g. no messages).
func (b *ChatBuilder) Build() (*ChatRequest, error) {
	if len(b.messages) == 0 {
		return nil, fmt.Errorf("ds4: chat: at least one message is required")
	}

	req := &ChatRequest{
		Model:            b.model,
		Messages:         b.messages,
		MaxTokens:        b.maxTokens,
		Temperature:      b.temperature,
		TopP:             b.topP,
		Stop:             b.stop,
		FrequencyPenalty: b.frequencyPenalty,
		PresencePenalty:  b.presencePenalty,
		ResponseFormat:   b.responseFormat,
		LogProbs:         b.logProbs,
		TopLogProbs:      b.topLogProbs,
	}
	if len(b.tools) > 0 {
		req.Tools = b.tools
		if b.toolChoice != nil {
			req.ToolChoice = b.toolChoice
		}
	}
	return req, nil
}

// Do sends the chat completion request and returns the full response.
func (b *ChatBuilder) Do() (*ChatResponse, error) {
	return b.DoWithContext(context.Background())
}

// DoWithContext is like Do but honours ctx for cancellation and deadlines.
func (b *ChatBuilder) DoWithContext(ctx context.Context) (*ChatResponse, error) {
	req, err := b.Build()
	if err != nil {
		return nil, err
	}

	data, err := b.client.doRequest(ctx, "POST", "/chat/completions", req)
	if err != nil {
		return nil, err
	}

	var resp ChatResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("ds4: unmarshal response: %w", err)
	}
	return &resp, nil
}

// Stream sends the request with streaming enabled and calls handler for each
// ChatStreamChunk. Return a non-nil error from handler to abort early.
func (b *ChatBuilder) Stream(handler func(ChatStreamChunk) error) error {
	return b.StreamWithContext(context.Background(), handler)
}

// StreamWithContext is like Stream but honours ctx for cancellation and deadlines.
func (b *ChatBuilder) StreamWithContext(ctx context.Context, handler func(ChatStreamChunk) error) error {
	req, err := b.Build()
	if err != nil {
		return err
	}
	req.Stream = true
	if b.includeUsage {
		req.StreamOptions = &StreamOptions{IncludeUsage: true}
	}

	body, err := b.client.doStreamRequest(ctx, "POST", "/chat/completions", req)
	if err != nil {
		return err
	}
	defer body.Close()

	return parseSSE(body, func(data []byte) error {
		var chunk ChatStreamChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			return fmt.Errorf("ds4: unmarshal stream chunk: %w", err)
		}
		return handler(chunk)
	})
}
