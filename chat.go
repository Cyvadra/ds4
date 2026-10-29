package ds4

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"math"
	"net/url"
)

// ChatRequest is the JSON body sent to POST /chat/completions.
type ChatRequest struct {
	Model           string          `json:"model"`
	Messages        []Message       `json:"messages"`
	MaxTokens       int             `json:"max_tokens,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"top_p,omitempty"`
	Stream          bool            `json:"stream,omitempty"`
	StreamOptions   *StreamOptions  `json:"stream_options,omitempty"`
	Stop            []string        `json:"stop,omitempty"`
	ResponseFormat  *ResponseFormat `json:"response_format,omitempty"`
	Thinking        *Thinking       `json:"thinking,omitempty"`
	ReasoningEffort string          `json:"reasoning_effort,omitempty"`
	Tools           []Tool          `json:"tools,omitempty"`
	ToolChoice      any             `json:"tool_choice,omitempty"`
	LogProbs        bool            `json:"logprobs,omitempty"`
	TopLogProbs     int             `json:"top_logprobs,omitempty"`
	UserID          string          `json:"user_id,omitempty"`
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

// LogProbs returns log-probability data for the first choice, or nil when the
// request did not enable logprobs.
func (r *ChatResponse) LogProbs() *LogProbs {
	if len(r.Choices) == 0 {
		return nil
	}
	return r.Choices[0].LogProbs
}

// Choice is one completion alternative inside a ChatResponse.
type Choice struct {
	Index        int       `json:"index"`
	Message      Message   `json:"message"`
	FinishReason string    `json:"finish_reason"`
	LogProbs     *LogProbs `json:"logprobs,omitempty"`
}

// LogProbs holds log-probability data for the generated tokens of a choice.
type LogProbs struct {
	Content []TokenLogProb `json:"content"`
}

// TokenLogProb is the log probability of a single generated token, along with
// the most likely alternatives at that position.
type TokenLogProb struct {
	Token       string       `json:"token"`
	LogProb     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes,omitempty"`
	TopLogProbs []TopLogProb `json:"top_logprobs,omitempty"`
}

// TopLogProb is one alternative token considered at a given position.
type TopLogProb struct {
	Token   string  `json:"token"`
	LogProb float64 `json:"logprob"`
	Bytes   []int   `json:"bytes,omitempty"`
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

	model           string
	messages        []Message
	maxTokens       int
	continueTokens  int
	continueRequest bool
	temperature     *float64
	topP            *float64
	stop            []string
	responseFormat  *ResponseFormat
	thinking        *Thinking
	reasoningEffort string
	tools           []Tool
	toolChoice      any
	logProbs        bool
	topLogProbs     int
	includeUsage    bool
	userID          string
	images          []imageAttachment
}

type imageAttachment struct {
	url        string
	data       []byte
	mediaType  string
	detail     string
	detailArgs int
}

const (
	maxInlineImageBytes = 32 << 20
	maxImagesPerRequest = 600
	maxImageURLLength   = 8192
	maxRequestBodyBytes = 48 << 20
)

func newChatBuilder(c *Client) *ChatBuilder {
	return &ChatBuilder{
		client:   c,
		model:    ModelDeepSeekV4Flash,
		messages: []Message{{Role: RoleSystem, Content: c.systemPrompt}},
	}
}

// Model overrides the model used for this request (default: ModelDeepSeekV4Flash).
func (b *ChatBuilder) Model(model string) *ChatBuilder {
	b.model = model
	return b
}

// System replaces the conversation's system prompt. If the message history
// contains no system message, it appends one.
func (b *ChatBuilder) System(content string) *ChatBuilder {
	for index, message := range b.messages {
		if message.Role == RoleSystem {
			b.messages[index].Content = content
			return b
		}
	}
	b.messages = append(b.messages, Message{Role: RoleSystem, Content: content})
	return b
}

// SystemAppend appends an additional system message without replacing the
// conversation's first system prompt.
func (b *ChatBuilder) SystemAppend(content string) *ChatBuilder {
	b.messages = append(b.messages, Message{Role: RoleSystem, Content: content})
	return b
}

// User appends a user message to the conversation.
func (b *ChatBuilder) User(content string) *ChatBuilder {
	b.messages = append(b.messages, Message{Role: RoleUser, Content: content})
	return b
}

// WithImageURL attaches a publicly accessible HTTP(S) image URL to this
// request. The image is sent with the final user message when Build, Do, or
// Stream is called, regardless of where this method appears in the chain. Each
// call appends another image, so URL and Base64 images may be freely mixed.
//
// detail is optional and may be ImageDetailLow, ImageDetailHigh,
// ImageDetailOriginal, or ImageDetailAuto. Omitting it uses the API default.
// ImageDetailLow scales the image to 512x512 before inference for lower
// latency and token usage; choose original/high for OCR or fine detail.
//
// Adding an image automatically selects ModelDeepSeekV4FlashVisionExp, even
// when Model was called with a different model. Build returns an error unless
// the conversation has a user message, and validates the URL and detail value.
//
//	resp, err := client.Chat().
//	    User("Describe this image.").
//	    WithImageURL("https://example.com/photo.jpg", ds4.ImageDetailLow).
//	    Do()
func (b *ChatBuilder) WithImageURL(rawURL string, detail ...string) *ChatBuilder {
	b.images = append(b.images, imageAttachment{
		url:        rawURL,
		detail:     optionalImageDetail(detail),
		detailArgs: len(detail),
	})
	return b
}

// WithImageBase64 attaches raw image bytes to this request as a Base64 data
// URL. mediaType must be image/jpeg, image/png, image/gif, or image/webp; the
// package performs Base64 encoding during Build. The image is sent with the
// final user message regardless of this method's position in the chain.
//
// Repeated calls support multiple images and may be mixed with WithImageURL.
// detail is optional and accepts the ImageDetail constants; omit it to use the
// API default. Adding an image automatically switches the request model to
// ModelDeepSeekV4FlashVisionExp. Build validates the image, MIME type, detail,
// and the DeepSeek 32 MiB per-inline-image and 600-image request limits.
//
//	data, err := os.ReadFile("screenshot.png")
//	if err != nil { return err }
//	resp, err := client.Chat().
//	    User("Read the text in this screenshot.").
//	    WithImageBase64(data, "image/png", ds4.ImageDetailOriginal).
//	    Do()
func (b *ChatBuilder) WithImageBase64(data []byte, mediaType string, detail ...string) *ChatBuilder {
	imageData := make([]byte, len(data))
	copy(imageData, data)
	b.images = append(b.images, imageAttachment{
		data:       imageData,
		mediaType:  mediaType,
		detail:     optionalImageDetail(detail),
		detailArgs: len(detail),
	})
	return b
}

func optionalImageDetail(details []string) string {
	if len(details) == 0 {
		return ""
	}
	return details[0]
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

// AppendResponse appends the first assistant message from resp to the
// conversation history. It preserves tool_calls and reasoning_content for a
// subsequent tool-result or user turn.
func (b *ChatBuilder) AppendResponse(resp *ChatResponse) *ChatBuilder {
	if message := resp.FirstMessage(); message != nil {
		b.messages = append(b.messages, *message)
	}
	return b
}

// WithMessages replaces the entire message history.
func (b *ChatBuilder) WithMessages(msgs []Message) *ChatBuilder {
	b.messages = make([]Message, len(msgs))
	copy(b.messages, msgs)
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

// Continue resumes generation from the final assistant message. It enables
// prefix completion and adds extraTokens to the request's MaxTokens value.
// Build returns an error unless the final message has the assistant role.
func (b *ChatBuilder) Continue(extraTokens int) *ChatBuilder {
	b.continueRequest = true
	b.continueTokens = extraTokens
	if len(b.messages) > 0 {
		b.messages[len(b.messages)-1].Prefix = true
	}
	return b
}

// ThinkingMode enables thinking on the currently selected model, so the
// response carries a chain-of-thought in ReasoningContent alongside the final
// answer in Content. Thinking is already enabled by default server-side; use
// Thinking(false) to turn it off.
func (b *ChatBuilder) ThinkingMode() *ChatBuilder {
	b.thinking = &Thinking{Type: "enabled"}
	return b
}

// Thinking sets the thinking-mode toggle. Enabled thinking returns
// reasoning_content alongside the final content.
func (b *ChatBuilder) Thinking(enabled bool) *ChatBuilder {
	typeName := "disabled"
	if enabled {
		typeName = "enabled"
	}
	b.thinking = &Thinking{Type: typeName}
	return b
}

// ReasoningEffort sets the requested thinking effort: EffortLow, EffortHigh,
// or EffortMax. The compatibility values "medium" and "xhigh" are also
// accepted and mapped by the API.
func (b *ChatBuilder) ReasoningEffort(effort string) *ChatBuilder {
	b.reasoningEffort = effort
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

// UserID tags the request with a caller-side identifier used for content-safety
// review, KVCache isolation, and scheduling isolation. It must not contain
// personal information.
func (b *ChatBuilder) UserID(id string) *ChatBuilder {
	b.userID = id
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
// each generated position. topN must be between 0 and 20. Read the result via
// ChatResponse.LogProbs.
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
	images := make([]imageAttachment, len(b.images))
	for index, image := range b.images {
		images[index] = image
		images[index].data = append([]byte(nil), image.data...)
	}

	cp := *b
	cp.messages = msgs
	cp.tools = tools
	cp.images = images
	return &cp
}

// Build constructs the ChatRequest from the builder state without sending it.
// Returns an error when the request would be invalid (e.g. no messages).
func (b *ChatBuilder) Build() (*ChatRequest, error) {
	if len(b.messages) == 0 {
		return nil, fmt.Errorf("ds4: chat: at least one message is required")
	}
	if err := validateChatRequest(b); err != nil {
		return nil, err
	}

	messages := append([]Message(nil), b.messages...)
	model := b.model
	if len(b.images) > 0 {
		finalUser := -1
		for index := len(messages) - 1; index >= 0; index-- {
			if messages[index].Role == RoleUser {
				finalUser = index
				break
			}
		}
		if finalUser == -1 {
			return nil, fmt.Errorf("ds4: chat: image attachments require a user message")
		}
		messages[finalUser].imageInputs = makeImageInputs(b.images)
		model = ModelDeepSeekV4FlashVisionExp
	}

	req := &ChatRequest{
		Model:           model,
		Messages:        messages,
		MaxTokens:       b.maxTokens + b.continueTokens,
		Temperature:     b.temperature,
		TopP:            b.topP,
		Stop:            b.stop,
		ResponseFormat:  b.responseFormat,
		LogProbs:        b.logProbs,
		TopLogProbs:     b.topLogProbs,
		UserID:          b.userID,
		Thinking:        b.thinking,
		ReasoningEffort: b.reasoningEffort,
	}
	if len(b.tools) > 0 {
		req.Tools = b.tools
		if b.toolChoice != nil {
			req.ToolChoice = b.toolChoice
		}
	}
	if len(b.images) > 0 {
		body, err := json.Marshal(req)
		if err != nil {
			return nil, fmt.Errorf("ds4: chat: marshal image request: %w", err)
		}
		if len(body) > maxRequestBodyBytes {
			return nil, fmt.Errorf("ds4: chat: image request body must not exceed 48 MiB")
		}
	}
	return req, nil
}

// validEfforts holds every value the API accepts, including the two
// compatibility aliases it maps onto EffortHigh and EffortMax.
var validEfforts = map[string]struct{}{
	EffortLow:  {},
	EffortHigh: {},
	EffortMax:  {},
	"medium":   {},
	"xhigh":    {},
}

func validateChatRequest(b *ChatBuilder) error {
	if b.continueRequest {
		if len(b.messages) == 0 || b.messages[len(b.messages)-1].Role != RoleAssistant {
			return fmt.Errorf("ds4: chat: Continue requires the final message to have assistant role")
		}
	}
	for index, message := range b.messages {
		if message.Prefix && (index != len(b.messages)-1 || message.Role != RoleAssistant) {
			return fmt.Errorf("ds4: chat: prefix must be set only on the final assistant message")
		}
	}
	if b.maxTokens < 0 || b.continueTokens < 0 {
		return fmt.Errorf("ds4: chat: MaxTokens must not be negative")
	}
	if err := validateRange("Temperature", b.temperature, 0, 2); err != nil {
		return err
	}
	if err := validateRange("TopP", b.topP, 0, 1); err != nil {
		return err
	}
	if b.reasoningEffort != "" {
		if _, ok := validEfforts[b.reasoningEffort]; !ok {
			return fmt.Errorf("ds4: chat: ReasoningEffort must be low, high, or max")
		}
	}
	if b.topLogProbs < 0 || b.topLogProbs > 20 {
		return fmt.Errorf("ds4: chat: LogProbs topN must be between 0 and 20")
	}
	if err := validateImageAttachments(b.images); err != nil {
		return err
	}
	return validateUserID(b.userID)
}

func validateImageAttachments(images []imageAttachment) error {
	if len(images) > maxImagesPerRequest {
		return fmt.Errorf("ds4: chat: at most %d images may be attached to one request", maxImagesPerRequest)
	}
	for _, image := range images {
		if image.detailArgs > 1 {
			return fmt.Errorf("ds4: chat: image detail accepts at most one value")
		}
		if image.detail != "" && !validImageDetail(image.detail) {
			return fmt.Errorf("ds4: chat: image detail must be low, high, original, or auto")
		}
		if image.url != "" {
			if len(image.url) > maxImageURLLength {
				return fmt.Errorf("ds4: chat: image URL must not exceed 8192 characters")
			}
			parsed, err := url.ParseRequestURI(image.url)
			if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
				return fmt.Errorf("ds4: chat: image URL must be an absolute http or https URL")
			}
			continue
		}
		if len(image.data) == 0 {
			return fmt.Errorf("ds4: chat: image data must not be empty")
		}
		if len(image.data) > maxInlineImageBytes {
			return fmt.Errorf("ds4: chat: inline image must not exceed 32 MiB")
		}
		if !validImageMediaType(image.mediaType) {
			return fmt.Errorf("ds4: chat: image media type must be image/jpeg, image/png, image/gif, or image/webp")
		}
	}
	return nil
}

func validImageDetail(detail string) bool {
	switch detail {
	case ImageDetailLow, ImageDetailHigh, ImageDetailOriginal, ImageDetailAuto:
		return true
	default:
		return false
	}
}

func validImageMediaType(mediaType string) bool {
	switch mediaType {
	case "image/jpeg", "image/png", "image/gif", "image/webp":
		return true
	default:
		return false
	}
}

func makeImageInputs(images []imageAttachment) []imageInput {
	inputs := make([]imageInput, 0, len(images))
	for _, image := range images {
		imageURL := image.url
		if imageURL == "" {
			imageURL = "data:" + image.mediaType + ";base64," + base64.StdEncoding.EncodeToString(image.data)
		}
		inputs = append(inputs, imageInput{URL: imageURL, Detail: image.detail})
	}
	return inputs
}

func validateUserID(id string) error {
	if id == "" {
		return nil
	}
	if len(id) > 512 {
		return fmt.Errorf("ds4: chat: UserID must not exceed 512 characters")
	}
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
		default:
			return fmt.Errorf("ds4: chat: UserID may only contain letters, digits, '-' and '_'")
		}
	}
	return nil
}

func validateRange(name string, value *float64, minimum, maximum float64) error {
	if value != nil && (math.IsNaN(*value) || *value < minimum || *value > maximum) {
		return fmt.Errorf("ds4: chat: %s must be between %g and %g", name, minimum, maximum)
	}
	return nil
}

// Do sends the chat completion request and returns the full response.
func (b *ChatBuilder) Do() (*ChatResponse, error) {
	return b.DoWithContext(context.Background())
}

// DoWithContext is like Do but honours ctx for cancellation and deadlines.
func (b *ChatBuilder) DoWithContext(ctx context.Context) (*ChatResponse, error) {
	req, target, err := b.prepare(ctx)
	if err != nil {
		return nil, err
	}
	if target.kind == backendAnthropic {
		return b.doAnthropic(ctx, target, req)
	}

	data, err := b.client.doRequest(ctx, target, "POST", b.endpoint(target), req)
	if err != nil {
		return nil, err
	}

	var resp ChatResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("ds4: unmarshal response: %w", err)
	}
	if target.kind == backendLocal {
		// Local servers send reasoning inline in content; restore the split.
		splitLocalReasoning(&resp)
	}
	return &resp, nil
}

// prepare builds the request, resolves its backend, and applies the
// per-backend adjustments.
func (b *ChatBuilder) prepare(ctx context.Context) (*ChatRequest, backend, error) {
	req, err := b.Build()
	if err != nil {
		return nil, backend{}, err
	}
	if len(b.images) > 0 && req.Model != b.model {
		// Images select the DeepSeek vision model unless the chosen model is
		// served by the Anthropic backend, which accepts images natively.
		if target, err := b.client.resolveBackend(ctx, b.model); err == nil && target.kind == backendAnthropic {
			req.Model = b.model
			return req, target, nil
		}
	}
	target, err := b.client.resolveBackend(ctx, req.Model)
	if err != nil {
		return nil, backend{}, err
	}
	if target.kind == backendLocal {
		req.Thinking = nil
		req.ReasoningEffort = ""
	}
	return req, target, nil
}

// Stream sends the request with streaming enabled and calls handler for each
// ChatStreamChunk. Return a non-nil error from handler to abort early.
func (b *ChatBuilder) Stream(handler func(ChatStreamChunk) error) error {
	return b.StreamWithContext(context.Background(), handler)
}

// StreamWithContext is like Stream but honours ctx for cancellation and deadlines.
func (b *ChatBuilder) StreamWithContext(ctx context.Context, handler func(ChatStreamChunk) error) error {
	req, target, err := b.prepare(ctx)
	if err != nil {
		return err
	}
	if target.kind == backendAnthropic {
		return b.streamAnthropic(ctx, target, req, handler)
	}
	req.Stream = true
	if b.includeUsage {
		req.StreamOptions = &StreamOptions{IncludeUsage: true}
	}

	body, err := b.client.doStreamRequest(ctx, target, "POST", b.endpoint(target), req)
	if err != nil {
		return err
	}
	defer body.Close()

	var splitters []*reasoningSplitter
	if err := parseSSE(body, func(data []byte) error {
		var chunk ChatStreamChunk
		if err := json.Unmarshal(data, &chunk); err != nil {
			return fmt.Errorf("ds4: unmarshal stream chunk: %w", err)
		}
		if target.kind == backendLocal {
			splitters = applyReasoningSplit(&chunk, splitters)
		}
		return handler(chunk)
	}); err != nil {
		return err
	}
	if target.kind == backendLocal {
		return flushReasoningSplit(splitters, handler)
	}
	return nil
}

func (b *ChatBuilder) endpoint(target backend) string {
	if target.kind == backendLocal {
		// Local servers have no /beta route; strict tools and chat-prefix
		// requests fall back to the ordinary chat endpoint.
		return "/chat/completions"
	}
	for _, message := range b.messages {
		if message.Prefix {
			return "/beta/chat/completions"
		}
	}
	for _, tool := range b.tools {
		if tool.Function.Strict {
			return "/beta/chat/completions"
		}
	}
	return "/chat/completions"
}
