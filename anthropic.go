package ds4

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// DefaultAnthropicMaxTokens is sent as max_tokens to the Anthropic backend
// when MaxTokens is unset, because the Messages API requires the field.
const DefaultAnthropicMaxTokens = 16000

type anthropicRequest struct {
	Model         string                 `json:"model"`
	MaxTokens     int                    `json:"max_tokens"`
	System        []anthropicBlock       `json:"system,omitempty"`
	Messages      []anthropicMessage     `json:"messages"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	Stream        bool                   `json:"stream,omitempty"`
	Thinking      *anthropicThinking     `json:"thinking,omitempty"`
	OutputConfig  *anthropicOutputConfig `json:"output_config,omitempty"`
	Tools         []anthropicTool        `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoice   `json:"tool_choice,omitempty"`
	Metadata      *anthropicMetadata     `json:"metadata,omitempty"`
}

type anthropicMessage struct {
	Role    string            `json:"role"`
	Content []json.RawMessage `json:"content"`
}

// anthropicBlock covers the content block fields ds4 reads and writes.
type anthropicBlock struct {
	Type      string                `json:"type"`
	Text      string                `json:"text,omitempty"`
	Thinking  string                `json:"thinking,omitempty"`
	Signature string                `json:"signature,omitempty"`
	Data      string                `json:"data,omitempty"`
	ID        string                `json:"id,omitempty"`
	Name      string                `json:"name,omitempty"`
	Input     json.RawMessage       `json:"input,omitempty"`
	ToolUseID string                `json:"tool_use_id,omitempty"`
	Content   string                `json:"content,omitempty"`
	Source    *anthropicImageSource `json:"source,omitempty"`
}

type anthropicImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type anthropicThinking struct {
	Type    string `json:"type"`
	Display string `json:"display,omitempty"`
}

type anthropicOutputConfig struct {
	Effort string `json:"effort,omitempty"`
}

type anthropicTool struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	InputSchema any    `json:"input_schema"`
	Strict      bool   `json:"strict,omitempty"`
}

type anthropicToolChoice struct {
	Type string `json:"type"`
	Name string `json:"name,omitempty"`
}

type anthropicMetadata struct {
	UserID string `json:"user_id"`
}

type anthropicResponse struct {
	ID         string          `json:"id"`
	Model      string          `json:"model"`
	Content    json.RawMessage `json:"content"`
	StopReason string          `json:"stop_reason"`
	Usage      anthropicUsage  `json:"usage"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

func (u anthropicUsage) toUsage() Usage {
	prompt := u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	return Usage{
		PromptTokens:          prompt,
		CompletionTokens:      u.OutputTokens,
		TotalTokens:           prompt + u.OutputTokens,
		PromptCacheHitTokens:  u.CacheReadInputTokens,
		PromptCacheMissTokens: u.InputTokens + u.CacheCreationInputTokens,
	}
}

func anthropicFinishReason(stopReason string) string {
	switch stopReason {
	case "max_tokens", "model_context_window_exceeded":
		return FinishReasonLength
	case "tool_use":
		return FinishReasonToolCalls
	case "refusal":
		return FinishReasonContentFilter
	case "":
		return ""
	default:
		return FinishReasonStop
	}
}

// toAnthropicRequest translates a built ChatRequest into a Messages API body.
func toAnthropicRequest(req *ChatRequest) (*anthropicRequest, error) {
	if req.ResponseFormat != nil && req.ResponseFormat.Type == "json_object" {
		return nil, fmt.Errorf("ds4: chat: JSONMode is not supported on the anthropic backend")
	}
	if req.LogProbs {
		return nil, fmt.Errorf("ds4: chat: LogProbs is not supported on the anthropic backend")
	}

	out := &anthropicRequest{
		Model:         req.Model,
		MaxTokens:     req.MaxTokens,
		StopSequences: req.Stop,
	}
	if out.MaxTokens == 0 {
		out.MaxTokens = DefaultAnthropicMaxTokens
	}
	if req.UserID != "" {
		out.Metadata = &anthropicMetadata{UserID: req.UserID}
	}
	if req.Thinking != nil {
		if req.Thinking.Type == "enabled" {
			out.Thinking = &anthropicThinking{Type: "adaptive", Display: "summarized"}
		} else {
			out.Thinking = &anthropicThinking{Type: "disabled"}
		}
	}
	if req.ReasoningEffort != "" {
		out.OutputConfig = &anthropicOutputConfig{Effort: req.ReasoningEffort}
	}
	for _, tool := range req.Tools {
		schema := tool.Function.Parameters
		if schema == nil {
			schema = map[string]any{"type": "object"}
		}
		out.Tools = append(out.Tools, anthropicTool{
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: schema,
			Strict:      tool.Function.Strict,
		})
	}
	if req.ToolChoice != nil {
		choice, err := anthropicToolChoiceFrom(req.ToolChoice)
		if err != nil {
			return nil, err
		}
		out.ToolChoice = choice
	}

	for _, message := range req.Messages {
		if message.Role == RoleSystem {
			out.System = append(out.System, anthropicBlock{Type: "text", Text: message.Content})
			continue
		}
		role, blocks, err := anthropicBlocksFor(message)
		if err != nil {
			return nil, err
		}
		if len(blocks) == 0 {
			continue
		}
		if last := len(out.Messages) - 1; last >= 0 && out.Messages[last].Role == role {
			out.Messages[last].Content = append(out.Messages[last].Content, blocks...)
			continue
		}
		out.Messages = append(out.Messages, anthropicMessage{Role: role, Content: blocks})
	}
	return out, nil
}

func anthropicBlocksFor(message Message) (string, []json.RawMessage, error) {
	var blocks []anthropicBlock
	role := RoleUser
	switch message.Role {
	case RoleUser:
		for _, image := range message.imageInputs {
			blocks = append(blocks, anthropicBlock{Type: "image", Source: anthropicImageSourceFor(image.URL)})
		}
		if message.Content != "" {
			blocks = append(blocks, anthropicBlock{Type: "text", Text: message.Content})
		}
	case RoleTool:
		blocks = append(blocks, anthropicBlock{Type: "tool_result", ToolUseID: message.ToolCallID, Content: message.Content})
	case RoleAssistant:
		role = RoleAssistant
		if len(message.anthropicContent) > 0 {
			var raw []json.RawMessage
			if err := json.Unmarshal(message.anthropicContent, &raw); err != nil {
				return "", nil, fmt.Errorf("ds4: chat: replay assistant content: %w", err)
			}
			return role, raw, nil
		}
		if message.Content != "" {
			blocks = append(blocks, anthropicBlock{Type: "text", Text: message.Content})
		}
		for _, call := range message.ToolCalls {
			input := json.RawMessage(call.Function.Arguments)
			if strings.TrimSpace(call.Function.Arguments) == "" {
				input = json.RawMessage("{}")
			} else if !json.Valid(input) {
				return "", nil, fmt.Errorf("ds4: chat: tool call %q has invalid JSON arguments", call.ID)
			}
			blocks = append(blocks, anthropicBlock{Type: "tool_use", ID: call.ID, Name: call.Function.Name, Input: input})
		}
	default:
		return "", nil, fmt.Errorf("ds4: chat: role %q is not supported on the anthropic backend", message.Role)
	}

	raw := make([]json.RawMessage, 0, len(blocks))
	for _, block := range blocks {
		data, err := json.Marshal(block)
		if err != nil {
			return "", nil, fmt.Errorf("ds4: chat: marshal content block: %w", err)
		}
		raw = append(raw, data)
	}
	return role, raw, nil
}

// anthropicImageSourceFor converts an image URL, or the Base64 data URL built
// by WithImageBase64, into an Anthropic image source.
func anthropicImageSourceFor(imageURL string) *anthropicImageSource {
	if rest, ok := strings.CutPrefix(imageURL, "data:"); ok {
		if mediaType, data, ok := strings.Cut(rest, ";base64,"); ok {
			return &anthropicImageSource{Type: "base64", MediaType: mediaType, Data: data}
		}
	}
	return &anthropicImageSource{Type: "url", URL: imageURL}
}

func anthropicToolChoiceFrom(choice any) (*anthropicToolChoice, error) {
	switch value := choice.(type) {
	case string:
		switch value {
		case "auto", "none":
			return &anthropicToolChoice{Type: value}, nil
		case "required":
			return &anthropicToolChoice{Type: "any"}, nil
		}
	case map[string]any:
		if function, ok := value["function"].(map[string]any); ok {
			if name, ok := function["name"].(string); ok && name != "" {
				return &anthropicToolChoice{Type: "tool", Name: name}, nil
			}
		}
	}
	return nil, fmt.Errorf("ds4: chat: unsupported ToolChoice %v for the anthropic backend", choice)
}

// fromAnthropicResponse converts a Messages API response into a ChatResponse.
func fromAnthropicResponse(resp *anthropicResponse) (*ChatResponse, error) {
	var blocks []anthropicBlock
	if len(resp.Content) > 0 {
		if err := json.Unmarshal(resp.Content, &blocks); err != nil {
			return nil, fmt.Errorf("ds4: unmarshal response content: %w", err)
		}
	}
	message := messageFromAnthropicBlocks(blocks)
	message.anthropicContent = resp.Content
	return &ChatResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   resp.Model,
		Choices: []Choice{{
			Message:      message,
			FinishReason: anthropicFinishReason(resp.StopReason),
		}},
		Usage: resp.Usage.toUsage(),
	}, nil
}

func messageFromAnthropicBlocks(blocks []anthropicBlock) Message {
	message := Message{Role: RoleAssistant}
	var content, reasoning strings.Builder
	for _, block := range blocks {
		switch block.Type {
		case "text":
			content.WriteString(block.Text)
		case "thinking":
			reasoning.WriteString(block.Thinking)
		case "tool_use":
			arguments := string(block.Input)
			if arguments == "" {
				arguments = "{}"
			}
			message.ToolCalls = append(message.ToolCalls, ToolCall{
				ID:       block.ID,
				Type:     "function",
				Function: FunctionCall{Name: block.Name, Arguments: arguments},
			})
		}
	}
	message.Content = content.String()
	message.ReasoningContent = reasoning.String()
	return message
}

func (b *ChatBuilder) doAnthropic(ctx context.Context, target backend, req *ChatRequest) (*ChatResponse, error) {
	body, err := toAnthropicRequest(req)
	if err != nil {
		return nil, err
	}
	data, err := b.client.doRequest(ctx, target, "POST", "/v1/messages", body)
	if err != nil {
		return nil, err
	}
	var resp anthropicResponse
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, fmt.Errorf("ds4: unmarshal response: %w", err)
	}
	return fromAnthropicResponse(&resp)
}

// anthropicStreamEvent covers every Messages streaming event payload.
type anthropicStreamEvent struct {
	Type         string            `json:"type"`
	Index        int               `json:"index"`
	Message      anthropicResponse `json:"message"`
	ContentBlock anthropicBlock    `json:"content_block"`
	Delta        struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage *anthropicUsage `json:"usage"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

type anthropicStreamBlock struct {
	block     anthropicBlock
	toolIndex int
	input     strings.Builder
}

func (b *ChatBuilder) streamAnthropic(ctx context.Context, target backend, req *ChatRequest, handler func(ChatStreamChunk) error) error {
	body, err := toAnthropicRequest(req)
	if err != nil {
		return err
	}
	body.Stream = true
	stream, err := b.client.doStreamRequest(ctx, target, "POST", "/v1/messages", body)
	if err != nil {
		return err
	}
	defer stream.Close()

	var (
		id, model  string
		usage      anthropicUsage
		blocks     = map[int]*anthropicStreamBlock{}
		order      []int
		toolCalls  int
		created    = time.Now().Unix()
		emitDeltas = func(delta Delta, finish *string, chunkUsage *Usage, content json.RawMessage) error {
			return handler(ChatStreamChunk{
				ID:               id,
				Object:           "chat.completion.chunk",
				Created:          created,
				Model:            model,
				Choices:          []StreamChoice{{Delta: delta, FinishReason: finish}},
				Usage:            chunkUsage,
				anthropicContent: content,
			})
		}
	)

	return parseSSE(stream, func(data []byte) error {
		var event anthropicStreamEvent
		if err := json.Unmarshal(data, &event); err != nil {
			return fmt.Errorf("ds4: unmarshal stream event: %w", err)
		}
		switch event.Type {
		case "message_start":
			id, model, usage = event.Message.ID, event.Message.Model, event.Message.Usage
			return emitDeltas(Delta{Role: RoleAssistant}, nil, nil, nil)

		case "content_block_start":
			state := &anthropicStreamBlock{block: event.ContentBlock, toolIndex: -1}
			blocks[event.Index] = state
			order = append(order, event.Index)
			switch event.ContentBlock.Type {
			case "tool_use":
				state.toolIndex = toolCalls
				toolCalls++
				return emitDeltas(Delta{ToolCalls: []StreamToolCall{{
					Index:    state.toolIndex,
					ID:       event.ContentBlock.ID,
					Type:     "function",
					Function: FunctionCall{Name: event.ContentBlock.Name},
				}}}, nil, nil, nil)
			case "text":
				if event.ContentBlock.Text != "" {
					return emitDeltas(Delta{Content: event.ContentBlock.Text}, nil, nil, nil)
				}
			}

		case "content_block_delta":
			state := blocks[event.Index]
			if state == nil {
				return nil
			}
			switch event.Delta.Type {
			case "text_delta":
				state.block.Text += event.Delta.Text
				return emitDeltas(Delta{Content: event.Delta.Text}, nil, nil, nil)
			case "thinking_delta":
				state.block.Thinking += event.Delta.Thinking
				return emitDeltas(Delta{ReasoningContent: event.Delta.Thinking}, nil, nil, nil)
			case "signature_delta":
				state.block.Signature += event.Delta.Signature
			case "input_json_delta":
				state.input.WriteString(event.Delta.PartialJSON)
				return emitDeltas(Delta{ToolCalls: []StreamToolCall{{
					Index:    state.toolIndex,
					Function: FunctionCall{Arguments: event.Delta.PartialJSON},
				}}}, nil, nil, nil)
			}

		case "message_delta":
			if event.Usage != nil {
				usage.OutputTokens = event.Usage.OutputTokens
			}
			content, err := assembleAnthropicContent(blocks, order)
			if err != nil {
				return err
			}
			finish := anthropicFinishReason(event.Delta.StopReason)
			var chunkUsage *Usage
			if b.includeUsage {
				converted := usage.toUsage()
				chunkUsage = &converted
			}
			return emitDeltas(Delta{}, &finish, chunkUsage, content)

		case "error":
			return &APIError{Message: event.Error.Message, Type: event.Error.Type}
		}
		return nil
	})
}

func assembleAnthropicContent(blocks map[int]*anthropicStreamBlock, order []int) (json.RawMessage, error) {
	raw := make([]map[string]any, 0, len(order))
	for _, index := range order {
		state := blocks[index]
		block := state.block
		switch block.Type {
		case "text":
			raw = append(raw, map[string]any{"type": "text", "text": block.Text})
		case "thinking":
			raw = append(raw, map[string]any{"type": "thinking", "thinking": block.Thinking, "signature": block.Signature})
		case "redacted_thinking":
			raw = append(raw, map[string]any{"type": "redacted_thinking", "data": block.Data})
		case "tool_use":
			input := json.RawMessage(bytes.TrimSpace([]byte(state.input.String())))
			if len(input) == 0 || !json.Valid(input) {
				input = json.RawMessage("{}")
			}
			raw = append(raw, map[string]any{"type": "tool_use", "id": block.ID, "name": block.Name, "input": input})
		}
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return nil, fmt.Errorf("ds4: marshal stream content: %w", err)
	}
	return data, nil
}
