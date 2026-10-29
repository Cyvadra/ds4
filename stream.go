package ds4

import (
	"bufio"
	"encoding/json"
	"io"
	"strings"
)

// ChatStreamChunk is a single incremental piece of a streaming chat completion.
type ChatStreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
	// Usage is only populated in the final chunk when stream_options.include_usage is true.
	Usage *Usage `json:"usage"`

	// anthropicContent carries the complete assistant content on the final
	// chunk of an Anthropic stream, for ChatStreamAccumulator.
	anthropicContent json.RawMessage
}

// StreamChoice is one alternative within a streaming chunk.
type StreamChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// Delta holds the incremental content fields for a streaming chunk.
type Delta struct {
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
	ToolCalls        []StreamToolCall `json:"tool_calls,omitempty"`
}

// UnmarshalJSON accepts reasoning fields used by OpenAI-compatible providers.
// reasoning_content remains canonical in the public API.
func (d *Delta) UnmarshalJSON(data []byte) error {
	type deltaAlias Delta
	var wire struct {
		deltaAlias
		Reasoning        string            `json:"reasoning"`
		ReasoningDetails []reasoningDetail `json:"reasoning_details"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*d = Delta(wire.deltaAlias)
	if d.ReasoningContent != "" {
		return nil
	}
	if wire.Reasoning != "" {
		d.ReasoningContent = wire.Reasoning
		return nil
	}
	for _, detail := range wire.ReasoningDetails {
		d.ReasoningContent += detail.Text
	}
	return nil
}

type reasoningDetail struct {
	Text string `json:"text"`
}

// StreamToolCall is an incremental tool call. Index identifies the call when
// multiple tool calls are interleaved across streaming chunks.
type StreamToolCall struct {
	Index    int          `json:"index"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function FunctionCall `json:"function"`
}

// FIMStreamChunk is a single incremental piece of a streaming FIM completion.
type FIMStreamChunk struct {
	ID      string            `json:"id"`
	Object  string            `json:"object"`
	Created int64             `json:"created"`
	Model   string            `json:"model"`
	Choices []FIMStreamChoice `json:"choices"`
}

// FIMStreamChoice is one alternative within a FIM streaming chunk.
type FIMStreamChoice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"`
}

// parseSSE reads an SSE (text/event-stream) response and calls handler for each
// "data: ..." line. It stops when it sees "data: [DONE]" or the reader is exhausted.
func parseSSE(r io.Reader, handler func([]byte) error) error {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			break
		}
		if err := handler([]byte(payload)); err != nil {
			return err
		}
	}
	return scanner.Err()
}

// ChatStreamAccumulator rebuilds the assistant message of a streamed chat
// completion, joining content, reasoning, and tool-call deltas. Feed it every
// chunk, then append Message() to the conversation so a following tool-result
// turn carries the full assistant message, including Anthropic thinking
// signatures.
//
//	var acc ds4.ChatStreamAccumulator
//	err := builder.Stream(func(chunk ds4.ChatStreamChunk) error {
//	    acc.Add(chunk)
//	    return nil
//	})
//	builder.AddMessage(acc.Message())
type ChatStreamAccumulator struct {
	content          strings.Builder
	reasoning        strings.Builder
	toolCalls        []ToolCall
	finishReason     string
	usage            *Usage
	anthropicContent json.RawMessage
}

// Add folds chunk into the accumulated message. Only the first choice is used.
func (a *ChatStreamAccumulator) Add(chunk ChatStreamChunk) {
	if chunk.Usage != nil {
		a.usage = chunk.Usage
	}
	if len(chunk.anthropicContent) > 0 {
		a.anthropicContent = chunk.anthropicContent
	}
	for _, choice := range chunk.Choices {
		if choice.Index != 0 {
			continue
		}
		a.content.WriteString(choice.Delta.Content)
		a.reasoning.WriteString(choice.Delta.ReasoningContent)
		for _, call := range choice.Delta.ToolCalls {
			for len(a.toolCalls) <= call.Index {
				a.toolCalls = append(a.toolCalls, ToolCall{Type: "function"})
			}
			target := &a.toolCalls[call.Index]
			if call.ID != "" {
				target.ID = call.ID
			}
			if call.Type != "" {
				target.Type = call.Type
			}
			target.Function.Name += call.Function.Name
			target.Function.Arguments += call.Function.Arguments
		}
		if choice.FinishReason != nil {
			a.finishReason = *choice.FinishReason
		}
	}
}

// Message returns the accumulated assistant message.
func (a *ChatStreamAccumulator) Message() Message {
	return Message{
		Role:             RoleAssistant,
		Content:          a.content.String(),
		ReasoningContent: a.reasoning.String(),
		ToolCalls:        append([]ToolCall(nil), a.toolCalls...),
		anthropicContent: a.anthropicContent,
	}
}

// FinishReason returns the last finish reason seen in the stream.
func (a *ChatStreamAccumulator) FinishReason() string { return a.finishReason }

// Usage returns the token usage reported by the stream, or nil.
func (a *ChatStreamAccumulator) Usage() *Usage { return a.usage }
