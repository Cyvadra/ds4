package ds4

import (
	"bufio"
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
}

// StreamChoice is one alternative within a streaming chunk.
type StreamChoice struct {
	Index        int     `json:"index"`
	Delta        Delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"`
}

// Delta holds the incremental content fields for a streaming chunk.
type Delta struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
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
