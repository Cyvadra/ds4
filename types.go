// Package ds4 provides a fluent Go client for the DeepSeek API.
//
// Basic usage:
//
//	client := ds4.New("YOUR_API_KEY")
//
//	resp, err := client.Chat().
//	    User("Hello!").
//	    Do()
//	fmt.Println(resp.Content())
package ds4

import "encoding/json"

// Model name constants accepted by the official DeepSeek API.
const (
	ModelDeepSeekV4Flash          = "deepseek-v4-flash"
	ModelDeepSeekV4Pro            = "deepseek-v4-pro"
	ModelDeepSeekV4FlashVisionExp = "deepseek-v4-flash-vision-exp"
)

// Image detail levels accepted by DeepSeek vision input. ImageDetailLow resizes
// an image to 512x512 before inference, reducing latency and token use. The
// other values preserve the original image; ImageDetailAuto currently behaves
// like ImageDetailOriginal.
const (
	ImageDetailLow      = "low"
	ImageDetailHigh     = "high"
	ImageDetailOriginal = "original"
	ImageDetailAuto     = "auto"
)

// Reasoning effort levels accepted by ReasoningEffort.
//
// deepseek-v4-flash honours all three. deepseek-v4-pro currently maps
// EffortLow up to EffortHigh.
const (
	EffortLow  = "low"
	EffortHigh = "high" // API default
	EffortMax  = "max"
)

// Message role constants.
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Finish reason constants returned in Choice.FinishReason.
const (
	FinishReasonStop          = "stop"
	FinishReasonLength        = "length"
	FinishReasonToolCalls     = "tool_calls"
	FinishReasonContentFilter = "content_filter"
)

// Message represents a single message in a conversation.
type Message struct {
	Role             string     `json:"role"`
	Content          string     `json:"content"`
	ReasoningContent string     `json:"reasoning_content,omitempty"`
	ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string     `json:"tool_call_id,omitempty"`
	Name             string     `json:"name,omitempty"`
	// Prefix, when true on an assistant message, tells the model to continue
	// generating from this content (chat prefix completion, beta).
	Prefix bool `json:"prefix,omitempty"`

	imageInputs []imageInput
}

type imageInput struct {
	URL    string
	Detail string
}

type messageContentPart struct {
	Type     string        `json:"type"`
	Text     string        `json:"text"`
	ImageURL *imageURLPart `json:"image_url,omitempty"`
}

type imageURLPart struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// MarshalJSON preserves the standard string content representation for text
// messages. Internal image inputs are emitted as OpenAI-compatible content
// blocks only for messages built with ChatBuilder image attachment methods.
func (m Message) MarshalJSON() ([]byte, error) {
	type messageAlias Message
	if len(m.imageInputs) == 0 {
		return json.Marshal(messageAlias(m))
	}

	parts := make([]messageContentPart, 0, len(m.imageInputs)+1)
	parts = append(parts, messageContentPart{Type: "text", Text: m.Content})
	for _, image := range m.imageInputs {
		parts = append(parts, messageContentPart{
			Type: "image_url",
			ImageURL: &imageURLPart{
				URL:    image.URL,
				Detail: image.Detail,
			},
		})
	}

	return json.Marshal(struct {
		Role             string     `json:"role"`
		Content          any        `json:"content"`
		ReasoningContent string     `json:"reasoning_content,omitempty"`
		ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
		ToolCallID       string     `json:"tool_call_id,omitempty"`
		Name             string     `json:"name,omitempty"`
		Prefix           bool       `json:"prefix,omitempty"`
	}{
		Role:             m.Role,
		Content:          parts,
		ReasoningContent: m.ReasoningContent,
		ToolCalls:        m.ToolCalls,
		ToolCallID:       m.ToolCallID,
		Name:             m.Name,
		Prefix:           m.Prefix,
	})
}

// Tool represents a callable tool that the model can invoke.
type Tool struct {
	Type     string   `json:"type"`
	Function Function `json:"function"`
}

// Function defines a tool's callable function.
type Function struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Strict      bool   `json:"strict,omitempty"`
	// Parameters should be a JSON Schema object describing the function's parameters.
	Parameters any `json:"parameters,omitempty"`
}

// ToolCall is a tool invocation returned by the model.
type ToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function FunctionCall `json:"function"`
}

// FunctionCall holds the name and JSON-encoded arguments of a tool invocation.
type FunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// ResponseFormat specifies the format of the model's output.
type ResponseFormat struct {
	// Type is "text" (default) or "json_object".
	Type string `json:"type"`
}

// Thinking controls whether the model emits reasoning_content.
// Type must be "enabled" or "disabled". Thinking is enabled by default
// server-side, so omitting this field still yields reasoning output.
type Thinking struct {
	Type string `json:"type"`
}

// Usage holds token usage statistics for a request.
type Usage struct {
	PromptTokens          int `json:"prompt_tokens"`
	CompletionTokens      int `json:"completion_tokens"`
	TotalTokens           int `json:"total_tokens"`
	PromptCacheHitTokens  int `json:"prompt_cache_hit_tokens"`
	PromptCacheMissTokens int `json:"prompt_cache_miss_tokens"`
}

// StreamOptions controls optional streaming behaviour.
type StreamOptions struct {
	// IncludeUsage, when true, appends a final chunk containing token usage.
	IncludeUsage bool `json:"include_usage"`
}

// NewFunction is a convenience constructor that creates a Tool of type "function".
//
//	schema := map[string]any{
//	    "type": "object",
//	    "properties": map[string]any{
//	        "city": map[string]any{"type": "string"},
//	    },
//	    "required": []string{"city"},
//	}
//	tool := ds4.NewFunction("get_weather", "Get current weather", schema)
func NewFunction(name, description string, parameters any) Tool {
	return Tool{
		Type: "function",
		Function: Function{
			Name:        name,
			Description: description,
			Parameters:  parameters,
		},
	}
}

// NewStrictFunction creates a function tool that asks the beta API to enforce
// its JSON Schema when generating tool call arguments.
func NewStrictFunction(name, description string, parameters any) Tool {
	tool := NewFunction(name, description, parameters)
	tool.Function.Strict = true
	return tool
}
