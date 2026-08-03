// Package ds4 provides a fluent Go client for the DeepSeek API.
//
// Basic usage:
//
//	client := ds4.New("YOUR_API_KEY")
//
//	resp, err := client.Chat().
//	    System("You are a helpful assistant.").
//	    User("Hello!").
//	    Do()
//	fmt.Println(resp.Content())
package ds4

// Model name constants.
const (
	ModelDeepSeekV4Flash  = "deepseek-v4-flash"
	ModelDeepSeekChat     = "deepseek-chat"
	ModelDeepSeekReasoner = "deepseek-reasoner"
	ModelDeepSeekCoder    = "deepseek-coder"
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
