package ds4_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"github.com/Cyvadra/ds4"
)

// ---------------------------------------------------------------------------
// Builder unit tests – no network required
// ---------------------------------------------------------------------------

func TestChatBuilder_BuildRequiresMessages(t *testing.T) {
	b := ds4.New("key").Chat()
	_, err := b.Build()
	if err == nil {
		t.Fatal("expected error when no messages provided")
	}
}

func TestChatBuilder_BuildBasic(t *testing.T) {
	req, err := ds4.New("key").Chat().
		System("You are helpful.").
		User("Hello").
		Build()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Model != ds4.ModelDeepSeekV4Flash {
		t.Errorf("expected default model %q, got %q", ds4.ModelDeepSeekV4Flash, req.Model)
	}
	if len(req.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(req.Messages))
	}
	if req.Messages[0].Role != ds4.RoleSystem {
		t.Errorf("expected system role, got %q", req.Messages[0].Role)
	}
	if req.Messages[1].Role != ds4.RoleUser {
		t.Errorf("expected user role, got %q", req.Messages[1].Role)
	}
}

func TestChatBuilder_Model(t *testing.T) {
	req, err := ds4.New("key").Chat().
		Model(ds4.ModelDeepSeekChat).
		User("Hi").
		Build()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Model != ds4.ModelDeepSeekChat {
		t.Errorf("expected model %q, got %q", ds4.ModelDeepSeekChat, req.Model)
	}
}

func TestChatBuilder_ThinkingMode(t *testing.T) {
	req, err := ds4.New("key").Chat().
		ThinkingMode().
		User("What is 2+2?").
		Build()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Model != ds4.ModelDeepSeekReasoner {
		t.Errorf("ThinkingMode should switch model to %q, got %q", ds4.ModelDeepSeekReasoner, req.Model)
	}
}

func TestChatBuilder_JSONMode(t *testing.T) {
	req, err := ds4.New("key").Chat().
		JSONMode().
		User("Return JSON").
		Build()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.ResponseFormat == nil || req.ResponseFormat.Type != "json_object" {
		t.Errorf("expected ResponseFormat.Type=json_object, got %v", req.ResponseFormat)
	}
}

func TestChatBuilder_Prefix(t *testing.T) {
	req, err := ds4.New("key").Chat().
		User("Write a function").
		Prefix("func hello() {\n").
		Build()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	last := req.Messages[len(req.Messages)-1]
	if last.Role != ds4.RoleAssistant {
		t.Errorf("prefix message should have role assistant, got %q", last.Role)
	}
	if !last.Prefix {
		t.Error("prefix message Prefix field should be true")
	}
	if last.Content != "func hello() {\n" {
		t.Errorf("unexpected prefix content: %q", last.Content)
	}
}

func TestChatBuilder_Tools(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"city": map[string]any{"type": "string"},
		},
		"required": []string{"city"},
	}
	tool := ds4.NewFunction("get_weather", "Get weather for a city", schema)

	req, err := ds4.New("key").Chat().
		Tool(tool).
		User("What's the weather in Tokyo?").
		Build()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(req.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(req.Tools))
	}
	if req.Tools[0].Function.Name != "get_weather" {
		t.Errorf("unexpected tool name: %q", req.Tools[0].Function.Name)
	}
}

func TestChatBuilder_ToolChoice(t *testing.T) {
	tool := ds4.NewFunction("fn", "", nil)
	req, err := ds4.New("key").Chat().
		Tool(tool).
		ToolChoice("auto").
		User("Hi").
		Build()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.ToolChoice != "auto" {
		t.Errorf("expected ToolChoice 'auto', got %v", req.ToolChoice)
	}
}

func TestChatBuilder_MultiRound(t *testing.T) {
	b := ds4.New("key").Chat().System("You are helpful.")

	// Simulate adding messages for a multi-round conversation.
	b.User("First question")
	req1, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	if len(req1.Messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(req1.Messages))
	}

	b.Assistant("First answer").User("Second question")
	req2, err := b.Build()
	if err != nil {
		t.Fatal(err)
	}
	if len(req2.Messages) != 4 {
		t.Fatalf("expected 4 messages for multi-round, got %d", len(req2.Messages))
	}
}

func TestChatBuilder_ToolResult(t *testing.T) {
	req, err := ds4.New("key").Chat().
		User("What's the weather?").
		ToolResult("call_abc123", `{"temp": 22, "unit": "C"}`).
		Build()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	last := req.Messages[len(req.Messages)-1]
	if last.Role != ds4.RoleTool {
		t.Errorf("expected tool role, got %q", last.Role)
	}
	if last.ToolCallID != "call_abc123" {
		t.Errorf("unexpected ToolCallID: %q", last.ToolCallID)
	}
}

func TestChatBuilder_Clone(t *testing.T) {
	original := ds4.New("key").Chat().System("base").User("Q")
	cloned := original.Clone()

	// Mutate the clone – should not affect original.
	cloned.User("extra")

	origReq, _ := original.Build()
	cloneReq, _ := cloned.Build()

	if len(origReq.Messages) != 2 {
		t.Errorf("original should still have 2 messages, got %d", len(origReq.Messages))
	}
	if len(cloneReq.Messages) != 3 {
		t.Errorf("clone should have 3 messages, got %d", len(cloneReq.Messages))
	}
}

func TestChatBuilder_Temperature(t *testing.T) {
	temp := 0.7
	req, err := ds4.New("key").Chat().
		Temperature(temp).
		User("Hi").
		Build()
	if err != nil {
		t.Fatal(err)
	}
	if req.Temperature == nil || *req.Temperature != temp {
		t.Errorf("expected temperature %v, got %v", temp, req.Temperature)
	}
}

func TestChatBuilder_MaxTokens(t *testing.T) {
	req, err := ds4.New("key").Chat().
		MaxTokens(512).
		User("Hi").
		Build()
	if err != nil {
		t.Fatal(err)
	}
	if req.MaxTokens != 512 {
		t.Errorf("expected MaxTokens 512, got %d", req.MaxTokens)
	}
}

func TestChatBuilder_Stop(t *testing.T) {
	req, err := ds4.New("key").Chat().
		Stop("STOP", "END").
		User("Hi").
		Build()
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Stop) != 2 {
		t.Errorf("expected 2 stop sequences, got %d", len(req.Stop))
	}
}

func TestChatBuilder_WithMessages(t *testing.T) {
	msgs := []ds4.Message{
		{Role: ds4.RoleUser, Content: "hello"},
	}
	req, err := ds4.New("key").Chat().WithMessages(msgs).Build()
	if err != nil {
		t.Fatal(err)
	}
	if len(req.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(req.Messages))
	}
}

func TestChatBuilder_AddMessage(t *testing.T) {
	req, err := ds4.New("key").Chat().
		AddMessage(ds4.Message{Role: ds4.RoleUser, Content: "raw message"}).
		Build()
	if err != nil {
		t.Fatal(err)
	}
	if req.Messages[0].Content != "raw message" {
		t.Errorf("unexpected content: %q", req.Messages[0].Content)
	}
}

// ---------------------------------------------------------------------------
// FIMBuilder unit tests
// ---------------------------------------------------------------------------

func TestFIMBuilder_BuildRequiresPrompt(t *testing.T) {
	_, err := ds4.New("key").FIM().Build()
	if err == nil {
		t.Fatal("expected error when prompt is empty")
	}
}

func TestFIMBuilder_Build(t *testing.T) {
	req, err := ds4.New("key").FIM().
		Prompt("func greet() {\n    return ").
		Suffix("\n}").
		MaxTokens(64).
		Build()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if req.Model != ds4.ModelDeepSeekV4Flash {
		t.Errorf("expected default model, got %q", req.Model)
	}
	if req.Prompt == "" {
		t.Error("prompt should not be empty")
	}
	if req.Suffix != "\n}" {
		t.Errorf("unexpected suffix: %q", req.Suffix)
	}
	if req.MaxTokens != 64 {
		t.Errorf("expected MaxTokens 64, got %d", req.MaxTokens)
	}
}

func TestFIMBuilder_Echo(t *testing.T) {
	req, err := ds4.New("key").FIM().Prompt("p").Echo().Build()
	if err != nil {
		t.Fatal(err)
	}
	if !req.Echo {
		t.Error("Echo should be true")
	}
}

// ---------------------------------------------------------------------------
// ChatResponse helper tests
// ---------------------------------------------------------------------------

func TestChatResponse_Helpers(t *testing.T) {
	resp := &ds4.ChatResponse{
		Choices: []ds4.Choice{
			{
				Index: 0,
				Message: ds4.Message{
					Role:             ds4.RoleAssistant,
					Content:          "Hello!",
					ReasoningContent: "I thought about it.",
					ToolCalls: []ds4.ToolCall{
						{ID: "tc1", Type: "function", Function: ds4.FunctionCall{Name: "fn", Arguments: "{}"}},
					},
				},
				FinishReason: ds4.FinishReasonToolCalls,
			},
		},
	}

	if resp.Content() != "Hello!" {
		t.Errorf("Content() = %q", resp.Content())
	}
	if resp.ReasoningContent() != "I thought about it." {
		t.Errorf("ReasoningContent() = %q", resp.ReasoningContent())
	}
	if len(resp.ToolCalls()) != 1 {
		t.Errorf("ToolCalls() length = %d", len(resp.ToolCalls()))
	}
	if resp.FinishReason() != ds4.FinishReasonToolCalls {
		t.Errorf("FinishReason() = %q", resp.FinishReason())
	}
	if resp.FirstMessage() == nil {
		t.Error("FirstMessage() should not be nil")
	}
}

func TestChatResponse_EmptyChoices(t *testing.T) {
	resp := &ds4.ChatResponse{}
	if resp.Content() != "" {
		t.Error("Content() on empty choices should be empty string")
	}
	if resp.ReasoningContent() != "" {
		t.Error("ReasoningContent() on empty choices should be empty string")
	}
	if resp.ToolCalls() != nil {
		t.Error("ToolCalls() on empty choices should be nil")
	}
	if resp.FinishReason() != "" {
		t.Error("FinishReason() on empty choices should be empty string")
	}
	if resp.FirstMessage() != nil {
		t.Error("FirstMessage() on empty choices should be nil")
	}
}

// ---------------------------------------------------------------------------
// FIMResponse helper tests
// ---------------------------------------------------------------------------

func TestFIMResponse_Helpers(t *testing.T) {
	resp := &ds4.FIMResponse{
		Choices: []ds4.FIMChoice{
			{Index: 0, Text: `"hello"`, FinishReason: ds4.FinishReasonStop},
		},
	}
	if resp.Text() != `"hello"` {
		t.Errorf("Text() = %q", resp.Text())
	}
	if resp.FinishReason() != ds4.FinishReasonStop {
		t.Errorf("FinishReason() = %q", resp.FinishReason())
	}
}

// ---------------------------------------------------------------------------
// APIError tests
// ---------------------------------------------------------------------------

func TestAPIError_Error(t *testing.T) {
	err := &ds4.APIError{StatusCode: 401, Message: "invalid API key"}
	s := err.Error()
	if !strings.Contains(s, "401") {
		t.Errorf("error string should contain status code, got %q", s)
	}
	if !strings.Contains(s, "invalid API key") {
		t.Errorf("error string should contain message, got %q", s)
	}
}

// ---------------------------------------------------------------------------
// Mock-server HTTP tests
// ---------------------------------------------------------------------------

func chatFixture() ds4.ChatResponse {
	return ds4.ChatResponse{
		ID:    "chatcmpl-test",
		Model: ds4.ModelDeepSeekV4Flash,
		Choices: []ds4.Choice{
			{
				Index: 0,
				Message: ds4.Message{
					Role:    ds4.RoleAssistant,
					Content: "Hi there!",
				},
				FinishReason: ds4.FinishReasonStop,
			},
		},
		Usage: ds4.Usage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
	}
}

func TestClient_Chat_MockServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if !strings.HasPrefix(r.Header.Get("Authorization"), "Bearer ") {
			t.Errorf("expected ****** in Authorization header, got: %s", r.Header.Get("Authorization"))
		}

		var req ds4.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if req.Model != ds4.ModelDeepSeekV4Flash {
			t.Errorf("unexpected model: %s", req.Model)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chatFixture())
	}))
	defer srv.Close()

	client := ds4.New("testkey").WithBaseURL(srv.URL)
	resp, err := client.Chat().User("Hello").Do()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content() != "Hi there!" {
		t.Errorf("unexpected content: %q", resp.Content())
	}
	if resp.Usage.TotalTokens != 8 {
		t.Errorf("unexpected total tokens: %d", resp.Usage.TotalTokens)
	}
}

func TestClient_Chat_ErrorResponse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprintln(w, `{"error":{"message":"invalid API key","type":"authentication_error","code":"invalid_api_key"}}`)
	}))
	defer srv.Close()

	client := ds4.New("bad").WithBaseURL(srv.URL)
	_, err := client.Chat().User("Hi").Do()
	if err == nil {
		t.Fatal("expected error for 401 response")
	}
	apiErr, ok := err.(*ds4.APIError)
	if !ok {
		t.Fatalf("expected *ds4.APIError, got %T: %v", err, err)
	}
	if apiErr.StatusCode != http.StatusUnauthorized {
		t.Errorf("expected status 401, got %d", apiErr.StatusCode)
	}
	if !strings.Contains(apiErr.Message, "invalid API key") {
		t.Errorf("unexpected error message: %q", apiErr.Message)
	}
}

func TestClient_Chat_JSONMode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ds4.ChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.ResponseFormat == nil || req.ResponseFormat.Type != "json_object" {
			t.Error("expected ResponseFormat.Type = json_object")
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(chatFixture())
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL)
	_, err := client.Chat().JSONMode().User("Return JSON").Do()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestClient_Chat_ThinkingMode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ds4.ChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Model != ds4.ModelDeepSeekReasoner {
			t.Errorf("expected model %s, got %s", ds4.ModelDeepSeekReasoner, req.Model)
		}

		resp := chatFixture()
		resp.Choices[0].Message.ReasoningContent = "step-by-step reasoning"
		resp.Choices[0].Message.Content = "4"
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL)
	resp, err := client.Chat().ThinkingMode().User("2+2?").Do()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ReasoningContent() != "step-by-step reasoning" {
		t.Errorf("unexpected reasoning content: %q", resp.ReasoningContent())
	}
	if resp.Content() != "4" {
		t.Errorf("unexpected content: %q", resp.Content())
	}
}

func TestClient_Chat_ToolCalls(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ds4.ChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if len(req.Tools) == 0 {
			t.Error("expected tools in request")
		}

		resp := chatFixture()
		resp.Choices[0].Message.Content = ""
		resp.Choices[0].Message.ToolCalls = []ds4.ToolCall{
			{
				ID:   "call_1",
				Type: "function",
				Function: ds4.FunctionCall{
					Name:      "get_weather",
					Arguments: `{"city":"Tokyo"}`,
				},
			},
		}
		resp.Choices[0].FinishReason = ds4.FinishReasonToolCalls
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	tool := ds4.NewFunction("get_weather", "Get weather", map[string]any{
		"type":       "object",
		"properties": map[string]any{"city": map[string]any{"type": "string"}},
	})

	client := ds4.New("key").WithBaseURL(srv.URL)
	resp, err := client.Chat().Tool(tool).User("Weather in Tokyo?").Do()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.FinishReason() != ds4.FinishReasonToolCalls {
		t.Errorf("unexpected finish reason: %q", resp.FinishReason())
	}
	tcs := resp.ToolCalls()
	if len(tcs) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(tcs))
	}
	if tcs[0].Function.Name != "get_weather" {
		t.Errorf("unexpected tool call name: %q", tcs[0].Function.Name)
	}
}

func TestClient_Chat_Prefix(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ds4.ChatRequest
		json.NewDecoder(r.Body).Decode(&req)

		// Last message should be an assistant message with Prefix: true.
		last := req.Messages[len(req.Messages)-1]
		if last.Role != ds4.RoleAssistant {
			t.Errorf("expected last message role assistant, got %q", last.Role)
		}
		if !last.Prefix {
			t.Error("expected Prefix=true on last message")
		}

		resp := chatFixture()
		resp.Choices[0].Message.Content = "func hello() {\n    fmt.Println(\"hello\")\n}"
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL)
	resp, err := client.Chat().
		User("Write a hello function").
		Prefix("func hello() {\n").
		Do()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content() == "" {
		t.Error("expected non-empty content")
	}
}

func TestClient_Chat_Stream(t *testing.T) {
	sseLines := []string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"He"},"finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ds4.ChatRequest
		json.NewDecoder(r.Body).Decode(&req)
		if !req.Stream {
			t.Error("expected stream=true in request")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range sseLines {
			fmt.Fprintln(w, line)
		}
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL)
	var result strings.Builder
	err := client.Chat().User("Say Hello!").Stream(func(chunk ds4.ChatStreamChunk) error {
		if len(chunk.Choices) > 0 {
			result.WriteString(chunk.Choices[0].Delta.Content)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.String() != "Hello!" {
		t.Errorf("expected 'Hello!', got %q", result.String())
	}
}

func TestClient_Chat_StreamWithContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"id":"1","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL)
	ctx := context.Background()
	var got string
	err := client.Chat().User("Hi").StreamWithContext(ctx, func(c ds4.ChatStreamChunk) error {
		if len(c.Choices) > 0 {
			got = c.Choices[0].Delta.Content
		}
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "hi" {
		t.Errorf("expected 'hi', got %q", got)
	}
}

func TestClient_Chat_Stream_HandlerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"id":"1","choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer srv.Close()

	sentinel := fmt.Errorf("stop early")
	client := ds4.New("key").WithBaseURL(srv.URL)
	err := client.Chat().User("x").Stream(func(_ ds4.ChatStreamChunk) error {
		return sentinel
	})
	if err != sentinel {
		t.Errorf("expected sentinel error, got %v", err)
	}
}

func TestClient_FIM_MockServer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/beta/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		var req ds4.FIMRequest
		json.NewDecoder(r.Body).Decode(&req)
		if req.Prompt == "" {
			t.Error("expected non-empty prompt")
		}

		resp := ds4.FIMResponse{
			ID:    "fim-test",
			Model: ds4.ModelDeepSeekV4Flash,
			Choices: []ds4.FIMChoice{
				{Index: 0, Text: `fmt.Println("hello")`, FinishReason: ds4.FinishReasonStop},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL)
	resp, err := client.FIM().
		Prompt("func main() {\n    ").
		Suffix("\n}").
		Do()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Text() == "" {
		t.Error("expected non-empty FIM text")
	}
	if resp.FinishReason() != ds4.FinishReasonStop {
		t.Errorf("unexpected finish reason: %q", resp.FinishReason())
	}
}

func TestClient_FIM_Stream(t *testing.T) {
	sseLines := []string{
		`data: {"id":"1","choices":[{"index":0,"text":"fmt","finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"text":".Println","finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"text":"()","finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ds4.FIMRequest
		json.NewDecoder(r.Body).Decode(&req)
		if !req.Stream {
			t.Error("expected stream=true")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range sseLines {
			fmt.Fprintln(w, line)
		}
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL)
	var result strings.Builder
	err := client.FIM().Prompt("func main() {\n    ").Stream(func(chunk ds4.FIMStreamChunk) error {
		if len(chunk.Choices) > 0 {
			result.WriteString(chunk.Choices[0].Text)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.String() != "fmt.Println()" {
		t.Errorf("expected 'fmt.Println()', got %q", result.String())
	}
}

func TestClient_WithHTTPClient(t *testing.T) {
	custom := &http.Client{}
	c := ds4.New("key").WithHTTPClient(custom)
	if c == nil {
		t.Error("expected non-nil client")
	}
}

func TestNewFunction(t *testing.T) {
	schema := map[string]any{"type": "object"}
	tool := ds4.NewFunction("fn", "desc", schema)
	if tool.Type != "function" {
		t.Errorf("expected type 'function', got %q", tool.Type)
	}
	if tool.Function.Name != "fn" {
		t.Errorf("expected name 'fn', got %q", tool.Function.Name)
	}
	if tool.Function.Description != "desc" {
		t.Errorf("expected description 'desc', got %q", tool.Function.Description)
	}
}

// ---------------------------------------------------------------------------
// Integration tests (require DEEPSEEK_API_KEY)
// ---------------------------------------------------------------------------

func TestIntegration_Chat(t *testing.T) {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		t.Skip("DEEPSEEK_API_KEY not set; skipping integration test")
	}

	client := ds4.New(apiKey)
	resp, err := client.Chat().
		System("You are a helpful assistant. Be concise.").
		User("Reply with exactly the word 'pong'.").
		MaxTokens(16).
		Do()
	if err != nil {
		t.Fatalf("integration: unexpected error: %v", err)
	}
	content := strings.ToLower(strings.TrimSpace(resp.Content()))
	if !strings.Contains(content, "pong") {
		t.Errorf("integration: expected 'pong' in response, got %q", resp.Content())
	}
}

func TestIntegration_Chat_Stream(t *testing.T) {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		t.Skip("DEEPSEEK_API_KEY not set; skipping integration test")
	}

	client := ds4.New(apiKey)
	var result strings.Builder
	err := client.Chat().
		User("Count from 1 to 3, one number per line, nothing else.").
		MaxTokens(32).
		Stream(func(chunk ds4.ChatStreamChunk) error {
			if len(chunk.Choices) > 0 {
				result.WriteString(chunk.Choices[0].Delta.Content)
			}
			return nil
		})
	if err != nil {
		t.Fatalf("integration stream: unexpected error: %v", err)
	}
	if result.Len() == 0 {
		t.Error("integration stream: expected non-empty streamed content")
	}
}
