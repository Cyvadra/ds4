package ds4_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Cyvadra/ds4"
)

// ---------------------------------------------------------------------------
// SplitReasoning unit tests
// ---------------------------------------------------------------------------

func TestSplitReasoning(t *testing.T) {
	tests := []struct {
		name          string
		text          string
		wantReasoning string
		wantAnswer    string
		wantSplits    bool
	}{
		{
			name:          "angle bracket markers",
			text:          "<reasoning>Let me verify.</reasoning>\n\nThe answer is 4.",
			wantReasoning: "Let me verify.",
			wantAnswer:    "The answer is 4.",
			wantSplits:    true,
		},
		{
			name:          "whitespace before open marker",
			text:          "\n\n<reasoning>  compute  </reasoning>\n42",
			wantReasoning: "compute",
			wantAnswer:    "42",
			wantSplits:    true,
		},
		{
			name:          "think slash think markers",
			text:          " thinkSum the numbers.\n/think\nThe sum is 3.",
			wantReasoning: "Sum the numbers.",
			wantAnswer:    "The sum is 3.",
			wantSplits:    true,
		},
		{
			name:       "no markers is plain answer",
			text:       "Hello there",
			wantAnswer: "Hello there",
		},
		{
			name:       "prose starting with think is not reasoning",
			text:       "thinking about it",
			wantAnswer: "thinking about it",
		},
		{
			name:       "lowercase think with no close stays answer",
			text:       "think about the options carefully",
			wantAnswer: "think about the options carefully",
		},
		{
			name:       "unclosed reasoning block stays answer",
			text:       "<reasoning>this never closes",
			wantAnswer: "<reasoning>this never closes",
		},
		{
			name:       "mid-text marker is not reasoning",
			text:       "First <reasoning>not at start</reasoning>rest",
			wantAnswer: "First <reasoning>not at start</reasoning>rest",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			reasoning, answer := ds4.SplitReasoning(test.text)
			if reasoning != test.wantReasoning {
				t.Errorf("reasoning = %q, want %q", reasoning, test.wantReasoning)
			}
			if answer != test.wantAnswer {
				t.Errorf("answer = %q, want %q", answer, test.wantAnswer)
			}
			if test.wantSplits && reasoning == "" {
				t.Error("expected a reasoning split")
			}
		})
	}
}

// ---------------------------------------------------------------------------
// Local chat: endpoint routing, thinking stripping, reasoning split
// ---------------------------------------------------------------------------

func TestLocal_Chat_UsesChatEndpointAndDropsThinking(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("local chat must use /chat/completions, got %s", r.URL.Path)
		}
		var req ds4.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Thinking != nil {
			t.Errorf("thinking must be dropped on local backend, got %v", req.Thinking)
		}
		if req.ReasoningEffort != "" {
			t.Errorf("reasoning_effort must be dropped on local backend, got %q", req.ReasoningEffort)
		}
		last := req.Messages[len(req.Messages)-1]
		if last.Role != ds4.RoleAssistant || !last.Prefix {
			t.Errorf("expected final assistant prefix message, got %+v", last)
		}
		if len(req.Tools) != 1 || !req.Tools[0].Function.Strict {
			t.Errorf("expected strict tool still sent, got %+v", req.Tools)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ds4.ChatResponse{
			ID:    "local-1",
			Model: "local-model",
			Choices: []ds4.Choice{{
				Index:        0,
				FinishReason: ds4.FinishReasonStop,
				Message: ds4.Message{
					Role:    ds4.RoleAssistant,
					Content: "<reasoning>Let me verify.</reasoning>\n\n4",
				},
			}},
		})
	}))
	defer srv.Close()

	client := ds4.New("key").
		WithBaseURL(srv.URL).
		WithLocalBackend(ds4.LocalBackend{FIM: true})

	tool := ds4.NewStrictFunction("get_weather", "Get weather", map[string]any{"type": "object"})
	resp, err := client.Chat().
		ThinkingMode().
		ReasoningEffort(ds4.EffortMax).
		User("2+2?").
		Prefix("Answer: ").
		Tool(tool).
		Do()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.ReasoningContent() != "Let me verify." {
		t.Errorf("ReasoningContent = %q", resp.ReasoningContent())
	}
	if resp.Content() != "4" {
		t.Errorf("Content = %q", resp.Content())
	}
}

func TestLocal_Chat_Stream_SplitsReasoning(t *testing.T) {
	sseLines := []string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"<reason"},"finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"ing>Let me compute"},"finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"</reasoning>The answer is 42"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range sseLines {
			fmt.Fprintln(w, line)
		}
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL).WithLocalBackend(ds4.LocalBackend{})
	var reasoning, content strings.Builder
	err := client.Chat().User("2+2?").Stream(func(chunk ds4.ChatStreamChunk) error {
		if len(chunk.Choices) == 0 {
			return nil
		}
		reasoning.WriteString(chunk.Choices[0].Delta.ReasoningContent)
		content.WriteString(chunk.Choices[0].Delta.Content)
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if reasoning.String() != "Let me compute" {
		t.Errorf("streamed reasoning = %q", reasoning.String())
	}
	if content.String() != "The answer is 42" {
		t.Errorf("streamed content = %q", content.String())
	}
}

func TestLocal_Chat_Stream_WithoutMarkersPassesThrough(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":"stop"}]}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL).WithLocalBackend(ds4.LocalBackend{})
	var content strings.Builder
	err := client.Chat().User("Hi").Stream(func(chunk ds4.ChatStreamChunk) error {
		if len(chunk.Choices) == 0 {
			return nil
		}
		if chunk.Choices[0].Delta.ReasoningContent != "" {
			t.Errorf("unexpected reasoning delta: %q", chunk.Choices[0].Delta.ReasoningContent)
		}
		content.WriteString(chunk.Choices[0].Delta.Content)
		return nil
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if content.String() != "Hello" {
		t.Errorf("content = %q", content.String())
	}
}

// ---------------------------------------------------------------------------
// Local FIM: /completions + input_prefix/input_suffix
// ---------------------------------------------------------------------------

type localFIMRequest struct {
	Model       string `json:"model"`
	InputPrefix string `json:"input_prefix"`
	InputSuffix string `json:"input_suffix"`
	MaxTokens   int    `json:"max_tokens,omitempty"`
	Stream      bool   `json:"stream,omitempty"`
}

func TestLocal_FIM(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/completions" {
			t.Errorf("local FIM must use /completions, got %s", r.URL.Path)
		}
		var req localFIMRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.InputPrefix != "func main() {\n    " {
			t.Errorf("input_prefix = %q", req.InputPrefix)
		}
		if req.InputSuffix != "\n}" {
			t.Errorf("input_suffix = %q", req.InputSuffix)
		}
		if req.MaxTokens != 64 {
			t.Errorf("max_tokens = %d", req.MaxTokens)
		}
		if req.Stream {
			t.Error("expected non-streaming request")
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(ds4.FIMResponse{
			ID:    "cmpl-1",
			Model: "local-model",
			Choices: []ds4.FIMChoice{{
				Index:        0,
				Text:         `fmt.Println("hi")`,
				FinishReason: ds4.FinishReasonStop,
			}},
			Usage: ds4.Usage{PromptTokens: 10, CompletionTokens: 3, TotalTokens: 13},
		})
	}))
	defer srv.Close()

	client := ds4.New("key").
		WithBaseURL(srv.URL).
		WithLocalBackend(ds4.LocalBackend{FIM: true})

	resp, err := client.FIM().
		Prompt("func main() {\n    ").
		Suffix("\n}").
		MaxTokens(64).
		Do()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Text() != `fmt.Println("hi")` {
		t.Errorf("Text = %q", resp.Text())
	}
	if resp.FinishReason() != ds4.FinishReasonStop {
		t.Errorf("FinishReason = %q", resp.FinishReason())
	}
}

func TestLocal_FIM_RequiresEnabledFlag(t *testing.T) {
	client := ds4.New("key").WithBaseURL("http://localhost:8080/v1").WithLocalBackend(ds4.LocalBackend{})
	_, err := client.FIM().Prompt("p").Do()
	if err == nil {
		t.Fatal("expected error when LocalBackend.FIM is disabled")
	}
	if !strings.Contains(err.Error(), "FIM") {
		t.Errorf("expected error mentioning FIM, got %v", err)
	}
}

func TestLocal_FIM_Stream(t *testing.T) {
	sseLines := []string{
		`data: {"id":"1","choices":[{"index":0,"text":"fmt","finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"text":".Println(\"hi\")","finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/completions" {
			t.Errorf("local FIM must use /completions, got %s", r.URL.Path)
		}
		var req localFIMRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if !req.Stream {
			t.Error("expected stream=true")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range sseLines {
			fmt.Fprintln(w, line)
		}
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL).WithLocalBackend(ds4.LocalBackend{FIM: true})
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
	if result.String() != `fmt.Println("hi")` {
		t.Errorf("streamed text = %q", result.String())
	}
}
