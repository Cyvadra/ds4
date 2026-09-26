package ds4_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Cyvadra/ds4"
)

func emptyOfficialModelsServer(t *testing.T) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, `{"data":[]}`)
	}))
	t.Cleanup(server.Close)
	return server
}

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
	official := emptyOfficialModelsServer(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			fmt.Fprint(w, `{"data":[{"id":"local-model"}]}`)
			return
		}
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
		WithBaseURL(official.URL).
		WithLocalURL(srv.URL).
		WithLocalFIM(true)

	tool := ds4.NewStrictFunction("get_weather", "Get weather", map[string]any{"type": "object"})
	resp, err := client.Chat().
		Model("local-model").
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
	official := emptyOfficialModelsServer(t)
	sseLines := []string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"<reason"},"finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"ing>Let me compute"},"finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"</reasoning>The answer is 42"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			fmt.Fprint(w, `{"data":[{"id":"local-model"}]}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range sseLines {
			fmt.Fprintln(w, line)
		}
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(official.URL).WithLocalURL(srv.URL)
	var reasoning, content strings.Builder
	err := client.Chat().Model("local-model").User("2+2?").Stream(func(chunk ds4.ChatStreamChunk) error {
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

func TestLocal_Chat_Stream_PreservesNativeReasoning(t *testing.T) {
	official := emptyOfficialModelsServer(t)
	sseLines := []string{
		`data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":null},"finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"delta":{"reasoning_content":"Let me compute"},"finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"delta":{"content":"The answer is 42"},"finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			fmt.Fprint(w, `{"data":[{"id":"local-model"}]}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range sseLines {
			fmt.Fprintln(w, line)
		}
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(official.URL).WithLocalURL(srv.URL)
	var reasoning, content strings.Builder
	err := client.Chat().Model("local-model").User("2+2?").Stream(func(chunk ds4.ChatStreamChunk) error {
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
	official := emptyOfficialModelsServer(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			fmt.Fprint(w, `{"data":[{"id":"local-model"}]}`)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintln(w, `data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":"stop"}]}`)
		fmt.Fprintln(w, `data: [DONE]`)
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(official.URL).WithLocalURL(srv.URL)
	var content strings.Builder
	err := client.Chat().Model("local-model").User("Hi").Stream(func(chunk ds4.ChatStreamChunk) error {
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
	official := emptyOfficialModelsServer(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			fmt.Fprint(w, `{"data":[{"id":"local-model"}]}`)
			return
		}
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
		WithBaseURL(official.URL).
		WithLocalURL(srv.URL).
		WithLocalFIM(true)

	resp, err := client.FIM().
		Model("local-model").
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
	official := emptyOfficialModelsServer(t)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"data":[{"id":"local-model"}]}`)
	}))
	defer srv.Close()
	client := ds4.New("key").WithBaseURL(official.URL).WithLocalURL(srv.URL)
	_, err := client.FIM().Model("local-model").Prompt("p").Do()
	if err == nil {
		t.Fatal("expected error when WithLocalFIM is disabled")
	}
	if !strings.Contains(err.Error(), "FIM") {
		t.Errorf("expected error mentioning FIM, got %v", err)
	}
}

func TestLocal_FIM_Stream(t *testing.T) {
	official := emptyOfficialModelsServer(t)
	sseLines := []string{
		`data: {"id":"1","choices":[{"index":0,"text":"fmt","finish_reason":null}]}`,
		`data: {"id":"1","choices":[{"index":0,"text":".Println(\"hi\")","finish_reason":"stop"}]}`,
		`data: [DONE]`,
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			fmt.Fprint(w, `{"data":[{"id":"local-model"}]}`)
			return
		}
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

	client := ds4.New("key").WithBaseURL(official.URL).WithLocalURL(srv.URL).WithLocalFIM(true)
	var result strings.Builder
	err := client.FIM().Model("local-model").Prompt("func main() {\n    ").Stream(func(chunk ds4.FIMStreamChunk) error {
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

func TestModelRouting_LocalOverridesOfficialAndCaches(t *testing.T) {
	var officialModels, localModels, officialChats, localChats atomic.Int32
	official := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			officialModels.Add(1)
			if got := r.Header.Get("Authorization"); got != "Bearer official-key" {
				t.Errorf("official Authorization = %q", got)
			}
			fmt.Fprint(w, `{"data":[{"id":"shared"},{"id":"official-only"}]}`)
		case "/chat/completions":
			officialChats.Add(1)
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"official"}}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer official.Close()

	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			localModels.Add(1)
			if got := r.Header.Get("Authorization"); got != "" {
				t.Errorf("local Authorization = %q, want empty", got)
			}
			fmt.Fprint(w, `{"data":[{"id":"shared"},{"id":"local-only"}]}`)
		case "/chat/completions":
			localChats.Add(1)
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"local"}}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer local.Close()

	client := ds4.New("official-key").WithBaseURL(official.URL).WithLocalKey("").WithLocalURL(local.URL)
	resp, err := client.Chat().Model("shared").User("hi").Do()
	if err != nil {
		t.Fatalf("shared model: %v", err)
	}
	if resp.Content() != "local" {
		t.Fatalf("shared response = %q", resp.Content())
	}
	if _, err := client.Chat().Model("official-only").User("hi").Do(); err != nil {
		t.Fatalf("official model: %v", err)
	}
	if officialModels.Load() != 1 || localModels.Load() != 1 {
		t.Fatalf("model calls = official %d, local %d", officialModels.Load(), localModels.Load())
	}
	if officialChats.Load() != 1 || localChats.Load() != 1 {
		t.Fatalf("chat calls = official %d, local %d", officialChats.Load(), localChats.Load())
	}

	if err := client.Ping(); err != nil {
		t.Fatalf("Ping: %v", err)
	}
	if officialModels.Load() != 2 || localModels.Load() != 2 {
		t.Fatalf("model calls after Ping = official %d, local %d", officialModels.Load(), localModels.Load())
	}
}

func TestModelRouting_LoadsEveryAdvertisedIdentifier(t *testing.T) {
	official := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			fmt.Fprint(w, `{"data":[{"id":"by-id","name":"by-name","model":"by-model","aliases":["by-alias"]}],"models":[{"id":"models-id","name":"models-name","model":"models-model","aliases":["models-alias"]}]}`)
		case "/chat/completions":
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"official"}}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer official.Close()

	client := ds4.New("key").WithBaseURL(official.URL)
	for _, model := range []string{"by-id", "by-name", "by-model", "by-alias", "models-id", "models-name", "models-model", "models-alias"} {
		response, err := client.Chat().Model(model).User("hi").Do()
		if err != nil {
			t.Fatalf("model %q: %v", model, err)
		}
		if response.Content() != "official" {
			t.Fatalf("model %q response = %q", model, response.Content())
		}
	}
}

func TestModelRouting_UnknownModelRefreshesOfficialBeforeUsingLocal(t *testing.T) {
	var officialModels atomic.Int32
	official := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			if officialModels.Add(1) == 1 {
				fmt.Fprint(w, `{"data":[{"id":"initial"}]}`)
				return
			}
			fmt.Fprint(w, `{"data":[{"id":"initial"},{"id":"new-model"}]}`)
		case "/chat/completions":
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"official"}}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer official.Close()

	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			fmt.Fprint(w, `{"data":[{"id":"local-only"}]}`)
		case "/chat/completions":
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"local"}}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer local.Close()

	client := ds4.New("key").WithBaseURL(official.URL).WithLocalURL(local.URL)
	if _, err := client.Chat().Model("initial").User("hi").Do(); err != nil {
		t.Fatalf("initial model: %v", err)
	}
	response, err := client.Chat().Model("new-model").User("hi").Do()
	if err != nil {
		t.Fatalf("new official model: %v", err)
	}
	if response.Content() != "official" || officialModels.Load() != 2 {
		t.Fatalf("response = %q, official model calls = %d", response.Content(), officialModels.Load())
	}
}

func TestModelRouting_DiscoveryFailureIsIndependent(t *testing.T) {
	var officialChats atomic.Int32
	official := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			fmt.Fprint(w, `{"data":[{"id":"official-model"}]}`)
			return
		}
		officialChats.Add(1)
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer official.Close()
	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "offline", http.StatusServiceUnavailable)
	}))
	defer local.Close()

	client := ds4.New("key").WithBaseURL(official.URL).WithLocalURL(local.URL)
	resp, err := client.Chat().Model("official-model").User("hi").Do()
	if err != nil {
		t.Fatalf("successful backend should remain usable: %v", err)
	}
	if resp.Content() != "ok" || officialChats.Load() != 1 {
		t.Fatalf("response = %q, chats = %d", resp.Content(), officialChats.Load())
	}
	if err := client.Ping(); err == nil || !strings.Contains(err.Error(), "local") {
		t.Fatalf("Ping error = %v, want local discovery error", err)
	}
}

func TestModelRouting_RetriesFailedLlamaCppDiscovery(t *testing.T) {
	var localModels, localChats atomic.Int32
	official := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			fmt.Fprint(w, `{"data":[{"id":"deepseek-flash"}]}`)
			return
		}
		http.NotFound(w, r)
	}))
	defer official.Close()

	local := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/models":
			if localModels.Add(1) == 1 {
				http.Error(w, "starting", http.StatusServiceUnavailable)
				return
			}
			fmt.Fprint(w, `{"models":[{"name":"ds4","model":"ds4","capabilities":["completion"]}],"object":"list","data":[{"id":"model-file.gguf","aliases":["ds4"],"object":"model","owned_by":"llamacpp"}]}`)
		case "/chat/completions":
			localChats.Add(1)
			fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"local"}}]}`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer local.Close()

	client := ds4.New("key").WithBaseURL(official.URL).WithLocalURL(local.URL)
	if _, err := client.Chat().Model("ds4").User("hi").Do(); err == nil || !strings.Contains(err.Error(), `unknown model "ds4"`) {
		t.Fatalf("first request error = %v, want discovery failure", err)
	}
	response, err := client.Chat().Model("ds4").User("hi").Do()
	if err != nil {
		t.Fatalf("retry request: %v", err)
	}
	if response.Content() != "local" || localModels.Load() != 2 || localChats.Load() != 1 {
		t.Fatalf("response = %q, model calls = %d, chat calls = %d", response.Content(), localModels.Load(), localChats.Load())
	}
}

func TestModelRouting_UnknownModelDoesNotSendCompletion(t *testing.T) {
	var completions atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			fmt.Fprint(w, `{"data":[{"id":"known"}]}`)
			return
		}
		completions.Add(1)
		http.Error(w, "unexpected", http.StatusInternalServerError)
	}))
	defer srv.Close()

	_, err := ds4.New("key").WithBaseURL(srv.URL).Chat().Model("unknown").User("hi").Do()
	if err == nil || !strings.Contains(err.Error(), `unknown model "unknown"`) {
		t.Fatalf("error = %v", err)
	}
	if completions.Load() != 0 {
		t.Fatalf("completion calls = %d, want 0", completions.Load())
	}
}

func TestModelRouting_ConcurrentFirstRequestsShareDiscovery(t *testing.T) {
	var models, completions atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			models.Add(1)
			time.Sleep(25 * time.Millisecond)
			fmt.Fprint(w, `{"data":[{"id":"model"}]}`)
			return
		}
		completions.Add(1)
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	}))
	defer srv.Close()

	client := ds4.New("key").WithBaseURL(srv.URL)
	const requests = 20
	var waitGroup sync.WaitGroup
	errors := make(chan error, requests)
	for request := 0; request < requests; request++ {
		waitGroup.Add(1)
		go func() {
			defer waitGroup.Done()
			_, err := client.Chat().Model("model").User("hi").Do()
			errors <- err
		}()
	}
	waitGroup.Wait()
	close(errors)
	for err := range errors {
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
	}
	if models.Load() != 1 || completions.Load() != requests {
		t.Fatalf("calls = models %d, completions %d", models.Load(), completions.Load())
	}
}
