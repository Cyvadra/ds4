package ds4_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/Cyvadra/ds4"
)

const anthropicTextResponse = `{"id":"msg_1","type":"message","role":"assistant","model":"claude-test",
"content":[{"type":"thinking","thinking":"Add them.","signature":"sig-abc"},{"type":"text","text":"42"},
{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"id":37}}],
"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":3,"cache_creation_input_tokens":2}}`

// anthropicServer serves /v1/models and records each /v1/messages body.
func anthropicServer(t *testing.T, messages func(w http.ResponseWriter, body map[string]any)) (*httptest.Server, *[]map[string]any) {
	t.Helper()
	var bodies []map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "ak" || r.Header.Get("anthropic-version") != ds4.DefaultAnthropicVersion || r.Header.Get("Authorization") != "" {
			t.Errorf("unexpected auth headers: %v", r.Header)
		}
		switch r.URL.Path {
		case "/v1/models":
			fmt.Fprint(w, `{"data":[{"id":"claude-test","type":"model"},{"id":"claude-old-20251001","type":"model"}],"has_more":false}`)
		case "/v1/messages":
			var body map[string]any
			data, _ := io.ReadAll(r.Body)
			if err := json.Unmarshal(data, &body); err != nil {
				t.Fatalf("decode body: %v", err)
			}
			bodies = append(bodies, body)
			messages(w, body)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	return server, &bodies
}

func anthropicClient(official, anthropic *httptest.Server) *ds4.Client {
	return ds4.New("k").WithBaseURL(official.URL).WithAnthropicURL(anthropic.URL).WithAnthropicKey("ak")
}

func TestAnthropic_ChatTranslatesRequestAndResponse(t *testing.T) {
	server, bodies := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		fmt.Fprint(w, anthropicTextResponse)
	})
	client := anthropicClient(emptyOfficialModelsServer(t), server)
	tool := ds4.NewStrictFunction("lookup", "Look up", map[string]any{"type": "object"})

	builder := client.Chat().Model("claude-test").
		SystemAppend("Be terse.").
		User("Look up 37.").
		WithImageURL("https://example.com/a.png").
		WithImageBase64([]byte("png"), "image/png").
		Tool(tool).ToolChoice(map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}}).
		ThinkingMode().ReasoningEffort("xhigh").Stop("END").UserID("u1")
	resp, err := builder.Do()
	if err != nil {
		t.Fatal(err)
	}

	if resp.Content() != "42" || resp.ReasoningContent() != "Add them." || resp.FinishReason() != ds4.FinishReasonToolCalls {
		t.Fatalf("unexpected response: %+v", resp)
	}
	if calls := resp.ToolCalls(); len(calls) != 1 || calls[0].ID != "toolu_1" || calls[0].Function.Arguments != `{"id":37}` {
		t.Fatalf("unexpected tool calls: %+v", calls)
	}
	if u := resp.Usage; u.PromptTokens != 15 || u.PromptCacheHitTokens != 3 || u.PromptCacheMissTokens != 12 || u.TotalTokens != 20 {
		t.Fatalf("unexpected usage: %+v", u)
	}

	body := (*bodies)[0]
	want := map[string]string{
		"model":          `"claude-test"`,
		"max_tokens":     `16000`,
		"system":         `[{"text":"You are a helpful assistant.","type":"text"},{"text":"Be terse.","type":"text"}]`,
		"thinking":       `{"display":"summarized","type":"adaptive"}`,
		"output_config":  `{"effort":"xhigh"}`,
		"tool_choice":    `{"name":"lookup","type":"tool"}`,
		"tools":          `[{"description":"Look up","input_schema":{"type":"object"},"name":"lookup","strict":true}]`,
		"stop_sequences": `["END"]`,
		"metadata":       `{"user_id":"u1"}`,
		"messages": `[{"content":[{"source":{"type":"url","url":"https://example.com/a.png"},"type":"image"},` +
			`{"source":{"data":"cG5n","media_type":"image/png","type":"base64"},"type":"image"},` +
			`{"text":"Look up 37.","type":"text"}],"role":"user"}]`,
	}
	for key, value := range want {
		got, _ := json.Marshal(body[key])
		if string(got) != value {
			t.Errorf("%s = %s, want %s", key, got, value)
		}
	}

	// The assistant turn is replayed verbatim, and tool results merge with the
	// following user message.
	if _, err := client.Chat().Model("claude-test").User("Look up 37.").AppendResponse(resp).ToolResult("toolu_1", "found").User("thanks").Do(); err != nil {
		t.Fatal(err)
	}
	messages, _ := json.Marshal((*bodies)[1]["messages"])
	for _, fragment := range []string{
		`"signature":"sig-abc"`,
		`{"content":[{"content":"found","tool_use_id":"toolu_1","type":"tool_result"},{"text":"thanks","type":"text"}],"role":"user"}`,
	} {
		if !strings.Contains(string(messages), fragment) {
			t.Errorf("messages %s missing %s", messages, fragment)
		}
	}
}

func TestAnthropic_ReplaysOpenAIStyleToolCalls(t *testing.T) {
	server, bodies := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		fmt.Fprint(w, anthropicTextResponse)
	})
	client := anthropicClient(emptyOfficialModelsServer(t), server)
	_, err := client.Chat().Model("claude-test").User("q").
		AddMessage(ds4.Message{Role: ds4.RoleAssistant, ToolCalls: []ds4.ToolCall{{ID: "c1", Type: "function", Function: ds4.FunctionCall{Name: "f"}}}}).
		ToolResult("c1", "r").ToolChoice("required").Do()
	if err != nil {
		t.Fatal(err)
	}
	got, _ := json.Marshal((*bodies)[0]["messages"])
	if !strings.Contains(string(got), `{"content":[{"id":"c1","input":{},"name":"f","type":"tool_use"}],"role":"assistant"}`) {
		t.Fatalf("unexpected messages: %s", got)
	}
	if choice, _ := json.Marshal((*bodies)[0]["tool_choice"]); choice != nil && string(choice) != "null" {
		t.Fatalf("tool_choice without tools should be omitted, got %s", choice)
	}
}

func TestAnthropic_Stream(t *testing.T) {
	events := []string{
		`{"type":"message_start","message":{"id":"msg_2","model":"claude-test","usage":{"input_tokens":7,"output_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-xyz"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hi"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_9","name":"lookup","input":{}}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"id\":"}}`,
		`{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"1}"}}`,
		`{"type":"content_block_stop","index":2}`,
		`{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`,
		`{"type":"message_stop"}`,
	}
	server, bodies := anthropicServer(t, func(w http.ResponseWriter, body map[string]any) {
		if body["stream"] != true {
			fmt.Fprint(w, anthropicTextResponse)
			return
		}
		for _, event := range events {
			fmt.Fprintf(w, "event: x\ndata: %s\n\n", event)
		}
	})
	client := anthropicClient(emptyOfficialModelsServer(t), server)
	builder := client.Chat().Model("claude-test").User("q").IncludeStreamUsage()

	var acc ds4.ChatStreamAccumulator
	if err := builder.Stream(func(chunk ds4.ChatStreamChunk) error {
		acc.Add(chunk)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	message := acc.Message()
	if message.Content != "Hi" || message.ReasoningContent != "hmm" || acc.FinishReason() != ds4.FinishReasonToolCalls {
		t.Fatalf("unexpected message: %+v", message)
	}
	if len(message.ToolCalls) != 1 || message.ToolCalls[0].ID != "toolu_9" || message.ToolCalls[0].Function.Arguments != `{"id":1}` {
		t.Fatalf("unexpected tool calls: %+v", message.ToolCalls)
	}
	if u := acc.Usage(); u == nil || u.PromptTokens != 7 || u.CompletionTokens != 9 {
		t.Fatalf("unexpected usage: %+v", u)
	}

	if _, err := builder.AddMessage(message).ToolResult("toolu_9", "ok").Do(); err != nil {
		t.Fatal(err)
	}
	replayed, _ := json.Marshal((*bodies)[1]["messages"])
	if !strings.Contains(string(replayed), `{"signature":"sig-xyz","thinking":"hmm","type":"thinking"}`) {
		t.Fatalf("signature not replayed: %s", replayed)
	}
}

func TestAnthropic_StreamErrorEvent(t *testing.T) {
	server, _ := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		fmt.Fprint(w, "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n")
	})
	err := anthropicClient(emptyOfficialModelsServer(t), server).Chat().Model("claude-test").User("q").
		Stream(func(ds4.ChatStreamChunk) error { return nil })
	var apiErr *ds4.APIError
	if !errors.As(err, &apiErr) || apiErr.Type != "overloaded_error" {
		t.Fatalf("expected overloaded APIError, got %v", err)
	}
}

func TestAnthropic_UnsupportedFeatures(t *testing.T) {
	server, bodies := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		fmt.Fprint(w, anthropicTextResponse)
	})
	client := anthropicClient(emptyOfficialModelsServer(t), server)
	if _, err := client.Chat().Model("claude-test").User("json").JSONMode().Do(); err == nil {
		t.Error("JSONMode should fail")
	}
	if _, err := client.Chat().Model("claude-test").User("q").LogProbs(2).Do(); err == nil {
		t.Error("LogProbs should fail")
	}
	if _, err := client.FIM().Model("claude-test").Prompt("x").Do(); err == nil {
		t.Error("FIM should fail")
	}
	if len(*bodies) != 0 {
		t.Fatalf("unsupported requests were sent: %d", len(*bodies))
	}
}

func TestAnthropic_RoutingPrecedenceAndStaticModels(t *testing.T) {
	var officialChats, anthropicModelCalls atomic.Int32
	official := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/models" {
			fmt.Fprint(w, `{"data":[{"id":"shared"},{"id":"only-official"}]}`)
			return
		}
		officialChats.Add(1)
		json.NewEncoder(w).Encode(chatFixture())
	}))
	defer official.Close()
	anthropic := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models" {
			anthropicModelCalls.Add(1)
		}
		if r.URL.Path == "/v1/messages" {
			fmt.Fprint(w, anthropicTextResponse)
		}
	}))
	defer anthropic.Close()

	client := ds4.New("k").WithBaseURL(official.URL).WithAnthropicURL(anthropic.URL).WithAnthropicModels("shared")
	resp, err := client.Chat().Model("shared").User("q").Do()
	if err != nil || resp.ID != "msg_1" {
		t.Fatalf("shared model should route to anthropic: %v %+v", err, resp)
	}
	if _, err := client.Chat().Model("only-official").User("q").Do(); err != nil || officialChats.Load() != 1 {
		t.Fatalf("official model misrouted: %v", err)
	}
	if anthropicModelCalls.Load() != 0 {
		t.Fatal("static models must skip /v1/models")
	}
}

func TestAnthropic_DiscoveryFailureIsIndependent(t *testing.T) {
	anthropic := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"type":"error","error":{"type":"authentication_error","message":"bad key"}}`, http.StatusUnauthorized)
	}))
	defer anthropic.Close()
	official := httptest.NewServer(withOfficialModels(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(chatFixture())
	}))
	defer official.Close()

	client := ds4.New("k").WithBaseURL(official.URL).WithAnthropicURL(anthropic.URL)
	if err := client.Ping(); err == nil || !strings.Contains(err.Error(), "bad key") {
		t.Fatalf("expected anthropic discovery error, got %v", err)
	}
	if _, err := client.Chat().User("q").Do(); err != nil {
		t.Fatalf("official model should still work: %v", err)
	}
}

func TestAnthropic_DatedModelsRouteByAlias(t *testing.T) {
	server, bodies := anthropicServer(t, func(w http.ResponseWriter, _ map[string]any) {
		fmt.Fprint(w, anthropicTextResponse)
	})
	client := anthropicClient(emptyOfficialModelsServer(t), server)
	for _, model := range []string{"claude-old", "claude-old-20251001"} {
		if _, err := client.Chat().Model(model).User("q").Do(); err != nil {
			t.Fatalf("%s: %v", model, err)
		}
	}
	if got := (*bodies)[0]["model"]; got != "claude-old" {
		t.Fatalf("alias should be sent unchanged, got %v", got)
	}
}

func TestAPIError_OverloadedIsRetryable(t *testing.T) {
	if !(&ds4.APIError{StatusCode: 529}).IsRetryable() {
		t.Fatal("529 should be retryable")
	}
}

func TestChatStreamAccumulator_OpenAIStream(t *testing.T) {
	var acc ds4.ChatStreamAccumulator
	stop := ds4.FinishReasonToolCalls
	acc.Add(ds4.ChatStreamChunk{Choices: []ds4.StreamChoice{{Delta: ds4.Delta{Content: "a", ToolCalls: []ds4.StreamToolCall{{Index: 1, ID: "b", Function: ds4.FunctionCall{Name: "g", Arguments: "{"}}}}}}})
	acc.Add(ds4.ChatStreamChunk{Choices: []ds4.StreamChoice{{Delta: ds4.Delta{Content: "b", ToolCalls: []ds4.StreamToolCall{{Index: 1, Function: ds4.FunctionCall{Arguments: "}"}}}}, FinishReason: &stop}}})
	message := acc.Message()
	if message.Content != "ab" || len(message.ToolCalls) != 2 || message.ToolCalls[1].Function.Arguments != "{}" || message.ToolCalls[1].ID != "b" || acc.FinishReason() != stop {
		t.Fatalf("unexpected message: %+v", message)
	}
}
