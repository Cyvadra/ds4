package ds4_test

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Cyvadra/ds4"
)

const integrationTimeout = 3 * time.Minute

func integrationClient(t *testing.T) (*ds4.Client, context.Context) {
	t.Helper()
	apiKey := os.Getenv("DEEPSEEK_API_KEY")
	if apiKey == "" {
		t.Skip("DEEPSEEK_API_KEY is not set")
	}
	ctx, cancel := context.WithTimeout(context.Background(), integrationTimeout)
	t.Cleanup(cancel)
	return ds4.New(apiKey).WithRetry(2), ctx
}

func requireChatResponse(t *testing.T, response *ds4.ChatResponse) {
	t.Helper()
	if response.ID == "" {
		t.Error("response ID is empty")
	}
	if response.Model == "" {
		t.Error("response model is empty")
	}
	if len(response.Choices) == 0 {
		t.Fatal("response has no choices")
	}
	if response.Usage.TotalTokens <= 0 {
		t.Errorf("expected token usage, got %+v", response.Usage)
	}
}

func integrationVisionImage(t *testing.T) ([]byte, string) {
	t.Helper()
	path := os.Getenv("DS4_VISION_IMAGE")
	if path == "" {
		t.Skip("DS4_VISION_IMAGE is not set")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read DS4_VISION_IMAGE %q: %v", path, err)
	}
	if len(data) == 0 {
		t.Fatalf("DS4_VISION_IMAGE %q is empty", path)
	}
	mediaType, ok := visionMediaType(filepath.Ext(path))
	if !ok {
		t.Fatalf("DS4_VISION_IMAGE %q must have a .jpg, .jpeg, .png, .gif, or .webp extension", path)
	}
	return data, mediaType
}

func visionMediaType(extension string) (string, bool) {
	switch strings.ToLower(extension) {
	case ".jpg", ".jpeg":
		return "image/jpeg", true
	case ".png":
		return "image/png", true
	case ".gif":
		return "image/gif", true
	case ".webp":
		return "image/webp", true
	default:
		return "", false
	}
}

// TestIntegration_Vision verifies that ChatBuilder sends a local image as an
// inline Base64 vision input and that DeepSeek processes it with the experimental
// vision model. It is intentionally opt-in because it makes a billable API call:
//
//	source .env
//	DS4_VISION_IMAGE=/absolute/path/to/image.jpg go test -run '^TestIntegration_Vision$' -v
//
// The prompt requires an exact marker in the answer so this test verifies model
// output rather than merely successful transport. Do not point DS4_VISION_IMAGE
// at a sensitive image because it is uploaded to DeepSeek.
func TestIntegration_Vision(t *testing.T) {
	client, ctx := integrationClient(t)
	image, mediaType := integrationVisionImage(t)

	response, err := client.Chat().
		Thinking(false).
		UserID("ds4_integration_vision").
		User("Describe the image in one concise sentence and include the exact word VISION_OK.").
		WithImageBase64(image, mediaType, ds4.ImageDetailOriginal).
		MaxTokens(128).
		DoWithContext(ctx)
	if err != nil {
		t.Fatal(err)
	}
	requireChatResponse(t, response)
	if response.Model != ds4.ModelDeepSeekV4FlashVisionExp {
		t.Errorf("expected vision model %q, got %q", ds4.ModelDeepSeekV4FlashVisionExp, response.Model)
	}
	if !strings.Contains(response.Content(), "VISION_OK") {
		t.Errorf("expected marker in vision response, got %q", response.Content())
	}
	if response.Content() == "" {
		t.Error("vision response content is empty")
	}
	t.Log(response.Content())
}

func TestIntegration_AllFeatures(t *testing.T) {
	client, ctx := integrationClient(t)

	t.Run("models", func(t *testing.T) {
		for _, model := range []string{ds4.ModelDeepSeekV4Flash, ds4.ModelDeepSeekV4Pro} {
			t.Run(model, func(t *testing.T) {
				response, err := client.Chat().
					Model(model).
					Thinking(false).
					UserID("ds4_integration_models").
					User("Reply with exactly: pong").
					MaxTokens(16).
					DoWithContext(ctx)
				if err != nil {
					t.Fatal(err)
				}
				requireChatResponse(t, response)
				if !strings.Contains(strings.ToLower(response.Content()), "pong") {
					t.Errorf("expected pong, got %q", response.Content())
				}
			})
		}
	})

	t.Run("thinking", func(t *testing.T) {
		response, err := client.Chat().
			Model(ds4.ModelDeepSeekV4Flash).
			ThinkingMode().
			ReasoningEffort(ds4.EffortLow).
			User("What is 17 + 25? Give only the number in the final answer.").
			MaxTokens(128).
			DoWithContext(ctx)
		if err != nil {
			t.Fatal(err)
		}
		requireChatResponse(t, response)
		if response.ReasoningContent() == "" {
			t.Error("thinking response has no reasoning_content")
		}
		if !strings.Contains(response.Content(), "42") {
			t.Errorf("expected final answer 42, got %q", response.Content())
		}
	})

	t.Run("multi-round", func(t *testing.T) {
		chat := client.Chat().Thinking(false).
			System("Remember the supplied code word and answer concisely.").
			User("The code word is cobalt. Acknowledge it.")
		first, err := chat.MaxTokens(32).DoWithContext(ctx)
		if err != nil {
			t.Fatal(err)
		}
		requireChatResponse(t, first)

		second, err := chat.AppendResponse(first).
			User("What was the code word? Reply with only that word.").
			DoWithContext(ctx)
		if err != nil {
			t.Fatal(err)
		}
		requireChatResponse(t, second)
		if !strings.Contains(strings.ToLower(second.Content()), "cobalt") {
			t.Errorf("conversation context was not preserved: %q", second.Content())
		}
	})

	t.Run("json", func(t *testing.T) {
		response, err := client.Chat().Thinking(false).JSONMode().
			System(`Return JSON matching {"ok": boolean, "value": integer}.`).
			User("Return json with ok true and value 7.").
			MaxTokens(64).
			DoWithContext(ctx)
		if err != nil {
			t.Fatal(err)
		}
		requireChatResponse(t, response)
		var value struct {
			OK    bool `json:"ok"`
			Value int  `json:"value"`
		}
		if err := json.Unmarshal([]byte(response.Content()), &value); err != nil {
			t.Fatalf("invalid JSON %q: %v", response.Content(), err)
		}
		if !value.OK || value.Value != 7 {
			t.Errorf("unexpected JSON value: %+v", value)
		}
	})

	t.Run("logprobs", func(t *testing.T) {
		response, err := client.Chat().Thinking(false).LogProbs(3).
			User("Reply with one word: blue").
			MaxTokens(16).
			DoWithContext(ctx)
		if err != nil {
			t.Fatal(err)
		}
		requireChatResponse(t, response)
		if response.LogProbs() == nil || len(response.LogProbs().Content) == 0 {
			t.Fatal("response has no logprobs content")
		}
		if response.LogProbs().Content[0].Token == "" {
			t.Error("first logprobs token is empty")
		}
	})

	t.Run("stream", func(t *testing.T) {
		var content strings.Builder
		var usage *ds4.Usage
		var finished bool
		err := client.Chat().Thinking(false).IncludeStreamUsage().
			User("Reply with exactly: streamed pong").
			MaxTokens(32).
			StreamWithContext(ctx, func(chunk ds4.ChatStreamChunk) error {
				if chunk.Usage != nil {
					usage = chunk.Usage
				}
				for _, choice := range chunk.Choices {
					content.WriteString(choice.Delta.Content)
					if choice.FinishReason != nil {
						finished = true
					}
				}
				return nil
			})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(strings.ToLower(content.String()), "streamed pong") {
			t.Errorf("unexpected streamed content %q", content.String())
		}
		if !finished {
			t.Error("stream had no finish reason")
		}
		if usage == nil || usage.TotalTokens <= 0 {
			t.Errorf("stream had no final usage: %+v", usage)
		}
	})

	t.Run("tool-round-trip", func(t *testing.T) {
		weather := ds4.NewFunction("get_weather", "Get weather for a city", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"city": map[string]any{"type": "string"},
			},
			"required": []string{"city"},
		})
		chat := client.Chat().Thinking(false).Tool(weather).
			ToolChoice(map[string]any{"type": "function", "function": map[string]any{"name": "get_weather"}}).
			User("What is the weather in Tokyo?")
		first, err := chat.MaxTokens(128).DoWithContext(ctx)
		if err != nil {
			t.Fatal(err)
		}
		requireChatResponse(t, first)
		if first.FinishReason() != ds4.FinishReasonToolCalls || len(first.ToolCalls()) == 0 {
			t.Fatalf("expected tool call, finish=%q calls=%+v", first.FinishReason(), first.ToolCalls())
		}
		call := first.ToolCalls()[0]
		if call.ID == "" || call.Function.Name != "get_weather" || !json.Valid([]byte(call.Function.Arguments)) {
			t.Fatalf("invalid tool call: %+v", call)
		}

		second, err := chat.AppendResponse(first).
			ToolResult(call.ID, `{"temperature_c":24,"condition":"sunny"}`).
			ToolChoice("auto").
			DoWithContext(ctx)
		if err != nil {
			t.Fatal(err)
		}
		requireChatResponse(t, second)
		if second.Content() == "" {
			t.Error("tool result produced no final answer")
		}
	})

	t.Run("strict-tool", func(t *testing.T) {
		tool := ds4.NewStrictFunction("lookup_id", "Look up an integer ID", map[string]any{
			"type": "object",
			"properties": map[string]any{
				"id": map[string]any{"type": "integer"},
			},
			"required":             []string{"id"},
			"additionalProperties": false,
		})
		response, err := client.Chat().Thinking(false).Tool(tool).
			ToolChoice(map[string]any{"type": "function", "function": map[string]any{"name": "lookup_id"}}).
			User("Look up ID 37.").
			MaxTokens(128).
			DoWithContext(ctx)
		if err != nil {
			t.Fatal(err)
		}
		requireChatResponse(t, response)
		if len(response.ToolCalls()) == 0 {
			t.Fatal("strict tool produced no tool call")
		}
		var arguments struct {
			ID int `json:"id"`
		}
		if err := json.Unmarshal([]byte(response.ToolCalls()[0].Function.Arguments), &arguments); err != nil {
			t.Fatal(err)
		}
		if arguments.ID != 37 {
			t.Errorf("strict tool returned ID %d", arguments.ID)
		}
	})

	t.Run("prefix", func(t *testing.T) {
		response, err := client.Chat().Thinking(false).
			User("Complete the arithmetic expression.").
			Prefix("1 + 1 = ").
			MaxTokens(16).
			DoWithContext(ctx)
		if err != nil {
			t.Fatal(err)
		}
		requireChatResponse(t, response)
		if !strings.Contains(response.Content(), "2") {
			t.Errorf("expected prefix completion to contain 2, got %q", response.Content())
		}
	})

	t.Run("fim", func(t *testing.T) {
		response, err := client.FIM().
			Prompt("func add(a, b int) int {\n\treturn ").
			Suffix("\n}").
			MaxTokens(32).
			DoWithContext(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if response.ID == "" || len(response.Choices) == 0 || response.Text() == "" {
			t.Fatalf("invalid FIM response: %+v", response)
		}
	})

	t.Run("fim-stream", func(t *testing.T) {
		var content strings.Builder
		var finished bool
		err := client.FIM().
			Prompt("func multiply(a, b int) int {\n\treturn ").
			Suffix("\n}").
			MaxTokens(32).
			StreamWithContext(ctx, func(chunk ds4.FIMStreamChunk) error {
				for _, choice := range chunk.Choices {
					content.WriteString(choice.Text)
					if choice.FinishReason != nil {
						finished = true
					}
				}
				return nil
			})
		if err != nil {
			t.Fatal(err)
		}
		if content.Len() == 0 || !finished {
			t.Errorf("incomplete FIM stream: content=%q finished=%v", content.String(), finished)
		}
	})
}
