// Command ds4check exercises a real DeepSeek or OpenAI-compatible deployment.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/Cyvadra/ds4"
)

type check func(context.Context, *ds4.Client, string) (string, error)

type checkResult struct {
	name string
	err  error
}

func main() {
	apiKey := flag.String("api-key", os.Getenv("DEEPSEEK_API_KEY"), "DeepSeek API key")
	baseURL := flag.String("base-url", os.Getenv("DEEPSEEK_BASE_URL"), "official API base URL")
	localURL := flag.String("local-url", os.Getenv("DS4_LOCAL_URL"), "local OpenAI-compatible API base URL")
	localKey := flag.String("local-key", os.Getenv("DS4_LOCAL_KEY"), "local API key")
	model := flag.String("model", os.Getenv("DS4_MODEL"), "model to test (required)")
	tests := flag.String("tests", "models,chat,json,stream,tools", "comma-separated checks: models,chat,json,stream,tools,thinking,prefix,fim,all")
	timeout := flag.Duration("timeout", 3*time.Minute, "timeout for each check")
	retries := flag.Int("retries", 1, "retry count for retryable API failures")
	flag.Parse()

	if *apiKey == "" && *localURL == "" {
		fail("supply -api-key (or DEEPSEEK_API_KEY), or -local-url for a local server")
	}
	if *model == "" {
		fail("supply -model (or DS4_MODEL); this prevents unexpected billable model selection")
	}

	client := ds4.New(*apiKey).WithRetry(*retries)
	if *baseURL != "" {
		client.WithBaseURL(*baseURL)
	}
	if *localURL != "" {
		client.WithLocalKey(*localKey).WithLocalURL(*localURL).WithLocalFIM(true)
	}

	selected, err := parseChecks(*tests)
	if err != nil {
		fail(err.Error())
	}
	checks := map[string]check{
		"models":   checkModels,
		"chat":     checkChat,
		"json":     checkJSON,
		"stream":   checkStream,
		"tools":    checkTools,
		"thinking": checkThinking,
		"prefix":   checkPrefix,
		"fim":      checkFIM,
	}

	results := make([]checkResult, 0, len(selected))
	for _, name := range selected {
		ctx, cancel := context.WithTimeout(context.Background(), *timeout)
		fmt.Printf("[%s]\n", name)
		output, err := checks[name](ctx, client, *model)
		cancel()
		if output != "" {
			fmt.Println(output)
		}
		fmt.Println()
		results = append(results, checkResult{name: name, err: err})
	}

	failed := false
	fmt.Println("Results:")
	for _, result := range results {
		if result.err != nil {
			failed = true
			fmt.Printf("[%s] FAIL: %v\n", result.name, result.err)
			continue
		}
		fmt.Printf("[%s] PASS\n", result.name)
	}
	if failed {
		os.Exit(1)
	}
}

func parseChecks(value string) ([]string, error) {
	available := []string{"models", "chat", "json", "stream", "tools", "thinking", "prefix", "fim"}
	seen := make(map[string]bool)
	var selected []string
	for _, name := range strings.Split(value, ",") {
		name = strings.TrimSpace(strings.ToLower(name))
		if name == "all" {
			return available, nil
		}
		found := false
		for _, availableName := range available {
			if name == availableName {
				found = true
				if !seen[name] {
					selected = append(selected, name)
					seen[name] = true
				}
			}
		}
		if !found {
			return nil, fmt.Errorf("unknown check %q", name)
		}
	}
	if len(selected) == 0 {
		return nil, fmt.Errorf("at least one check is required")
	}
	return selected, nil
}

func checkModels(ctx context.Context, client *ds4.Client, _ string) (string, error) {
	return "Model discovery completed.", client.Ping()
}

func checkChat(ctx context.Context, client *ds4.Client, model string) (string, error) {
	response, err := client.Chat().Model(model).Thinking(false).User("Reply with exactly: ds4-ok").MaxTokens(16).DoWithContext(ctx)
	if err != nil {
		return "", err
	}
	output := formatChatResponse(response)
	if !strings.Contains(strings.ToLower(response.Content()), "ds4-ok") {
		return output, fmt.Errorf("expected ds4-ok, got %q", response.Content())
	}
	return output, nil
}

func checkJSON(ctx context.Context, client *ds4.Client, model string) (string, error) {
	response, err := client.Chat().Model(model).Thinking(false).JSONMode().System(`Return JSON matching {"ok": boolean}.`).User("Return JSON with ok true.").MaxTokens(32).DoWithContext(ctx)
	if err != nil {
		return "", err
	}
	output := formatChatResponse(response)
	var value struct {
		OK bool `json:"ok"`
	}
	if err := json.Unmarshal([]byte(response.Content()), &value); err != nil || !value.OK {
		return output, fmt.Errorf("expected JSON {ok:true}, got %q", response.Content())
	}
	return output, nil
}

func checkStream(ctx context.Context, client *ds4.Client, model string) (string, error) {
	var content strings.Builder
	err := client.Chat().Model(model).Thinking(false).User("Reply with exactly: ds4-stream-ok").MaxTokens(16).StreamWithContext(ctx, func(chunk ds4.ChatStreamChunk) error {
		for _, choice := range chunk.Choices {
			content.WriteString(choice.Delta.Content)
		}
		return nil
	})
	if err != nil {
		return content.String(), err
	}
	if !strings.Contains(strings.ToLower(content.String()), "ds4-stream-ok") {
		return content.String(), fmt.Errorf("expected ds4-stream-ok, got %q", content.String())
	}
	return content.String(), nil
}

func checkTools(ctx context.Context, client *ds4.Client, model string) (string, error) {
	tool := ds4.NewFunction("lookup_id", "Look up an integer ID", map[string]any{
		"type": "object", "properties": map[string]any{"id": map[string]any{"type": "integer"}}, "required": []string{"id"},
	})
	response, err := client.Chat().Model(model).Thinking(false).Tool(tool).ToolChoice(map[string]any{"type": "function", "function": map[string]any{"name": "lookup_id"}}).User("Look up ID 37.").MaxTokens(64).DoWithContext(ctx)
	if err != nil {
		return "", err
	}
	output := formatChatResponse(response)
	if len(response.ToolCalls()) == 0 || !json.Valid([]byte(response.ToolCalls()[0].Function.Arguments)) {
		return output, fmt.Errorf("expected a tool call, got %+v", response.ToolCalls())
	}
	return output, nil
}

func checkThinking(ctx context.Context, client *ds4.Client, model string) (string, error) {
	response, err := client.Chat().Model(model).ThinkingMode().ReasoningEffort(ds4.EffortLow).User("What is 17 + 25? Give only the number in the final answer.").MaxTokens(128).DoWithContext(ctx)
	if err != nil {
		return "", err
	}
	output := formatChatResponse(response)
	if !strings.Contains(response.Content(), "42") {
		return output, fmt.Errorf("expected 42, got %q", response.Content())
	}
	if response.ReasoningContent() == "" {
		return output, fmt.Errorf("expected reasoning content, got none")
	}

	var streamedReasoning, streamedContent strings.Builder
	err = client.Chat().Model(model).ThinkingMode().ReasoningEffort(ds4.EffortLow).User("What is 17 + 25? Give only the number in the final answer.").MaxTokens(128).StreamWithContext(ctx, func(chunk ds4.ChatStreamChunk) error {
		for _, choice := range chunk.Choices {
			streamedReasoning.WriteString(choice.Delta.ReasoningContent)
			streamedContent.WriteString(choice.Delta.Content)
		}
		return nil
	})
	streamOutput := fmt.Sprintf("stream reasoning: %s\nstream content: %s", streamedReasoning.String(), streamedContent.String())
	if err != nil {
		return output + "\n" + streamOutput, err
	}
	if streamedReasoning.Len() == 0 {
		return output + "\n" + streamOutput, fmt.Errorf("expected streamed reasoning content, got none")
	}
	if !strings.Contains(streamedContent.String(), "42") {
		return output + "\n" + streamOutput, fmt.Errorf("expected streamed answer 42, got %q", streamedContent.String())
	}
	output += "\n" + streamOutput
	return output, nil
}

func checkPrefix(ctx context.Context, client *ds4.Client, model string) (string, error) {
	response, err := client.Chat().Model(model).Thinking(false).User("Complete the arithmetic expression.").Prefix("1 + 1 = ").MaxTokens(16).DoWithContext(ctx)
	if err != nil {
		return "", err
	}
	output := formatChatResponse(response)
	if !strings.Contains(response.Content(), "2") {
		return output, fmt.Errorf("expected 2, got %q", response.Content())
	}
	return output, nil
}

func checkFIM(ctx context.Context, client *ds4.Client, model string) (string, error) {
	response, err := client.FIM().Model(model).Prompt("func add(a, b int) int {\n\treturn ").Suffix("\n}").MaxTokens(32).DoWithContext(ctx)
	if err != nil {
		return "", err
	}
	if response.Text() == "" {
		return formatFIMResponse(response), fmt.Errorf("received an empty FIM response")
	}
	return formatFIMResponse(response), nil
}

func formatChatResponse(response *ds4.ChatResponse) string {
	data, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		return fmt.Sprintf("response: %+v", response)
	}
	return string(data)
}

func formatFIMResponse(response *ds4.FIMResponse) string {
	data, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		return fmt.Sprintf("response: %+v", response)
	}
	return string(data)
}

func fail(message string) {
	fmt.Fprintln(os.Stderr, "ds4check:", message)
	os.Exit(2)
}
