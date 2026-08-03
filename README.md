# ds4

A fluent Go client for the [DeepSeek API](https://api-docs.deepseek.com/).

## Install

```sh
go get github.com/Cyvadra/ds4
```

## Quick start

```go
import "github.com/Cyvadra/ds4"

client := ds4.New(os.Getenv("DEEPSEEK_API_KEY"))
```

New chats include the system prompt `You are a helpful assistant.`. Pass a
second argument to `New` to set a client-specific default; `System` replaces
that prompt for an individual chat.

```go
client := ds4.New(os.Getenv("DEEPSEEK_API_KEY"), "Answer concisely in Chinese.")
```

Use `SystemAppend` when the API should receive an additional system message
without replacing the first prompt:

```go
resp, err := client.Chat().
    SystemAppend("Use metric units.").
    User("How far is a marathon?").
    Do()
```

---

## Chat completions

```go
resp, err := client.Chat().
    User("What is Go?").
    Do()
fmt.Println(resp.Content())
```

### Multi-round chat

```go
b := client.Chat().System("You are helpful assistant.")

resp1, _ := b.User("What is a goroutine?").Do()
resp2, _ := b.Assistant(resp1.Content()).User("How do I cancel one?").Do()
fmt.Println(resp2.Content())
```

### Thinking mode

Thinking is **enabled by default** by the API, with effort `high`. Reasoning
tokens are billed at output rates, so disable it explicitly for latency- or
cost-sensitive calls.

```go
resp, err := client.Chat().
    ThinkingMode().
    ReasoningEffort(ds4.EffortMax).
    User("Prove that sqrt(2) is irrational.").
    Do()
fmt.Println("Reasoning:", resp.ReasoningContent())
fmt.Println("Answer:   ", resp.Content())
```

`ThinkingMode()` toggles reasoning on the model you already selected; it never
changes the model. To turn reasoning off:

```go
resp, err := client.Chat().Thinking(false).User("Ping").Do()
```

Thinking mode ignores `Temperature` and `TopP`.

### JSON mode

```go
resp, err := client.Chat().
    JSONMode().
    User(`Return {"name": "Alice", "age": 30} as JSON`).
    Do()
fmt.Println(resp.Content()) // {"name":"Alice","age":30}
```

Include `json` in the system or user prompt and reserve enough `MaxTokens` for a complete object.

### Chat prefix completion (beta)

The model continues generating from the supplied prefix. The client routes this beta feature through `/beta/chat/completions`.

```go
resp, err := client.Chat().
    User("Write a Go hello-world function").
    Prefix("func helloWorld() {\n").
    Do()
fmt.Println(resp.Content())
```

To continue an existing assistant response, append it to the conversation and
call `Continue`. The argument is added to the current `MaxTokens` value for
that request; the final message must have the `assistant` role.

```go
chat := client.Chat().MaxTokens(128).User("Tell a story")
resp, err := chat.Do()
if err != nil {
    return err
}

more, err := chat.AppendResponse(resp).Continue(128).Do()
```

### Tool calls (function calling)

```go
schema := map[string]any{
    "type": "object",
    "properties": map[string]any{
        "city": map[string]any{"type": "string", "description": "City name"},
    },
    "required": []string{"city"},
}
weather := ds4.NewFunction("get_weather", "Get current weather for a city", schema)

resp, err := client.Chat().
    Tool(weather).
    User("What's the weather like in Tokyo?").
    Do()

if resp.FinishReason() == ds4.FinishReasonToolCalls {
    for _, tc := range resp.ToolCalls() {
        fmt.Printf("call %s(%s)\n", tc.Function.Name, tc.Function.Arguments)
    }
}
```

Use `ds4.NewStrictFunction` to enforce a tool's JSON Schema. Strict tools are beta features and are automatically routed through `/beta/chat/completions`.

To continue after a tool call, append the full assistant response before adding each tool result:

```go
chat := client.Chat().Tool(weather).User("What's the weather like in Tokyo?")
resp, err := chat.Do()
if err != nil {
    return err
}
chat.AppendResponse(resp)
for _, call := range resp.ToolCalls() {
    chat.ToolResult(call.ID, `{"temperature": 24}`)
}
resp, err = chat.Do()
```

### Streaming

```go
err := client.Chat().
    User("Tell me a short story.").
    Stream(func(chunk ds4.ChatStreamChunk) error {
        if len(chunk.Choices) > 0 {
            fmt.Print(chunk.Choices[0].Delta.Content)
        }
        return nil
    })
```

---

## FIM (Fill-In-the-Middle) – beta

```go
resp, err := client.FIM().
    Prompt("func greet(name string) string {\n    return ").
    Suffix("\n}").
    MaxTokens(64).
    Do()
fmt.Println(resp.Text())
```

FIM `MaxTokens` must be between `0` and `4096`.

### FIM streaming

```go
err := client.FIM().
    Prompt("func add(a, b int) int {\n    return ").
    Suffix("\n}").
    Stream(func(chunk ds4.FIMStreamChunk) error {
        if len(chunk.Choices) > 0 {
            fmt.Print(chunk.Choices[0].Text)
        }
        return nil
    })
```

---

## Builder options

| Method | Description |
|--------|-------------|
| `Model(name)` | Override the model (default: `deepseek-v4-flash`) |
| `Temperature(t)` | Sampling temperature 0–2 (ignored in thinking mode) |
| `TopP(p)` | Nucleus sampling 0–1 (ignored in thinking mode) |
| `MaxTokens(n)` | Max tokens to generate |
| `Stop(seq...)` | Stop sequences |
| `LogProbs(topN)` | Return log probabilities, `topN` 0–20 |
| `UserID(id)` | Tag the request for cache isolation and safety review |
| `IncludeStreamUsage()` | Include token usage in final stream chunk |
| `Thinking(enabled)` | Enable or disable thinking output |
| `ReasoningEffort(level)` | Request `low`, `high`, or `max` thinking effort |
| `Clone()` | Deep-copy builder (fork a conversation) |

Read log probabilities from the response:

```go
resp, _ := client.Chat().LogProbs(5).User("Hi").Do()
for _, tok := range resp.LogProbs().Content {
    fmt.Println(tok.Token, tok.LogProb)
}
```

`frequency_penalty` and `presence_penalty` are deprecated by the API and have
no effect, so this client does not send them.

## Model constants

```go
ds4.ModelDeepSeekV4Flash // "deepseek-v4-flash" (default)
ds4.ModelDeepSeekV4Pro   // "deepseek-v4-pro"
```

Effort levels: `ds4.EffortLow`, `ds4.EffortHigh` (API default), `ds4.EffortMax`.

## Context support

The client sets no overall request deadline, because that would also cut off
streamed responses mid-generation. Bound individual calls with the
`...WithContext(ctx)` variant of every `Do()` / `Stream()` method:

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

resp, err := client.Chat().User("Hello").DoWithContext(ctx)
```

## Error handling

Network and API errors are returned as standard Go `error` values.
API errors can be inspected via type assertion:

```go
if apiErr, ok := err.(*ds4.APIError); ok {
    fmt.Println(apiErr.StatusCode, apiErr.Message)
}
```

Retries are opt-in and apply only to retryable statuses (429, 500, 503), using
exponential backoff with jitter and honouring `Retry-After`:

```go
client := ds4.New(apiKey).WithRetry(3)
```

## Real API tests

The integration suite makes billable requests against every supported feature,
including both V4 models, thinking, JSON, logprobs, streaming, tools, prefix
completion, and FIM. It skips when `DEEPSEEK_API_KEY` is absent.

```sh
DEEPSEEK_API_KEY=your_key go test -run '^TestIntegration_AllFeatures$' -v
```
