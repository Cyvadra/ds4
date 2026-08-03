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

---

## Chat completions

```go
resp, err := client.Chat().
    System("You are a helpful assistant.").
    User("What is Go?").
    Do()
fmt.Println(resp.Content())
```

### Multi-round chat

```go
b := client.Chat().System("You are helpful.")

resp1, _ := b.User("What is a goroutine?").Do()
resp2, _ := b.Assistant(resp1.Content()).User("How do I cancel one?").Do()
fmt.Println(resp2.Content())
```

### Thinking mode (deepseek-reasoner)

```go
resp, err := client.Chat().
    ThinkingMode().
    User("Prove that sqrt(2) is irrational.").
    Do()
fmt.Println("Reasoning:", resp.ReasoningContent())
fmt.Println("Answer:   ", resp.Content())
```

### JSON mode

```go
resp, err := client.Chat().
    JSONMode().
    User(`Return {"name": "Alice", "age": 30} as JSON`).
    Do()
fmt.Println(resp.Content()) // {"name":"Alice","age":30}
```

### Chat prefix completion (beta)

The model continues generating from the supplied prefix:

```go
resp, err := client.Chat().
    User("Write a Go hello-world function").
    Prefix("func helloWorld() {\n").
    Do()
fmt.Println(resp.Content())
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
| `Temperature(t)` | Sampling temperature 0–2 |
| `TopP(p)` | Nucleus sampling 0–1 |
| `MaxTokens(n)` | Max tokens to generate |
| `Stop(seq...)` | Stop sequences |
| `FrequencyPenalty(p)` | Reduce repetition |
| `PresencePenalty(p)` | Encourage new topics |
| `LogProbs(topN)` | Return log probabilities |
| `IncludeStreamUsage()` | Include token usage in final stream chunk |
| `Clone()` | Deep-copy builder (fork a conversation) |

## Model constants

```go
ds4.ModelDeepSeekV4Flash  // "deepseek-v4-flash"  (default)
ds4.ModelDeepSeekChat     // "deepseek-chat"
ds4.ModelDeepSeekReasoner // "deepseek-reasoner"  (thinking mode)
ds4.ModelDeepSeekCoder    // "deepseek-coder"
```

## Context support

Every `Do()` / `Stream()` method has a `...WithContext(ctx)` variant:

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
