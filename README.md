# ds4

A fluent Go client for the [DeepSeek API](https://api-docs.deepseek.com/) with
model-aware routing for optional OpenAI-compatible local deployments and
Anthropic Messages API servers.

## Install

```sh
go get github.com/Cyvadra/ds4
```

## Quick start

```go
client := ds4.New(os.Getenv("DEEPSEEK_API_KEY"))

resp, err := client.Chat().
    User("What is Go?").
    Do()
if err != nil {
    return err
}
fmt.Println(resp.Content())
```

New chats begin with `You are a helpful assistant.`. Set a client-wide prompt
when creating the client, replace it for one conversation with `System`, or
add another system message with `SystemAppend`.

```go
client := ds4.New(apiKey, "Answer concisely in Chinese.")

resp, err := client.Chat().
    System("You are a patient Go tutor.").
    SystemAppend("Use metric units.").
    User("How far is a marathon?").
    Do()
```

## Configuration reference

| Surface | Field or method | Default | Purpose |
|---|---|---|---|
| Client | `New(apiKey, defaultSystemPrompt...)` | `DefaultSystemPrompt` | Sets the official API credential and optional default system prompt. |
| Client | `WithBaseURL(url)` | `https://api.deepseek.com` | Uses an alternate official API base URL, such as a proxy. |
| Client | `WithLocalURL(url)` | disabled | Adds an OpenAI-compatible server to model routing. |
| Client | `WithLocalKey(key)` | empty | Sets the credential for the local server. |
| Client | `WithLocalFIM(enabled)` | `false` | Enables local FIM requests through `/completions`. |
| Client | `WithAnthropicURL(url)` | disabled | Adds an Anthropic Messages API server (base URL without `/v1`) to model routing. |
| Client | `WithAnthropicKey(key)` | empty | Sets the `x-api-key` credential for the Anthropic backend. |
| Client | `WithAnthropicVersion(v)` | `2023-06-01` | Sets the `anthropic-version` header. |
| Client | `WithAnthropicWorkspace(id)` | empty | Sends `anthropic-workspace-id`, required for keys not scoped to a workspace. |
| Client | `WithAnthropicModels(names...)` | discover via `/v1/models` | Registers Anthropic-backend models statically, for servers without a models endpoint. |
| Client | `WithHTTPClient(client)` | built-in `net/http` client | Replaces the HTTP transport; avoid a client-wide timeout because it limits streams. |
| Client | `WithRetry(n)` | `0` | Retries retryable 429, 500, and 503 responses up to `n` additional times. |
| Chat | `System(content)`, `SystemAppend(content)`, `User(content)`, `Assistant(content)`, `ToolResult(id, content)`, `AddMessage(message)`, `WithMessages(messages)` | one default system message | Composes or replaces the conversation history. |
| Chat and FIM | `Model(name)` | `deepseek-flash` | Selects an advertised model. Official constants: `ModelDeepSeekV4Flash`, `ModelDeepSeekV4Pro`, `ModelDeepSeekV4FlashVisionExp`. |
| Chat and FIM | `MaxTokens(n)` | API default | Limits generated tokens. FIM accepts `0` through `4096`. |
| Chat and FIM | `Temperature(t)` | API default | Sets sampling temperature from `0` through `2`; ignored while thinking is enabled. |
| Chat and FIM | `TopP(p)` | API default | Sets nucleus sampling from `0` through `1`; ignored while thinking is enabled. |
| Chat and FIM | `Stop(seq...)` | none | Stops generation when a sequence is produced. |
| Chat | `Thinking(enabled)` / `ThinkingMode()` | enabled by API | Controls reasoning output on the selected model. |
| Chat | `ReasoningEffort(level)` | `high` | Requests `EffortLow`, `EffortHigh`, or `EffortMax`. |
| Chat | `JSONMode()` | text | Requests a JSON object response. |
| Chat | `LogProbs(topN)` | disabled | Returns token log probabilities with `topN` from `0` through `20`. |
| Chat | `Tool(tool)`, `WithTools(tools)`, `ToolChoice(choice)` | no tools | Adds or replaces function tools and optionally selects one. |
| Chat | `IncludeStreamUsage()` | disabled | Includes token usage in the final streaming chunk. |
| Chat | `UserID(id)` | empty | Tags a request for cache isolation and safety review; do not use personal information. |
| Chat | `WithImageURL(url, detail...)` | none | Attaches an HTTP(S) image URL to the final user message and automatically uses the vision model. |
| Chat | `WithImageBase64(data, mediaType, detail...)` | none | Encodes raw JPEG, PNG, GIF, or WebP bytes as an image attachment and automatically uses the vision model. |
| FIM | `Prompt(text)` / `Suffix(text)` | prompt required; no suffix | Sets the text before and after the insertion point. |
| FIM | `Echo()` | disabled | Includes the prompt in the response text. |
| `ds4check` | `-api-key`, `-base-url`, `-local-url`, `-local-key`, `-anthropic-url`, `-anthropic-key`, `-anthropic-models`, `-model` | environment equivalents or empty | Selects the deployment and an explicitly named model. Environment equivalents are `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DS4_LOCAL_URL`, `DS4_LOCAL_KEY`, `DS4_ANTHROPIC_URL`, `DS4_ANTHROPIC_KEY` (falling back to `ANTHROPIC_API_KEY`), `DS4_ANTHROPIC_MODELS`, and `DS4_MODEL`. |
| `ds4check` | `-tests`, `-timeout`, `-retries` | `models,chat,json,stream,tools`; `3m`; `1` | Selects checks, sets a per-check deadline, and retries transient API failures. |

## Anthropic Messages backend

`WithAnthropicURL` adds a server that speaks the Anthropic Messages API
(`POST /v1/messages`). It works with Claude and with DeepSeek's
Anthropic-compatible endpoint. Requests are routed by model name like the other
backends; when several backends advertise a name, local wins, then Anthropic,
then official.

```go
client := ds4.New(os.Getenv("DEEPSEEK_API_KEY")).
    WithAnthropicURL("https://api.anthropic.com").
    WithAnthropicKey(os.Getenv("ANTHROPIC_API_KEY"))

resp, err := client.Chat().Model("claude-opus-5-5").User("Hello").Do()

// Keys that are not scoped to one workspace must name the workspace,
// otherwise every request fails with HTTP 400.
client.WithAnthropicWorkspace("wrkspc_...")

// DeepSeek's endpoint has no /v1/models, so register its models explicitly.
deepseek := ds4.New("").
    WithAnthropicURL("https://api.deepseek.com/anthropic").
    WithAnthropicKey(os.Getenv("DEEPSEEK_API_KEY")).
    WithAnthropicModels(ds4.ModelDeepSeekV4Pro)
```

The same builder methods and response types are used:

| Builder feature | Anthropic request |
|---|---|
| System messages | joined in order into top-level `system` |
| `MaxTokens` | `max_tokens`, default `DefaultAnthropicMaxTokens` (16000) |
| `Stop`, `UserID` | `stop_sequences`, `metadata.user_id` |
| `Thinking(true)` / `ThinkingMode()` | `thinking: {type: "adaptive", display: "summarized"}` |
| `Thinking(false)` | `thinking: {type: "disabled"}` |
| `ReasoningEffort(level)` | `output_config.effort` |
| Tools, `ToolCalls`, `ToolResult` | `tools`, `tool_use`, `tool_result` blocks |
| `ToolChoice` `"auto"`/`"none"`/`"required"`/function map | `auto`/`none`/`any`/`tool` |
| `WithImageURL` / `WithImageBase64` | `image` blocks; the selected model is kept |

**Workspace:** an Anthropic API key that is not scoped to a single workspace
is rejected with `HTTP 400: This API key is not scoped to a workspace` unless
you call `WithAnthropicWorkspace` (or pass `-anthropic-workspace` /
`DS4_ANTHROPIC_WORKSPACE` to `ds4check`). Find the ID under Workspaces in the
Claude Console.

**Model names:** `/v1/models` lists some models only by dated ID
(`claude-haiku-4-5-20251001`). ds4 also registers the undated alias
(`claude-haiku-4-5`), so either name routes to the Anthropic backend.

`JSONMode`, `LogProbs`, and FIM return an error on this backend. Responses
report thinking as `ReasoningContent` and map `stop_reason` to the usual
finish reasons.

To continue a tool-use loop with thinking, append the assistant message with
`AppendResponse`. For a stream, collect the chunks with `ChatStreamAccumulator`
and pass `acc.Message()` to `AddMessage`. Both keep the original content blocks,
including thinking signatures, and replay them verbatim. JSON serialization of a
`Message` does not keep them.

Current Claude models reject some settings: `Thinking(false)`, `Temperature`,
`TopP`, forced `ToolChoice`, and assistant prefill (`Prefix`, `Continue`). On
those models, use `ReasoningEffort` and `"auto"` tool choice instead. Older
models without adaptive thinking, such as Claude Haiku 4.5, reject
`ThinkingMode()`; leave thinking unset for them.

## Chat completions

### Image understanding

`WithImageURL` and `WithImageBase64` add images to the current request. They
may appear anywhere in the fluent chain: when the request is built, ds4 adds
all attached images to the final `User` message and switches the model to
`ModelDeepSeekV4FlashVisionExp`. This model choice overrides an earlier
`Model(...)` call. A request with images must contain at least one user message.

Attach a public image URL:

```go
resp, err := client.Chat().
    WithImageURL("https://example.com/chart.png", ds4.ImageDetailOriginal).
    User("Summarize this chart and list the largest changes.").
    Do()
if err != nil {
    return err
}
fmt.Println(resp.Content())
```

For local images, pass raw bytes and one of `image/jpeg`, `image/png`,
`image/gif`, or `image/webp`. ds4 creates the required Base64 data URL.

```go
image, err := os.ReadFile("screenshot.png")
if err != nil {
    return err
}
resp, err := client.Chat().
    User("Read the visible error and suggest a fix.").
    WithImageBase64(image, "image/png", ds4.ImageDetailOriginal).
    Do()
```

Calls append images, so URL and Base64 input can be mixed in one request:

```go
resp, err := client.Chat().
    User("Compare these two interfaces for accessibility.").
    WithImageURL("https://example.com/before.png", ds4.ImageDetailLow).
    WithImageBase64(after, "image/png", ds4.ImageDetailOriginal).
    Do()
```

The optional detail argument accepts `ImageDetailLow`, `ImageDetailHigh`,
`ImageDetailOriginal`, or `ImageDetailAuto`. `ImageDetailLow` downscales to
512x512 before inference and is useful for inexpensive general inspection;
`ImageDetailOriginal` and `ImageDetailHigh` preserve detail for OCR and charts.
Omit detail to use the API default.

DeepSeek accepts JPEG, PNG, GIF, and WebP. ds4 validates up to 600 images per
request and the 32 MiB maximum for each inline Base64 image. HTTP(S) URLs are
also limited by DeepSeek to 8192 characters, a 32 MiB image download, and a
60-second download window. The full HTTP body must be at most 48 MiB; use the
official Files API directly for images larger than 32 MiB or for reusable files.

### Multi-round conversations

Reuse a builder to retain the conversation history. `AppendResponse` preserves
the complete assistant message, including reasoning and tool calls, for the
next turn.

```go
chat := client.Chat().System("You are a helpful assistant.")

first, err := chat.User("What is a goroutine?").Do()
if err != nil {
    return err
}

second, err := chat.AppendResponse(first).
    User("How do I cancel one?").
    Do()
if err != nil {
    return err
}
fmt.Println(second.Content())
```

Use `Clone()` to fork a conversation without sharing its message history.

### Thinking mode

The API enables thinking by default and uses `high` effort. Reasoning tokens
are billed at output rates, so disable it for latency- or cost-sensitive calls.

```go
resp, err := client.Chat().
    ThinkingMode().
    ReasoningEffort(ds4.EffortMax).
    User("Prove that sqrt(2) is irrational.").
    Do()
if err != nil {
    return err
}
fmt.Println("Reasoning:", resp.ReasoningContent())
fmt.Println("Answer:   ", resp.Content())
```

`ThinkingMode()` toggles reasoning without changing the chosen model. Disable
it explicitly with `Thinking(false)`.

### JSON mode

```go
resp, err := client.Chat().
    JSONMode().
    User(`Return {"name": "Alice", "age": 30} as JSON`).
    Do()
if err != nil {
    return err
}
fmt.Println(resp.Content()) // {"name":"Alice","age":30}
```

Include `json` in the system or user prompt and allow enough `MaxTokens` for a
complete object.

### Tool calls

```go
schema := map[string]any{
    "type": "object",
    "properties": map[string]any{
        "city": map[string]any{"type": "string", "description": "City name"},
    },
    "required": []string{"city"},
}
weather := ds4.NewFunction("get_weather", "Get current weather for a city", schema)

chat := client.Chat().
    Tool(weather).
    User("What's the weather like in Tokyo?")
resp, err := chat.Do()
if err != nil {
    return err
}

if resp.FinishReason() == ds4.FinishReasonToolCalls {
    chat.AppendResponse(resp)
    for _, call := range resp.ToolCalls() {
        chat.ToolResult(call.ID, `{"temperature":24}`)
    }
    resp, err = chat.Do()
}
```

Use `NewStrictFunction` to enforce a tool's JSON Schema. Strict tools use the
beta chat endpoint on DeepSeek.

### Prefix completion

Prefix completion continues from an assistant prefix and uses the beta chat
endpoint on DeepSeek.

```go
resp, err := client.Chat().
    User("Write a Go hello-world function").
    Prefix("func helloWorld() {\n").
    Do()
if err != nil {
    return err
}
fmt.Println(resp.Content())
```

To request more of a completed assistant response, append it and call
`Continue`. Its argument is added to the current `MaxTokens` value.

```go
chat := client.Chat().MaxTokens(128).User("Tell a story")
resp, err := chat.Do()
if err != nil {
    return err
}
more, err := chat.AppendResponse(resp).Continue(128).Do()
```

### Streaming

```go
err := client.Chat().
    User("Tell me a short story.").
    Stream(func(chunk ds4.ChatStreamChunk) error {
        for _, choice := range chunk.Choices {
            fmt.Print(choice.Delta.Content)
        }
        return nil
    })
```

## Fill-In-the-Middle

FIM is a beta completion API for generating text between a prefix and suffix.

```go
resp, err := client.FIM().
    Prompt("func greet(name string) string {\n    return ").
    Suffix("\n}").
    MaxTokens(64).
    Do()
if err != nil {
    return err
}
fmt.Println(resp.Text())
```

Stream FIM output with the same callback pattern:

```go
err := client.FIM().
    Prompt("func add(a, b int) int {\n    return ").
    Suffix("\n}").
    Stream(func(chunk ds4.FIMStreamChunk) error {
        for _, choice := range chunk.Choices {
            fmt.Print(choice.Text)
        }
        return nil
    })
```

## Local model routing

The same client can route calls to DeepSeek and a local llama.cpp
`llama-server` or other OpenAI-compatible deployment. On first use, ds4 loads
each configured server's model list and keeps the routing map in memory. A
local model wins when both servers advertise the same ID; unknown models fail
before a completion request is sent.

```go
client := ds4.New(apiKey).
    WithLocalKey("").
    WithLocalURL("http://localhost:8080/v1").
    WithLocalFIM(true)

resp, err := client.Chat().
    Model("noip-2000:2b").
    User("Explain goroutines.").
    Do()
```

`Ping()` eagerly refreshes both model lists. Discovery failures are isolated:
models from any server that responded successfully remain usable.

```go
if err := client.Ping(); err != nil {
    log.Printf("model discovery is incomplete: %v", err)
}
```

Local servers retain chat, multi-round conversations, streaming, JSON mode,
logprobs, tools, retries, and error handling. DeepSeek-private request fields
(`thinking`, `reasoning_effort`, and `user_id`) are omitted. Strict tools use
the regular local chat endpoint, while `Prefix` and `Continue` become ordinary
assistant messages.

For local models that emit reasoning inline, ds4 extracts common marker forms
and exposes them through `ReasoningContent()` and streaming reasoning deltas.
Adapt the marker set when a model uses a different template:

```go
ds4.DefaultReasoningMarkers = []ds4.ReasoningMarker{
    {Open: "<reasoning>", Close: "</reasoning>"},
}
reasoning, answer := ds4.SplitReasoning(raw)
```

Local FIM is sent to `/completions` with `input_prefix` and `input_suffix`.
Local FIM responses do not populate usage cache fields.

## Context, errors, and retries

The default client has no overall timeout so a long stream is not interrupted.
Use the `WithContext` variants to bound individual operations:

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

resp, err := client.Chat().User("Hello").DoWithContext(ctx)
```

Network and API failures are standard Go errors. API responses can be
inspected as `*ds4.APIError`:

```go
if apiErr, ok := err.(*ds4.APIError); ok {
    fmt.Println(apiErr.StatusCode, apiErr.Message)
}
```

Retries use exponential backoff with jitter and honor `Retry-After` for 429,
500, and 503 responses.

## Live deployment checker

`ds4check` makes billable requests to verify a real DeepSeek or
OpenAI-compatible deployment. It checks `models`, `chat`, `json`, `stream`,
and `tools` by default; add `thinking`, `prefix`, `fim`, or select `all`.
It prints `PASS` or `FAIL` for each selected check and exits non-zero when any
check fails.

```sh
go run ./cmd/ds4check \
    -api-key "$DEEPSEEK_API_KEY" \
    -model deepseek-flash \
    -tests all
```

## Tests

The integration suite covers both official V4 models, thinking, JSON,
logprobs, streaming, tools, prefix completion, and FIM. It skips when
`DEEPSEEK_API_KEY` is unavailable.

```sh
DEEPSEEK_API_KEY=your_key go test -run '^TestIntegration_AllFeatures$' -v
```
