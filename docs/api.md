# DeepSeek V4 API behavior

Snapshot date: **2026-08-03**.

## Transport and authentication

- Base URL: `https://api.deepseek.com`
- Chat endpoint: `POST /chat/completions`
- Model discovery: `GET /models`
- Authentication: `Authorization: Bearer <DEEPSEEK_API_KEY>`
- Request media type: `application/json`
- Streaming transport: server-sent events (SSE), terminated by
  `data: [DONE]`

## Model discovery and routing

`Client` may be configured with both the official API and a local
OpenAI-compatible endpoint using `WithLocalKey` and `WithLocalURL`. Before the
first completion request, the client loads `/models` from each configured
backend. It loads official models first and local models second, so a duplicate
model ID routes to the local endpoint.

The routing map is retained in memory for the client lifetime. `Ping()` forces
a refresh and reports discovery errors; without `Ping`, later requests do not
repeat model discovery. A failure from one backend does not remove models
successfully loaded from the other. A model absent from the resulting map is
rejected before a completion request is sent.

Each `/models` and completion request uses the key belonging to its selected
backend. An empty local key omits the `Authorization` header.

Minimal request:

```json
{
  "model": "deepseek-flash",
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
```

The documented V4 model values are `deepseek-flash`, `deepseek-v4-pro`, and
`deepseek-flash`. Important optional Chat fields include `thinking`,
`reasoning_effort`, `max_tokens`, `response_format`, `stop`, `stream`,
`stream_options`, `temperature`, `top_p`, `tools`, `tool_choice`, `logprobs`,
and `top_logprobs`.

## Image understanding

The vision model is `deepseek-flash`. Its Chat Completions image
input is an OpenAI-compatible `content` block array on a `user` message:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {
      "url": "https://example.com/image.png",
      "detail": "original"
    }}
  ]
}
```

`ChatBuilder.WithImageURL(url, detail...)` emits an `image_url` block for an
absolute HTTP(S) URL. `ChatBuilder.WithImageBase64(data, mediaType, detail...)`
encodes raw JPEG, PNG, GIF, or WebP bytes into a Base64 data URL and emits the
same block. Calls may be mixed and repeated; at build time ds4 attaches all
images to the final user message, irrespective of fluent-call order, and sets
the request model to `deepseek-flash`.

The optional detail value is `low`, `high`, `original`, or `auto`. `low`
downscales to 512x512 before inference; the other values preserve the original
image, with `auto` currently behaving like `original`.

Images are valid only in `user` messages. The supported formats are JPEG, PNG,
GIF, and WebP. DeepSeek limits inline and URL images to 32 MiB each, permits
600 images per request, limits the overall request body to 48 MiB, and permits
an 8192-character URL with a 60-second download time. A maximum dimension is
8192 pixels per side, reduced to 4096 when a request has 15 or more images.
Use the Files API outside this package for the 64 MiB uploaded-file path.

## Thinking responses

Enable or disable thinking with:

```json
{
  "thinking": {"type": "enabled"},
  "reasoning_effort": "high"
}
```

Thinking is enabled by default and the default effort is `high`. A non-streamed
assistant message returns reasoning in `reasoning_content` and the final answer
in `content`; streamed responses expose corresponding deltas.

For an ordinary multi-round conversation without tool calls, old
`reasoning_content` is optional and is ignored if sent. For a tool-calling
reasoning sequence, preserve the complete assistant message, including
`reasoning_content` and `tool_calls`, in every subsequent request. Omitting it
can produce HTTP 400.

## Tool calls

Tools use OpenAI-compatible function definitions. The model proposes calls but
does not execute application functions. The caller must:

1. Send the available tools with the conversation.
2. Append the full assistant message containing `tool_calls`.
3. Execute each selected function in the application.
4. Append a `tool` message with the matching `tool_call_id` and result.
5. Request the model's next response.

Strict function schemas are beta. Use the beta route and set `strict: true` on
every function. The server validates schemas. Supported schema types include
`object`, `string`, `number`, `integer`, `boolean`, `array`, `enum`, and
`anyOf`. Every object property must be required and `additionalProperties`
must be `false`.

## JSON output

Set `response_format` to `{"type":"json_object"}`. The prompt must contain
the word `json`; DeepSeek also recommends including an example of the desired
shape and allowing enough `max_tokens` to avoid truncation. The official guide
warns that JSON mode can occasionally return empty content, in which case the
prompt should be adjusted and the request retried as appropriate.

JSON mode guarantees syntactically valid JSON, not conformance to an arbitrary
application schema. Use strict tool calls when server-enforced function
arguments are required.

## Beta completions

Beta features use `https://api.deepseek.com/beta` as their base URL, equivalent
to prefixing normal paths with `/beta` when the client base remains
`https://api.deepseek.com`.

### Chat prefix completion

Use the normal Chat Completions schema. The final message must have role
`assistant` and `prefix: true`; its `content` is the text the model continues.

### Fill-in-the-middle

FIM accepts a required `prompt` and an optional `suffix`, filling content
between them. It is beta, supports a maximum of 4K generated tokens, and is
listed as non-thinking only for both V4 models.

## Parameter notes

- `temperature` ranges from 0 to 2 and defaults to 1.
- `top_p` ranges up to 1 and defaults to 1.
- DeepSeek recommends changing either `temperature` or `top_p`, not both.
- `top_logprobs` ranges from 0 to 20 and requires `logprobs: true`.
- `frequency_penalty` and `presence_penalty` are deprecated and have no effect.
- `user_id` permits ASCII letters, digits, `-`, and `_`, up to 512 characters;
  it must not contain private user information.

## HTTP errors

| Status | Meaning | Typical action |
| ---: | --- | --- |
| 400 | Invalid request format | Correct the body using the returned message |
| 401 | Authentication failed | Check the API key |
| 402 | Insufficient balance | Check or fund the account |
| 422 | Invalid parameters | Correct fields using the returned message |
| 429 | Rate limit reached | Pace or back off requests |
| 500 | Server error | Retry after a short delay |
| 503 | Service overloaded | Retry after a short delay |

Retries should use bounded exponential backoff with jitter and should not be
applied blindly to invalid requests or authentication failures.

## Sources

- [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion)
- [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)
- [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls)
- [JSON Output](https://api-docs.deepseek.com/guides/json_mode)
- [Chat Prefix Completion](https://api-docs.deepseek.com/guides/chat_prefix_completion)
- [FIM Completion](https://api-docs.deepseek.com/guides/fim_completion)
- [Vision](https://api-docs.deepseek.com/guides/vision)
- [Error Codes](https://api-docs.deepseek.com/quick_start/error_codes)