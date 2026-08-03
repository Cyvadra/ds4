# DeepSeek V4 models and limits

Snapshot date: **2026-08-03**.

## Model aliases

The public API exposes two V4 aliases:

| API model ID | Version behind the alias | Intended tradeoff |
| --- | --- | --- |
| `deepseek-v4-flash` | `DeepSeek-V4-Flash-0731` | Higher throughput and lower cost |
| `deepseek-v4-pro` | `DeepSeek-V4-Pro` | Higher-capability V4 service tier |

The aliases are rolling service names. In particular, DeepSeek states that
`deepseek-v4-flash` now resolves to `DeepSeek-V4-Flash-0731` without requiring
callers to change the model ID.

Both models use the OpenAI-format base URL `https://api.deepseek.com` and
support thinking and non-thinking modes. Thinking is enabled by default.

## Capacity and feature matrix

| Capability | V4 Flash | V4 Pro |
| --- | --- | --- |
| Context length | 1M tokens | 1M tokens |
| Maximum output | 384K tokens | 384K tokens |
| JSON output | Yes | Yes |
| Tool calls | Yes | Yes |
| Responses API | Yes | Not yet listed as supported |
| Anthropic-compatible API | Yes | Yes |
| Chat prefix completion | Beta | Beta |
| FIM completion | Beta, non-thinking only | Beta, non-thinking only |

The input plus generated output must fit within the model context. A request's
`max_tokens` controls the generated portion and does not increase the context
window.

## Thinking effort

For Chat Completions, send `thinking.type` as `enabled` or `disabled`. When
thinking is enabled, `reasoning_effort` accepts `low`, `high`, or `max` and
defaults to `high`.

| Requested value | V4 Flash mapping | V4 Pro mapping on snapshot date |
| --- | --- | --- |
| `low` | `low` | `high` |
| `high` | `high` | `high` |
| `max` | `max` | `max` |
| Compatibility value `medium` | `high` | `high` |
| Compatibility value `xhigh` | `high` | `max` |

DeepSeek described V4 Pro's reduced effort mapping as temporary and expected
full support in early August 2026. Check the live Thinking Mode page before
depending on that rollout.

In thinking mode, `temperature`, `top_p`, `presence_penalty`, and
`frequency_penalty` have no effect. The API accepts them for compatibility
without applying them. The Chat API reference separately marks both penalty
parameters as deprecated for all requests.

## Pricing snapshot

Official prices were listed per one million tokens:

| Billing item | V4 Flash | V4 Pro |
| --- | ---: | ---: |
| Cached input | $0.0028 | $0.003625 |
| Uncached input | $0.14 | $0.435 |
| Output | $0.28 | $0.87 |

DeepSeek announced that peak/off-peak pricing would be introduced, with peak
prices at twice the regular rate during `09:00-12:00` and `14:00-18:00` Beijing
time (`UTC+8`). No effective date was shown in the retrieved page. Treat this
table as a dated snapshot and consult the live
[Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing) page for
billing decisions.

## Sources

- [Your First API Call](https://api-docs.deepseek.com/)
- [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion)
- [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode)