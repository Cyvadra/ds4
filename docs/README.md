# DeepSeek V4 documentation

This directory contains a concise, repository-local reference for the public
DeepSeek V4 API. It summarizes the official DeepSeek documentation instead of
mirroring it verbatim.

The information was retrieved on **2026-08-03**. Model aliases, limits,
pricing, and beta behavior can change, so follow the linked official pages when
current production values matter.

## Contents

- [Models and limits](models.md)
- [API usage and feature behavior](api.md)

## Official sources

| Topic | Canonical source |
| --- | --- |
| Quick start and current model aliases | [Your First API Call](https://api-docs.deepseek.com/) |
| Models, limits, features, and pricing | [Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing) |
| Chat request and response schema | [Chat Completions API](https://api-docs.deepseek.com/api/create-chat-completion) |
| Thinking controls and conversation rules | [Thinking Mode](https://api-docs.deepseek.com/guides/thinking_mode) |
| Structured JSON output | [JSON Output](https://api-docs.deepseek.com/guides/json_mode) |
| Function calling and strict schemas | [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls) |
| Assistant-prefix continuation | [Chat Prefix Completion](https://api-docs.deepseek.com/guides/chat_prefix_completion) |
| Fill-in-the-middle completion | [FIM Completion](https://api-docs.deepseek.com/guides/fim_completion) |
| HTTP failures | [Error Codes](https://api-docs.deepseek.com/quick_start/error_codes) |
| Documentation changes | [Change Log](https://api-docs.deepseek.com/updates) |

DeepSeek documents both OpenAI-compatible and Anthropic-compatible interfaces.
This repository uses the OpenAI-compatible HTTP interface.