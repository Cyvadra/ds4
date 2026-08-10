package ds4

import "strings"

// ReasoningMarker describes how a local reasoning model delimits its
// chain-of-thought. DeepSeek's official API returns reasoning in a separate
// reasoning_content field; llama.cpp usually emits it inline between marker
// tokens instead. SplitReasoning and the local streaming path use these
// markers to restore the reasoning/answer split on the client side.
type ReasoningMarker struct {
	// Open is the marker that starts the reasoning block.
	Open string
	// Close is the marker that ends the reasoning block.
	Close string
	// WordBoundary, when true, requires Open to begin at a word boundary: it
	// must be preceded by whitespace (or the start of the text) and the
	// character after it must not be a lowercase letter. It is used for
	// word-like markers such as "think" so that prose like "thinking" is not
	// mistaken for reasoning, while reasoning that immediately follows the
	// marker ("thinkLet me...") still is.
	WordBoundary bool
}

// DefaultReasoningMarkers covers the delimiters emitted by common DeepSeek
// GGUF conversions running under llama.cpp. Override the package variable to
// match a different model template.
var DefaultReasoningMarkers = []ReasoningMarker{
	{Open: "<reasoning>", Close: "</reasoning>"},
	{Open: "think", Close: "/think", WordBoundary: true},
}

func isSpaceByte(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

func isLowerByte(b byte) bool {
	return b >= 'a' && b <= 'z'
}

// matchOpen reports whether text begins (after leading whitespace) with one of
// the default open markers, returning the matched marker and the buffer offset
// just past it.
func matchOpen(text string) (ReasoningMarker, int, bool) {
	start := 0
	for start < len(text) && isSpaceByte(text[start]) {
		start++
	}
	rest := text[start:]
	for _, m := range DefaultReasoningMarkers {
		if !strings.HasPrefix(rest, m.Open) {
			continue
		}
		if m.WordBoundary {
			after := start + len(m.Open)
			if after < len(text) && isLowerByte(text[after]) {
				continue
			}
		}
		return m, start + len(m.Open), true
	}
	return ReasoningMarker{}, 0, false
}

// canRuleOutOpen reports whether no default open marker can still begin at the
// start of text: either the leading whitespace is implausibly long, or the
// first non-whitespace run is neither an open marker nor a prefix of one.
func canRuleOutOpen(text string) bool {
	trimmed := strings.TrimLeft(text, " \t\r\n")
	if len(text)-len(trimmed) > 16 {
		return true
	}
	if trimmed == "" {
		return false
	}
	for _, m := range DefaultReasoningMarkers {
		if strings.HasPrefix(m.Open, trimmed) {
			return false
		}
	}
	return true
}

// SplitReasoning separates text into (reasoning, answer) using
// DefaultReasoningMarkers. It only splits when an open marker begins the text
// (after leading whitespace) and a matching close marker follows; otherwise it
// returns the text unchanged as the answer. This mirrors how DeepSeek
// reasoning models emit output under llama.cpp, where the chain-of-thought is
// the first thing produced.
func SplitReasoning(text string) (reasoning, answer string) {
	m, openEnd, ok := matchOpen(text)
	if !ok {
		return "", text
	}
	rest := text[openEnd:]
	closeIdx := strings.Index(rest, m.Close)
	if closeIdx < 0 {
		return "", text
	}
	return strings.TrimSpace(rest[:closeIdx]), strings.TrimLeft(rest[closeIdx+len(m.Close):], " \t\r\n")
}

// closePrefixLen returns the length of the longest prefix of close present at
// text[p:].
func closePrefixLen(text string, p int, close string) int {
	n := len(text)
	k := 0
	for k < len(close) && p+k < n && text[p+k] == close[k] {
		k++
	}
	return k
}

// earliestClose returns the position of the first close marker in text[from:],
// or of a partial marker that could still grow into the full marker. partial
// reports the second case.
func earliestClose(text string, from int, close string) (pos int, partial bool) {
	for p := from; p < len(text); p++ {
		k := closePrefixLen(text, p, close)
		if k == len(close) {
			return p, false
		}
		if k > 0 && p+k == len(text) {
			return p, true
		}
	}
	return -1, false
}

// reasoningSplitter incrementally separates reasoning from the answer in a
// streamed local response, where llama.cpp sends both through the same content
// channel. It buffers text until markers can be classified unambiguously, so a
// marker split across chunks is never misclassified.
type reasoningSplitter struct {
	buf        strings.Builder // all raw content received
	mode       int             // 0 deciding, 1 answer-only, 2 in reasoning, 3 after reasoning
	marker     ReasoningMarker // the open marker matched in mode 2/3
	reasonAt   int             // buf offset where reasoning begins
	answerAt   int             // buf offset where the answer begins
	reasonedTo int             // buf offset already emitted as reasoning
	answeredTo int             // buf offset already emitted as answer
}

func newReasoningSplitter() *reasoningSplitter { return &reasoningSplitter{} }

// Write feeds the next content fragment through the splitter and returns the
// newly classified reasoning and answer text. Either may be empty.
func (s *reasoningSplitter) Write(p string) (reasoningDelta, contentDelta string) {
	s.buf.WriteString(p)
	text := s.buf.String()
	n := len(text)

	for {
		switch s.mode {
		case 0:
			if m, openEnd, ok := matchOpen(text); ok {
				s.mode = 2
				s.marker = m
				s.reasonAt = openEnd
				s.reasonedTo = openEnd
				continue
			}
			if canRuleOutOpen(text) {
				s.mode = 1
				s.answeredTo = 0
				continue
			}
			return "", "" // hold everything until the start is decided

		case 1, 3:
			// No reasoning block, or the block already ended: the rest is answer.
			contentDelta = text[s.answeredTo:]
			s.answeredTo = n
			return "", contentDelta

		case 2:
			closeAt, partial := earliestClose(text, s.reasonAt, s.marker.Close)
			emitTo := n
			if closeAt >= 0 {
				if !partial {
					// Full close marker: reasoning ends here.
					reasoningDelta = text[s.reasonedTo:closeAt]
					s.reasonedTo = closeAt
					s.mode = 3
					s.answerAt = closeAt + len(s.marker.Close)
					s.answeredTo = s.answerAt
					continue
				}
				// Only a prefix of the close marker is present; it may still
				// complete, so stop before it.
				emitTo = closeAt
			}
			reasoningDelta = text[s.reasonedTo:emitTo]
			s.reasonedTo = emitTo
			return reasoningDelta, ""
		}
	}
}

// Flush returns any reasoning/answer text still held back at the end of the
// stream.
func (s *reasoningSplitter) Flush() (reasoningDelta, contentDelta string) {
	text := s.buf.String()
	switch s.mode {
	case 2:
		// Stream ended mid-reasoning: emit everything remaining as reasoning.
		reasoningDelta = text[s.reasonedTo:]
		s.reasonedTo = len(text)
		return reasoningDelta, ""
	default:
		// mode 0: nothing was decided; treat the whole stream as the answer.
		return "", text[s.answeredTo:]
	}
}

// splitLocalReasoning restores the reasoning/answer split of a local chat
// response, where the server sends both in the content field.
func splitLocalReasoning(resp *ChatResponse) {
	for i := range resp.Choices {
		message := &resp.Choices[i].Message
		reasoning, answer := SplitReasoning(message.Content)
		if reasoning == "" {
			continue
		}
		message.ReasoningContent = reasoning
		message.Content = answer
	}
}

// applyReasoningSplit routes each choice's streamed content through a
// reasoning splitter and returns the (possibly grown) splitter slice.
func applyReasoningSplit(chunk *ChatStreamChunk, splitters []*reasoningSplitter) []*reasoningSplitter {
	for i := range chunk.Choices {
		choice := &chunk.Choices[i]
		for len(splitters) <= choice.Index {
			splitters = append(splitters, newReasoningSplitter())
		}
		rDelta, cDelta := splitters[choice.Index].Write(choice.Delta.Content)
		choice.Delta.ReasoningContent = rDelta
		choice.Delta.Content = cDelta
	}
	return splitters
}

// flushReasoningSplit delivers any text still held back by the splitters once
// the stream ends.
func flushReasoningSplit(splitters []*reasoningSplitter, handler func(ChatStreamChunk) error) error {
	for i, s := range splitters {
		rDelta, cDelta := s.Flush()
		if rDelta == "" && cDelta == "" {
			continue
		}
		chunk := ChatStreamChunk{
			Choices: []StreamChoice{{
				Index: i,
				Delta: Delta{ReasoningContent: rDelta, Content: cDelta},
			}},
		}
		if err := handler(chunk); err != nil {
			return err
		}
	}
	return nil
}
