package ds4

import "testing"

// A non-zero http.Client.Timeout also bounds reading of a streamed body, which
// would truncate long generations mid-stream.
func TestDefaultClientHasNoOverallTimeout(t *testing.T) {
	if got := New("key").httpClient.Timeout; got != 0 {
		t.Fatalf("default client must not set an overall timeout, got %v", got)
	}
}
