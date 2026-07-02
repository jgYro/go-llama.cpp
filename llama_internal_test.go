package llama

import "testing"

func TestCleanPredictionResultTrimsStopSuffixOnly(t *testing.T) {
	got := cleanPredictionResult(" promptresult llama", "prompt", []string{"llama"})
	if got != "result " {
		t.Fatalf("unexpected cleaned result: %q", got)
	}

	got = cleanPredictionResult(" promptresult mall", "prompt", []string{"llama"})
	if got != "result mall" {
		t.Fatalf("stop prompt should only trim a full suffix, got %q", got)
	}
}
