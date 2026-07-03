package llama

import "testing"

func TestCleanPredictionResultTrimsStopSuffixOnly(t *testing.T) {
	got := cleanPredictionResult(" result llama", []string{"llama"})
	if got != " result " {
		t.Fatalf("unexpected cleaned result: %q", got)
	}

	got = cleanPredictionResult(" result mall", []string{"llama"})
	if got != " result mall" {
		t.Fatalf("stop prompt should only trim a full suffix, got %q", got)
	}
}

func TestFormatLogitBiasCombinesRawAndStructured(t *testing.T) {
	got := formatLogitBias(PredictOptions{
		LogitBias: "1:-2",
		LogitBiases: []LogitBias{
			{Token: 42, Bias: -100},
			{Token: 100, Bias: 1.5},
		},
	})
	want := "1:-2,42:-100,100:1.5"
	if got != want {
		t.Fatalf("unexpected logit bias spec: got %q want %q", got, want)
	}
}

func TestBuiltinChatTemplates(t *testing.T) {
	templates, err := BuiltinChatTemplates()
	if err != nil {
		t.Fatalf("BuiltinChatTemplates failed: %v", err)
	}
	if len(templates) == 0 {
		t.Fatalf("expected at least one built-in chat template")
	}
}
