package llama

import (
	"encoding/json"
	"os"
	"runtime"
	"strings"
	"testing"
)

func TestGraniteChatTemplateAndPredict(t *testing.T) {
	modelPath := os.Getenv("TEST_GRANITE_MODEL")
	if modelPath == "" {
		t.Skip("TEST_GRANITE_MODEL is not set")
	}

	model, err := New(modelPath, SetContext(512), SetNBatch(128), SetGPULayers(999))
	if err != nil {
		t.Fatalf("load granite model: %v", err)
	}
	defer model.Free()

	prompt, err := model.ApplyChatTemplate([]ChatMessage{
		{Role: "system", Content: "You are a terse assistant. Return valid JSON only, with no markdown."},
		{Role: "user", Content: "Create a JSON object with exactly these keys: model, formatting_ok, answer. Use model=granite, formatting_ok=true, answer=Paris."},
	}, true)
	if err != nil {
		t.Fatalf("apply chat template: %v", err)
	}
	if !strings.Contains(prompt, "<|start_of_role|>system<|end_of_role|>") ||
		!strings.Contains(prompt, "<|start_of_role|>assistant<|end_of_role|>") {
		t.Fatalf("unexpected granite prompt template: %q", prompt)
	}

	out, err := model.Predict(prompt,
		SetTokens(64),
		SetThreads(runtime.NumCPU()),
		SetTemperature(0),
		SetTopK(0),
		SetTopP(1),
		SetSeed(42),
		SetStopWords("<|end_of_text|>"),
	)
	if err != nil {
		t.Fatalf("predict: %v", err)
	}

	var got struct {
		Model        string `json:"model"`
		FormattingOK bool   `json:"formatting_ok"`
		Answer       string `json:"answer"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(out)), &got); err != nil {
		t.Fatalf("response is not valid JSON: %q: %v", out, err)
	}
	if got.Model != "granite" || !got.FormattingOK || got.Answer != "Paris" {
		t.Fatalf("unexpected response: %+v from %q", got, out)
	}
}
