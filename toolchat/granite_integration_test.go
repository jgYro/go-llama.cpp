package toolchat

import (
	"context"
	"encoding/json"
	"os"
	"runtime"
	"strings"
	"testing"

	llama "github.com/go-skynet/go-llama.cpp"
)

func TestGraniteToolRunner(t *testing.T) {
	modelPath := os.Getenv("TEST_GRANITE_MODEL")
	if modelPath == "" {
		t.Skip("TEST_GRANITE_MODEL is not set")
	}

	model, err := llama.New(modelPath, llama.SetContext(1024), llama.SetNBatch(128), llama.SetGPULayers(999))
	if err != nil {
		t.Fatalf("load granite model: %v", err)
	}
	defer model.Free()

	runner := Runner{
		Model:        model,
		MaxTurns:     3,
		SystemPrompt: "You are a terse assistant. Follow the JSON protocol exactly and do not use markdown.",
		Tools: []Tool{{
			Name:        "get_weather",
			Description: "Return deterministic weather for a city.",
			Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string"}},"required":["city"]}`),
			Call: func(ctx context.Context, args json.RawMessage) (any, error) {
				var got struct {
					City string `json:"city"`
					Unit string `json:"unit"`
				}
				if err := json.Unmarshal(args, &got); err != nil {
					return nil, err
				}
				return map[string]any{
					"city":          got.City,
					"unit":          got.Unit,
					"temperature_f": 72,
					"condition":     "clear",
				}, nil
			},
		}},
		PredictOptions: []llama.PredictOption{
			llama.SetTokens(160),
			llama.SetThreads(runtime.NumCPU()),
			llama.SetTemperature(0),
			llama.SetTopK(0),
			llama.SetTopP(1),
			llama.SetSeed(42),
			llama.SetStopWords("<|end_of_text|>"),
		},
	}

	resp, err := runner.Generate(context.Background(), []llama.ChatMessage{
		{Role: "user", Content: `Use get_weather for city "Boston" with unit "fahrenheit", then answer using the tool result.`},
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v\nraw outputs: %#v", err, resp.RawOutputs)
	}
	if len(resp.ToolCalls) == 0 {
		t.Fatalf("Granite did not request a tool call; response: %+v", resp)
	}
	if resp.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("unexpected tool call: %+v", resp.ToolCalls[0])
	}
	if resp.Content == "" || !strings.Contains(resp.Content, "72") {
		t.Fatalf("unexpected final content: %q", resp.Content)
	}
}
