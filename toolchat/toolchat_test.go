package toolchat

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	llama "github.com/go-skynet/go-llama.cpp"
)

type fakeModel struct {
	outputs  []string
	prompts  []string
	messages [][]llama.ChatMessage
}

func (f *fakeModel) ApplyChatTemplate(messages []llama.ChatMessage, addGenerationPrompt bool) (string, error) {
	copied := append([]llama.ChatMessage(nil), messages...)
	f.messages = append(f.messages, copied)

	var prompt strings.Builder
	for _, message := range messages {
		prompt.WriteString(message.Role)
		prompt.WriteString(": ")
		prompt.WriteString(message.Content)
		prompt.WriteByte('\n')
	}
	if addGenerationPrompt {
		prompt.WriteString("assistant: ")
	}
	f.prompts = append(f.prompts, prompt.String())
	return prompt.String(), nil
}

func (f *fakeModel) Predict(text string, opts ...llama.PredictOption) (string, error) {
	if len(f.outputs) == 0 {
		return "", errors.New("no queued output")
	}
	out := f.outputs[0]
	f.outputs = f.outputs[1:]
	return out, nil
}

func TestRunnerExecutesToolThenReturnsFinal(t *testing.T) {
	model := &fakeModel{outputs: []string{
		`{"type":"tool_call","tool_calls":[{"name":"get_weather","arguments":{"city":"Boston","unit":"fahrenheit"}}]}`,
		`{"type":"final","content":"Boston is clear and 72 F."}`,
	}}

	runner := Runner{
		Model:    model,
		MaxTurns: 3,
		Tools: []Tool{{
			Name:        "get_weather",
			Description: "Get current weather.",
			Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"},"unit":{"type":"string"}},"required":["city"]}`),
			Call: func(ctx context.Context, args json.RawMessage) (any, error) {
				var got struct {
					City string `json:"city"`
					Unit string `json:"unit"`
				}
				if err := json.Unmarshal(args, &got); err != nil {
					return nil, err
				}
				if got.City != "Boston" || got.Unit != "fahrenheit" {
					t.Fatalf("unexpected tool args: %+v", got)
				}
				return map[string]any{"city": got.City, "temperature_f": 72, "condition": "clear"}, nil
			},
		}},
	}

	resp, err := runner.Generate(context.Background(), []llama.ChatMessage{
		{Role: "user", Content: "Use the weather tool for Boston."},
	})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	if resp.Content != "Boston is clear and 72 F." {
		t.Fatalf("unexpected final content: %q", resp.Content)
	}
	if resp.Turns != 2 {
		t.Fatalf("expected 2 turns, got %d", resp.Turns)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("unexpected tool calls: %+v", resp.ToolCalls)
	}
	if len(resp.ToolResults) != 1 || !strings.Contains(string(resp.ToolResults[0].Content), "temperature_f") {
		t.Fatalf("unexpected tool results: %+v", resp.ToolResults)
	}
	if len(model.messages) != 2 {
		t.Fatalf("expected two prompt applications, got %d", len(model.messages))
	}
	if !strings.Contains(model.messages[0][0].Content, "Available tools") {
		t.Fatalf("first prompt did not include tool descriptions: %q", model.messages[0][0].Content)
	}
	lastMessage := model.messages[1][len(model.messages[1])-1]
	if lastMessage.Role != "user" || !strings.Contains(lastMessage.Content, "temperature_f") {
		t.Fatalf("second prompt did not include tool result: %+v", lastMessage)
	}
}

func TestRunnerReturnsUnknownToolError(t *testing.T) {
	model := &fakeModel{outputs: []string{
		`{"type":"tool_call","tool_calls":[{"name":"missing_tool","arguments":{}}]}`,
	}}
	runner := Runner{
		Model: model,
		Tools: []Tool{{
			Name: "get_weather",
			Call: func(ctx context.Context, args json.RawMessage) (any, error) {
				return map[string]string{"ok": "true"}, nil
			},
		}},
	}

	_, err := runner.Generate(context.Background(), []llama.ChatMessage{{Role: "user", Content: "Call a tool."}})
	if err == nil || !strings.Contains(err.Error(), "unknown tool") {
		t.Fatalf("expected unknown tool error, got %v", err)
	}
}

func TestParseEnvelopeSupportsOpenAIStyleToolCall(t *testing.T) {
	env, err := parseEnvelope(`{
		"type": "tool_call",
		"tool_calls": [{
			"id": "call_1",
			"type": "function",
			"function": {
				"name": "get_weather",
				"arguments": "{\"city\":\"Boston\"}"
			}
		}]
	}`)
	if err != nil {
		t.Fatalf("parseEnvelope returned error: %v", err)
	}

	if len(env.ToolCalls) != 1 {
		t.Fatalf("expected one tool call, got %+v", env.ToolCalls)
	}
	call := env.ToolCalls[0]
	if call.ID != "call_1" || call.Name != "get_weather" || string(call.Arguments) != `{"city":"Boston"}` {
		t.Fatalf("unexpected tool call: %+v", call)
	}
}

func TestParseEnvelopeSupportsLegacyToolObject(t *testing.T) {
	env, err := parseEnvelope(`{"tool":"get_weather","arguments":{"city":"Boston"}}`)
	if err != nil {
		t.Fatalf("parseEnvelope returned error: %v", err)
	}
	if env.Type != "tool_call" || len(env.ToolCalls) != 1 || env.ToolCalls[0].Name != "get_weather" {
		t.Fatalf("unexpected envelope: %+v", env)
	}
}

func TestEnvelopeGrammar(t *testing.T) {
	grammar, err := EnvelopeGrammar([]Tool{{Name: "get_weather"}}, ToolChoiceAuto)
	if err != nil {
		t.Fatalf("EnvelopeGrammar returned error: %v", err)
	}
	for _, want := range []string{
		"root ::= final | tool-call",
		`"\"get_weather\"" ws`,
		`"\"type\""`,
		`"\"tool_calls\""`,
	} {
		if !strings.Contains(grammar, want) {
			t.Fatalf("grammar missing %q:\n%s", want, grammar)
		}
	}
}

func TestEnvelopeGrammarValidatesToolNames(t *testing.T) {
	_, err := EnvelopeGrammar([]Tool{{Name: "bad name"}}, ToolChoiceAuto)
	if err == nil || !strings.Contains(err.Error(), "unsupported character") {
		t.Fatalf("expected invalid name error, got %v", err)
	}
}

func TestRunnerHandlesToolErrorsAsResults(t *testing.T) {
	model := &fakeModel{outputs: []string{
		`{"type":"tool_call","tool_calls":[{"name":"unstable","arguments":{}}]}`,
		`{"type":"final","content":"The tool failed."}`,
	}}
	runner := Runner{
		Model: model,
		Tools: []Tool{{
			Name: "unstable",
			Call: func(ctx context.Context, args json.RawMessage) (any, error) {
				return nil, errors.New("temporary outage")
			},
		}},
	}

	resp, err := runner.Generate(context.Background(), []llama.ChatMessage{{Role: "user", Content: "Try the tool."}})
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if len(resp.ToolResults) != 1 || resp.ToolResults[0].Error != "temporary outage" {
		t.Fatalf("expected tool error result, got %+v", resp.ToolResults)
	}
	if !strings.Contains(model.messages[1][len(model.messages[1])-1].Content, "temporary outage") {
		t.Fatalf("tool error was not fed back into the prompt: %+v", model.messages[1])
	}
}
