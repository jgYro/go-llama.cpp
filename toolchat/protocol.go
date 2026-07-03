package toolchat

import "fmt"

// Envelope is a parsed model response in the JSON tool protocol.
type Envelope struct {
	Type      string     // "final" or "tool_call"
	Content   string     // final answer content when Type == "final"
	ToolCalls []ToolCall // requested calls when Type == "tool_call"
}

// ParseEnvelope parses a raw model response into an Envelope, tolerating
// markdown fences, surrounding prose, OpenAI-style function payloads, and
// legacy single-tool objects. It lets external drivers (for example a Genkit
// model implementation) run their own tool loop on the same protocol Runner
// uses.
func ParseEnvelope(raw string) (Envelope, error) {
	env, err := parseEnvelope(raw)
	if err != nil {
		return Envelope{}, err
	}

	out := Envelope{Type: env.Type, Content: env.Content}
	for i, call := range env.ToolCalls {
		id := call.ID
		if id == "" {
			id = fmt.Sprintf("call_%d", i+1)
		}
		out.ToolCalls = append(out.ToolCalls, ToolCall{
			ID:        id,
			Name:      call.Name,
			Arguments: call.Arguments,
		})
	}
	return out, nil
}

// SystemPrompt returns the JSON-protocol system prompt Runner uses,
// combining base instructions with descriptors for the given tools and
// tool choice.
func SystemPrompt(base string, tools []Tool, choice ToolChoice) (string, error) {
	if choice == "" {
		choice = ToolChoiceAuto
	}
	if choice != ToolChoiceAuto && choice != ToolChoiceRequired && choice != ToolChoiceNone {
		return "", fmt.Errorf("unsupported tool choice %q", choice)
	}

	normalized, _, err := normalizeTools(tools, choice, false)
	if err != nil {
		return "", err
	}
	return systemPrompt(base, normalized, choice), nil
}

// FormatToolResults renders executed tool results as the user-turn message
// the protocol expects to follow a tool_call response.
func FormatToolResults(results []ToolResult) string {
	return formatToolResults(results)
}
