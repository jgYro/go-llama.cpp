package toolchat

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"

	llama "github.com/go-skynet/go-llama.cpp"
)

const defaultMaxTurns = 4

type ToolChoice string

const (
	ToolChoiceAuto     ToolChoice = "auto"
	ToolChoiceRequired ToolChoice = "required"
	ToolChoiceNone     ToolChoice = "none"
)

type Model interface {
	ApplyChatTemplate(messages []llama.ChatMessage, addGenerationPrompt bool) (string, error)
	Predict(text string, opts ...llama.PredictOption) (string, error)
}

type Tool struct {
	Name        string
	Description string
	Schema      json.RawMessage
	Call        func(context.Context, json.RawMessage) (any, error)
}

type ToolCall struct {
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type ToolResult struct {
	Call    ToolCall        `json:"call"`
	Content json.RawMessage `json:"content,omitempty"`
	Error   string          `json:"error,omitempty"`
}

type Response struct {
	Content     string
	ToolCalls   []ToolCall
	ToolResults []ToolResult
	Turns       int
	RawOutputs  []string
	Messages    []llama.ChatMessage
}

type Runner struct {
	Model           Model
	Tools           []Tool
	MaxTurns        int
	PredictOptions  []llama.PredictOption
	SystemPrompt    string
	ToolChoice      ToolChoice
	DisableGrammar  bool
	FailOnToolError bool
}

type runnerConfig struct {
	model           Model
	tools           []Tool
	toolsByName     map[string]Tool
	maxTurns        int
	predictOptions  []llama.PredictOption
	systemPrompt    string
	toolChoice      ToolChoice
	disableGrammar  bool
	failOnToolError bool
	grammar         string
}

func (r Runner) Generate(ctx context.Context, messages []llama.ChatMessage) (Response, error) {
	cfg, err := r.prepare()
	if err != nil {
		return Response{}, err
	}

	history := append([]llama.ChatMessage(nil), messages...)
	resp := Response{Messages: append([]llama.ChatMessage(nil), history...)}

	for turn := 0; turn < cfg.maxTurns; turn++ {
		if err := ctx.Err(); err != nil {
			return resp, err
		}

		promptMessages := cfg.promptMessages(history)
		prompt, err := cfg.model.ApplyChatTemplate(promptMessages, true)
		if err != nil {
			return resp, fmt.Errorf("apply chat template: %w", err)
		}

		opts := append([]llama.PredictOption(nil), cfg.predictOptions...)
		if !cfg.disableGrammar && cfg.grammar != "" {
			opts = append(opts, llama.WithGrammar(cfg.grammar))
		}

		raw, err := cfg.model.Predict(prompt, opts...)
		resp.RawOutputs = append(resp.RawOutputs, raw)
		if err != nil {
			return resp, fmt.Errorf("predict: %w", err)
		}
		resp.Turns = turn + 1

		env, err := parseEnvelope(raw)
		if err != nil {
			return resp, fmt.Errorf("parse model response: %w", err)
		}

		switch env.Type {
		case "final":
			resp.Content = env.Content
			resp.Messages = append([]llama.ChatMessage(nil), history...)
			resp.Messages = append(resp.Messages, llama.ChatMessage{Role: "assistant", Content: raw})
			return resp, nil
		case "tool_call":
			if cfg.toolChoice == ToolChoiceNone {
				return resp, errors.New("model returned a tool call when tool choice is none")
			}
			results, err := cfg.executeToolCalls(ctx, turn, env.ToolCalls)
			if err != nil {
				return resp, err
			}

			for _, result := range results {
				resp.ToolCalls = append(resp.ToolCalls, result.Call)
				resp.ToolResults = append(resp.ToolResults, result)
			}

			history = append(history, llama.ChatMessage{Role: "assistant", Content: raw})
			history = append(history, llama.ChatMessage{Role: "user", Content: formatToolResults(results)})
			resp.Messages = append([]llama.ChatMessage(nil), history...)
		default:
			return resp, fmt.Errorf("unsupported response type %q", env.Type)
		}
	}

	return resp, fmt.Errorf("tool loop reached max turns %d without a final response", cfg.maxTurns)
}

func (r Runner) prepare() (*runnerConfig, error) {
	if r.Model == nil {
		return nil, errors.New("toolchat runner requires a model")
	}

	choice := r.ToolChoice
	if choice == "" {
		choice = ToolChoiceAuto
	}
	if choice != ToolChoiceAuto && choice != ToolChoiceRequired && choice != ToolChoiceNone {
		return nil, fmt.Errorf("unsupported tool choice %q", choice)
	}

	maxTurns := r.MaxTurns
	if maxTurns <= 0 {
		maxTurns = defaultMaxTurns
	}

	tools, toolsByName, err := normalizeTools(r.Tools, choice, true)
	if err != nil {
		return nil, err
	}

	grammar := ""
	if !r.DisableGrammar {
		grammar, err = EnvelopeGrammar(tools, choice)
		if err != nil {
			return nil, err
		}
	}

	return &runnerConfig{
		model:           r.Model,
		tools:           tools,
		toolsByName:     toolsByName,
		maxTurns:        maxTurns,
		predictOptions:  append([]llama.PredictOption(nil), r.PredictOptions...),
		systemPrompt:    systemPrompt(r.SystemPrompt, tools, choice),
		toolChoice:      choice,
		disableGrammar:  r.DisableGrammar,
		failOnToolError: r.FailOnToolError,
		grammar:         grammar,
	}, nil
}

func normalizeTools(tools []Tool, choice ToolChoice, requireCall bool) ([]Tool, map[string]Tool, error) {
	if choice == ToolChoiceRequired && len(tools) == 0 {
		return nil, nil, errors.New("tool choice required needs at least one tool")
	}

	normalized := append([]Tool(nil), tools...)
	toolsByName := make(map[string]Tool, len(normalized))
	for i := range normalized {
		tool := normalized[i]
		if err := validateToolName(tool.Name); err != nil {
			return nil, nil, err
		}
		if _, ok := toolsByName[tool.Name]; ok {
			return nil, nil, fmt.Errorf("duplicate tool name %q", tool.Name)
		}
		if len(tool.Schema) != 0 && !json.Valid(tool.Schema) {
			return nil, nil, fmt.Errorf("tool %q has invalid JSON schema", tool.Name)
		}
		if requireCall && choice != ToolChoiceNone && tool.Call == nil {
			return nil, nil, fmt.Errorf("tool %q needs a call function", tool.Name)
		}
		if len(tool.Schema) == 0 {
			tool.Schema = json.RawMessage(`{"type":"object","additionalProperties":true}`)
			normalized[i] = tool
		}
		toolsByName[tool.Name] = tool
	}

	sort.SliceStable(normalized, func(i, j int) bool {
		return normalized[i].Name < normalized[j].Name
	})

	return normalized, toolsByName, nil
}

func validateToolName(name string) error {
	if name == "" {
		return errors.New("tool name cannot be empty")
	}
	for i, r := range name {
		valid := r == '_' || r == '-' || r >= '0' && r <= '9' || r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z'
		if !valid {
			return fmt.Errorf("tool name %q contains unsupported character %q", name, r)
		}
		if i == 0 && r >= '0' && r <= '9' {
			return fmt.Errorf("tool name %q cannot start with a digit", name)
		}
	}
	return nil
}

func (c *runnerConfig) promptMessages(history []llama.ChatMessage) []llama.ChatMessage {
	if len(history) > 0 && history[0].Role == "system" {
		out := append([]llama.ChatMessage(nil), history...)
		out[0].Content = c.systemPrompt + "\n\n" + out[0].Content
		return out
	}

	out := make([]llama.ChatMessage, 0, len(history)+1)
	out = append(out, llama.ChatMessage{Role: "system", Content: c.systemPrompt})
	out = append(out, history...)
	return out
}

func (c *runnerConfig) executeToolCalls(ctx context.Context, turn int, calls []toolCallEnvelope) ([]ToolResult, error) {
	if len(calls) == 0 {
		return nil, errors.New("tool_call response did not include tool_calls")
	}

	results := make([]ToolResult, 0, len(calls))
	for i, call := range calls {
		toolCall := ToolCall{
			ID:        call.ID,
			Name:      call.Name,
			Arguments: call.Arguments,
		}
		if toolCall.ID == "" {
			toolCall.ID = fmt.Sprintf("call_%d_%d", turn+1, i+1)
		}
		tool, ok := c.toolsByName[toolCall.Name]
		if !ok {
			return results, fmt.Errorf("model requested unknown tool %q", toolCall.Name)
		}
		if !json.Valid(toolCall.Arguments) {
			return results, fmt.Errorf("model generated invalid arguments for tool %q", toolCall.Name)
		}
		if err := ctx.Err(); err != nil {
			return results, err
		}

		value, callErr := tool.Call(ctx, toolCall.Arguments)
		result := ToolResult{Call: toolCall}
		if callErr != nil {
			if c.failOnToolError {
				return results, fmt.Errorf("tool %q failed: %w", toolCall.Name, callErr)
			}
			result.Error = callErr.Error()
			result.Content = mustMarshalJSON(map[string]string{"error": callErr.Error()})
			results = append(results, result)
			continue
		}

		content, err := marshalToolOutput(value)
		if err != nil {
			return results, fmt.Errorf("marshal result from tool %q: %w", toolCall.Name, err)
		}
		result.Content = content
		results = append(results, result)
	}
	return results, nil
}

func systemPrompt(base string, tools []Tool, choice ToolChoice) string {
	var b strings.Builder
	if strings.TrimSpace(base) != "" {
		b.WriteString(strings.TrimSpace(base))
		b.WriteString("\n\n")
	}

	b.WriteString("Use the local llama.cpp JSON protocol for every assistant response.\n")
	b.WriteString("Final answers must be exactly one JSON object: {\"type\":\"final\",\"content\":\"...\"}.\n")

	if choice == ToolChoiceNone || len(tools) == 0 {
		b.WriteString("Do not call tools.\n")
		return b.String()
	}

	b.WriteString("When a tool is needed, respond with exactly one JSON object: {\"type\":\"tool_call\",\"tool_calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}.\n")
	if choice == ToolChoiceRequired {
		b.WriteString("You must call one of the available tools before giving a final answer.\n")
	} else {
		b.WriteString("Call a tool only when the available tool result is needed to answer correctly.\n")
	}
	b.WriteString("After tool results are provided, use them to continue and return either another tool_call or a final answer.\n")
	b.WriteString("Available tools in OpenAI-compatible function shape:\n")
	b.WriteString(toolDescriptorsJSON(tools))
	return b.String()
}

func toolDescriptorsJSON(tools []Tool) string {
	type function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters"`
	}
	type descriptor struct {
		Type     string   `json:"type"`
		Function function `json:"function"`
	}

	descriptors := make([]descriptor, 0, len(tools))
	for _, tool := range tools {
		descriptors = append(descriptors, descriptor{
			Type: "function",
			Function: function{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Schema,
			},
		})
	}

	out, err := json.MarshalIndent(descriptors, "", "  ")
	if err != nil {
		return "[]"
	}
	return string(out)
}

func formatToolResults(results []ToolResult) string {
	type toolResultMessage struct {
		ID      string          `json:"id,omitempty"`
		Name    string          `json:"name"`
		Result  json.RawMessage `json:"result,omitempty"`
		Error   string          `json:"error,omitempty"`
		Errored bool            `json:"errored,omitempty"`
	}

	var b strings.Builder
	b.WriteString("Tool results are available as JSON lines:\n")
	for _, result := range results {
		message := toolResultMessage{
			ID:      result.Call.ID,
			Name:    result.Call.Name,
			Result:  result.Content,
			Error:   result.Error,
			Errored: result.Error != "",
		}
		line, err := json.Marshal(message)
		if err != nil {
			continue
		}
		b.Write(line)
		b.WriteByte('\n')
	}
	b.WriteString("Use the tool results above. Return the next JSON protocol object only.")
	return b.String()
}

func marshalToolOutput(value any) (json.RawMessage, error) {
	switch v := value.(type) {
	case nil:
		return json.RawMessage("null"), nil
	case json.RawMessage:
		if !json.Valid(v) {
			return nil, errors.New("json.RawMessage is invalid JSON")
		}
		return append(json.RawMessage(nil), v...), nil
	case []byte:
		if json.Valid(v) {
			return append(json.RawMessage(nil), v...), nil
		}
		out, err := json.Marshal(string(v))
		return json.RawMessage(out), err
	default:
		out, err := json.Marshal(v)
		return json.RawMessage(out), err
	}
}

func mustMarshalJSON(value any) json.RawMessage {
	out, err := json.Marshal(value)
	if err != nil {
		return json.RawMessage(`null`)
	}
	return out
}

type responseEnvelope struct {
	Type      string             `json:"type"`
	Content   string             `json:"content"`
	Tool      string             `json:"tool"`
	Name      string             `json:"name"`
	Arguments json.RawMessage    `json:"arguments"`
	ToolCalls []toolCallEnvelope `json:"tool_calls"`
}

type toolCallEnvelope struct {
	ID        string                `json:"id"`
	Type      string                `json:"type"`
	Tool      string                `json:"tool"`
	Name      string                `json:"name"`
	Arguments json.RawMessage       `json:"arguments"`
	Function  *functionCallEnvelope `json:"function"`
}

type functionCallEnvelope struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

func parseEnvelope(raw string) (responseEnvelope, error) {
	body, err := extractJSONObject(raw)
	if err != nil {
		return responseEnvelope{}, err
	}

	var env responseEnvelope
	if err := json.Unmarshal([]byte(body), &env); err != nil {
		return responseEnvelope{}, err
	}
	if env.Type == "" {
		switch {
		case env.Tool != "" || env.Name != "" || len(env.ToolCalls) > 0:
			env.Type = "tool_call"
		case env.Content != "":
			env.Type = "final"
		}
	}

	switch env.Type {
	case "final":
		return env, nil
	case "tool_call":
		calls, err := normalizeToolCalls(env)
		if err != nil {
			return responseEnvelope{}, err
		}
		env.ToolCalls = calls
		return env, nil
	default:
		return responseEnvelope{}, fmt.Errorf("response must have type %q or %q", "final", "tool_call")
	}
}

func normalizeToolCalls(env responseEnvelope) ([]toolCallEnvelope, error) {
	calls := append([]toolCallEnvelope(nil), env.ToolCalls...)
	if len(calls) == 0 && (env.Tool != "" || env.Name != "") {
		calls = append(calls, toolCallEnvelope{
			Tool:      env.Tool,
			Name:      env.Name,
			Arguments: env.Arguments,
		})
	}
	if len(calls) == 0 {
		return nil, errors.New("tool_call response needs at least one tool call")
	}

	for i := range calls {
		if calls[i].Function != nil {
			if calls[i].Name == "" {
				calls[i].Name = calls[i].Function.Name
			}
			if len(calls[i].Arguments) == 0 {
				calls[i].Arguments = calls[i].Function.Arguments
			}
		}
		if calls[i].Name == "" {
			calls[i].Name = calls[i].Tool
		}
		if calls[i].Name == "" {
			return nil, fmt.Errorf("tool call %d is missing a name", i)
		}
		args, err := normalizeArguments(calls[i].Arguments)
		if err != nil {
			return nil, fmt.Errorf("tool call %q arguments: %w", calls[i].Name, err)
		}
		calls[i].Arguments = args
	}

	return calls, nil
}

func normalizeArguments(raw json.RawMessage) (json.RawMessage, error) {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return json.RawMessage(`{}`), nil
	}
	if !json.Valid(raw) {
		return nil, errors.New("invalid JSON")
	}

	var asString string
	if err := json.Unmarshal(raw, &asString); err == nil {
		asString = strings.TrimSpace(asString)
		if asString == "" {
			return json.RawMessage(`{}`), nil
		}
		if !json.Valid([]byte(asString)) {
			return nil, errors.New("string value does not contain JSON")
		}
		return json.RawMessage(asString), nil
	}

	return append(json.RawMessage(nil), raw...), nil
}

func extractJSONObject(raw string) (string, error) {
	body := strings.TrimSpace(raw)
	body = stripMarkdownFence(body)
	start := strings.Index(body, "{")
	end := strings.LastIndex(body, "}")
	if start < 0 || end < start {
		return "", errors.New("response does not contain a JSON object")
	}
	return body[start : end+1], nil
}

func stripMarkdownFence(body string) string {
	if !strings.HasPrefix(body, "```") {
		return body
	}
	lines := strings.Split(body, "\n")
	if len(lines) < 2 {
		return body
	}
	lines = lines[1:]
	if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(lines[len(lines)-1]), "```") {
		lines = lines[:len(lines)-1]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func EnvelopeGrammar(tools []Tool, choice ToolChoice) (string, error) {
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

	root := "root ::= final"
	if choice == ToolChoiceRequired {
		root = "root ::= tool-call"
	} else if choice == ToolChoiceAuto && len(normalized) > 0 {
		root = "root ::= final | tool-call"
	}

	toolRules := ""
	if len(normalized) > 0 && choice != ToolChoiceNone {
		parts := make([]string, 0, len(normalized))
		for _, tool := range normalized {
			parts = append(parts, grammarJSONStringLiteral(tool.Name)+" ws")
		}
		toolRules = `
tool-call ::= "{" ws "\"type\"" ws ":" ws "\"tool_call\"" ws "," ws "\"tool_calls\"" ws ":" ws "[" ws tool-call-item ("," ws tool-call-item)* "]" ws "}" ws
tool-call-item ::= "{" ws "\"name\"" ws ":" ws tool-name "," ws "\"arguments\"" ws ":" ws object "}" ws
tool-name ::= ` + strings.Join(parts, " | ") + "\n"
	}

	return root + `
final ::= "{" ws "\"type\"" ws ":" ws "\"final\"" ws "," ws "\"content\"" ws ":" ws string "}" ws` + toolRules + `
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4})
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws
ws ::= | " " | "\n" [ \t]{0,20}
`, nil
}

func grammarJSONStringLiteral(value string) string {
	return strconv.Quote(strconv.Quote(value))
}
