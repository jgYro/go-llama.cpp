package toolchat

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestSchemaGrammarObjectRequiredAndOptional(t *testing.T) {
	grammar, err := SchemaGrammar(json.RawMessage(`{
		"type": "object",
		"properties": {
			"city": {"type": "string"},
			"unit": {"enum": ["celsius", "fahrenheit"]}
		},
		"required": ["city"]
	}`))
	if err != nil {
		t.Fatalf("SchemaGrammar returned error: %v", err)
	}

	for _, want := range []string{
		`"\"city\"" ws ":" ws string`,
		`("," ws "\"unit\"" ws ":" ws `,
		`"\"celsius\"" ws | "\"fahrenheit\"" ws`,
		"root ::= schema",
	} {
		if !strings.Contains(grammar, want) {
			t.Fatalf("grammar missing %q:\n%s", want, grammar)
		}
	}
}

func TestSchemaGrammarAllOptionalProperties(t *testing.T) {
	grammar, err := SchemaGrammar(json.RawMessage(`{
		"type": "object",
		"properties": {"a": {"type": "integer"}, "b": {"type": "boolean"}}
	}`))
	if err != nil {
		t.Fatalf("SchemaGrammar returned error: %v", err)
	}
	// Both single-property and both-property forms must be reachable.
	for _, want := range []string{
		`"\"a\"" ws ":" ws integer ("," ws "\"b\"" ws ":" ws boolean)?`,
		` | "\"b\"" ws ":" ws boolean`,
	} {
		if !strings.Contains(grammar, want) {
			t.Fatalf("grammar missing %q:\n%s", want, grammar)
		}
	}
}

func TestSchemaGrammarArrayEnumAndRef(t *testing.T) {
	grammar, err := SchemaGrammar(json.RawMessage(`{
		"type": "object",
		"properties": {
			"tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
			"next": {"$ref": "#/$defs/node"}
		},
		"required": ["tags"],
		"$defs": {
			"node": {
				"type": "object",
				"properties": {"next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}},
				"required": ["next"]
			}
		}
	}`))
	if err != nil {
		t.Fatalf("SchemaGrammar returned error: %v", err)
	}
	for _, want := range []string{
		`"[" ws string ("," ws string)* "]" ws`, // minItems 1: at least one element
		"schema-def-node",                       // recursive $defs rule exists
	} {
		if !strings.Contains(grammar, want) {
			t.Fatalf("grammar missing %q:\n%s", want, grammar)
		}
	}
}

func TestSchemaGrammarRejectsUnsupported(t *testing.T) {
	if _, err := SchemaGrammar(json.RawMessage(`false`)); err == nil {
		t.Fatal("false schema should be rejected")
	}
	if _, err := SchemaGrammar(json.RawMessage(`{"allOf": [{"type":"string"}, {"type":"number"}]}`)); err == nil {
		t.Fatal("multi-schema allOf should be rejected")
	}
	if _, err := SchemaGrammar(json.RawMessage(`{"$ref": "https://example.com/schema"}`)); err == nil {
		t.Fatal("remote $ref should be rejected")
	}
}

func TestEnvelopeGrammarConstrainsToolArguments(t *testing.T) {
	grammar, err := EnvelopeGrammar([]Tool{
		{
			Name: "get_weather",
			Schema: json.RawMessage(`{
				"type": "object",
				"properties": {"city": {"type": "string"}},
				"required": ["city"]
			}`),
		},
		{Name: "noop"},
	}, ToolChoiceAuto)
	if err != nil {
		t.Fatalf("EnvelopeGrammar returned error: %v", err)
	}

	for _, want := range []string{
		"tool-call-item ::= tool-item-0 | tool-item-1",
		`"\"arguments\"" ws ":" ws tool-args-0 `,
		`tool-args-0 ::= "{" ws "\"city\"" ws ":" ws string "}" ws`,
		// The schemaless tool keeps the generic object rule.
		`"\"noop\"" ws "," ws "\"arguments\"" ws ":" ws object`,
	} {
		if !strings.Contains(grammar, want) {
			t.Fatalf("grammar missing %q:\n%s", want, grammar)
		}
	}
}

func TestEnvelopeGrammarFallsBackOnUnsupportedSchema(t *testing.T) {
	grammar, err := EnvelopeGrammar([]Tool{{
		Name:   "odd",
		Schema: json.RawMessage(`{"allOf": [{"type":"string"}, {"type":"number"}]}`),
	}}, ToolChoiceAuto)
	if err != nil {
		t.Fatalf("EnvelopeGrammar returned error: %v", err)
	}
	if !strings.Contains(grammar, `"\"arguments\"" ws ":" ws object`) {
		t.Fatalf("expected generic object fallback:\n%s", grammar)
	}
}
