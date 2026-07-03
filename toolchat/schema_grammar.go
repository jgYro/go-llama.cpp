package toolchat

import (
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

// jsonPrimitiveRules are the shared GBNF rules for JSON documents. Every
// grammar produced by this package appends them exactly once.
const jsonPrimitiveRules = `value  ::= object | array | string | number | boolean | null
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
integer ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ws
boolean ::= ("true" | "false") ws
null ::= "null" ws
ws ::= | " " | "\n" [ \t]{0,20}
`

// grammarBuiltinNames may not be redefined by schema-derived rules.
var grammarBuiltinNames = map[string]bool{
	"root": true, "final": true, "tool-call": true, "tool-call-item": true,
	"value": true, "object": true, "array": true, "string": true,
	"number": true, "integer": true, "boolean": true, "null": true, "ws": true,
}

// SchemaGrammar converts a JSON Schema document into a standalone GBNF
// grammar whose root rule matches documents shaped by the schema.
//
// Supported subset: "type" (including lists of types), object "properties"
// with "required" and "additionalProperties": false, arrays with "items" and
// "minItems", "enum", "const", "anyOf"/"oneOf", single-schema "allOf", and
// local "$ref" into "#/$defs" or "#/definitions" (recursion included).
// Object properties are generated required-first in sorted order.
//
// Constraints the grammar cannot express (pattern, format, numeric bounds,
// minLength, multi-schema allOf targets, ...) are ignored, so the grammar
// accepts a superset of the schema; callers must still validate parsed
// values.
func SchemaGrammar(schema json.RawMessage) (string, error) {
	conv := newSchemaConverter("schema")
	rootRule, err := conv.compileSchema(schema)
	if err != nil {
		return "", err
	}

	var b strings.Builder
	b.WriteString("root ::= " + rootRule + "\n")
	b.WriteString(conv.rulesText())
	b.WriteString(jsonPrimitiveRules)
	return b.String(), nil
}

type schemaConverter struct {
	doc    any
	prefix string
	rules  map[string]string
	order  []string
	refs   map[string]string
}

func newSchemaConverter(prefix string) *schemaConverter {
	return &schemaConverter{
		prefix: prefix,
		rules:  make(map[string]string),
		refs:   make(map[string]string),
	}
}

// compileSchema converts a schema document and returns the name of the rule
// matching it. Generated rules are collected via rulesText.
func (c *schemaConverter) compileSchema(schema json.RawMessage) (string, error) {
	var doc any
	if err := json.Unmarshal(schema, &doc); err != nil {
		return "", fmt.Errorf("invalid JSON schema: %w", err)
	}
	c.doc = doc
	return c.compileNode(doc, c.prefix)
}

func (c *schemaConverter) rulesText() string {
	var b strings.Builder
	for _, name := range c.order {
		b.WriteString(name)
		b.WriteString(" ::= ")
		b.WriteString(c.rules[name])
		b.WriteByte('\n')
	}
	return b.String()
}

// reserve claims a fresh rule name derived from base; define fills its body.
func (c *schemaConverter) reserve(base string) string {
	name := base
	for i := 2; ; i++ {
		_, taken := c.rules[name]
		if !taken && !grammarBuiltinNames[name] {
			break
		}
		name = fmt.Sprintf("%s-%d", base, i)
	}
	c.rules[name] = ""
	c.order = append(c.order, name)
	return name
}

func (c *schemaConverter) define(base, body string) string {
	name := c.reserve(base)
	c.rules[name] = body
	return name
}

func (c *schemaConverter) compileNode(node any, base string) (string, error) {
	switch n := node.(type) {
	case bool:
		if n {
			return "value", nil
		}
		return "", errors.New(`schema "false" matches no value`)
	case map[string]any:
		return c.compileSchemaObject(n, base)
	default:
		return "", fmt.Errorf("unsupported schema of type %T", node)
	}
}

func (c *schemaConverter) compileSchemaObject(n map[string]any, base string) (string, error) {
	if ref, ok := n["$ref"].(string); ok {
		return c.compileRef(ref)
	}
	if enum, ok := n["enum"].([]any); ok {
		return c.compileLiterals(enum, base)
	}
	if constValue, ok := n["const"]; ok {
		return c.compileLiterals([]any{constValue}, base)
	}
	for _, key := range []string{"anyOf", "oneOf"} {
		if alts, ok := n[key].([]any); ok {
			return c.compileAlternatives(alts, base)
		}
	}
	if all, ok := n["allOf"].([]any); ok {
		if len(all) == 1 {
			return c.compileNode(all[0], base)
		}
		return "", errors.New("allOf with multiple schemas is not supported")
	}

	switch t := n["type"].(type) {
	case string:
		return c.compileTyped(t, n, base)
	case []any:
		names := make([]string, 0, len(t))
		for _, item := range t {
			s, ok := item.(string)
			if !ok {
				return "", errors.New(`schema "type" list must contain strings`)
			}
			name, err := c.compileTyped(s, n, base+"-"+sanitizeRuleName(s))
			if err != nil {
				return "", err
			}
			names = append(names, name)
		}
		if len(names) == 0 {
			return "", errors.New(`schema "type" list must not be empty`)
		}
		return c.define(base, strings.Join(names, " | ")), nil
	case nil:
		if _, ok := n["properties"]; ok {
			return c.compileObject(n, base)
		}
		if _, ok := n["items"]; ok {
			return c.compileArray(n, base)
		}
		return "value", nil
	default:
		return "", errors.New(`schema "type" must be a string or a list of strings`)
	}
}

func (c *schemaConverter) compileTyped(t string, n map[string]any, base string) (string, error) {
	switch t {
	case "object":
		return c.compileObject(n, base)
	case "array":
		return c.compileArray(n, base)
	case "string":
		return "string", nil
	case "number":
		return "number", nil
	case "integer":
		return "integer", nil
	case "boolean":
		return "boolean", nil
	case "null":
		return "null", nil
	default:
		return "", fmt.Errorf("unsupported schema type %q", t)
	}
}

func (c *schemaConverter) compileRef(ref string) (string, error) {
	if name, ok := c.refs[ref]; ok {
		return name, nil
	}

	root, ok := c.doc.(map[string]any)
	if !ok {
		return "", fmt.Errorf("cannot resolve $ref %q", ref)
	}
	var target any
	var defName string
	switch {
	case strings.HasPrefix(ref, "#/$defs/"):
		defName = ref[len("#/$defs/"):]
		section, _ := root["$defs"].(map[string]any)
		target = section[defName]
	case strings.HasPrefix(ref, "#/definitions/"):
		defName = ref[len("#/definitions/"):]
		section, _ := root["definitions"].(map[string]any)
		target = section[defName]
	default:
		return "", fmt.Errorf("unsupported $ref %q (only local #/$defs and #/definitions)", ref)
	}
	if target == nil {
		return "", fmt.Errorf("$ref %q does not resolve", ref)
	}

	// Register the rule name before compiling the body so recursive
	// definitions terminate instead of looping.
	name := c.reserve(c.prefix + "-def-" + sanitizeRuleName(defName))
	c.refs[ref] = name
	inner, err := c.compileNode(target, name+"-body")
	if err != nil {
		return "", err
	}
	c.rules[name] = inner
	return name, nil
}

func (c *schemaConverter) compileLiterals(values []any, base string) (string, error) {
	if len(values) == 0 {
		return "", errors.New("enum must not be empty")
	}
	parts := make([]string, 0, len(values))
	for _, v := range values {
		encoded, err := json.Marshal(v)
		if err != nil {
			return "", fmt.Errorf("enum value is not encodable: %w", err)
		}
		parts = append(parts, strconv.Quote(string(encoded))+" ws")
	}
	return c.define(base, strings.Join(parts, " | ")), nil
}

func (c *schemaConverter) compileAlternatives(alts []any, base string) (string, error) {
	if len(alts) == 0 {
		return "", errors.New("anyOf/oneOf must not be empty")
	}
	names := make([]string, 0, len(alts))
	for i, alt := range alts {
		name, err := c.compileNode(alt, fmt.Sprintf("%s-alt-%d", base, i))
		if err != nil {
			return "", err
		}
		names = append(names, name)
	}
	return c.define(base, strings.Join(names, " | ")), nil
}

func (c *schemaConverter) compileArray(n map[string]any, base string) (string, error) {
	items, ok := n["items"]
	if !ok || items == nil {
		return "array", nil
	}
	itemName, err := c.compileNode(items, base+"-item")
	if err != nil {
		return "", err
	}

	inner := itemName + ` ("," ws ` + itemName + `)*`
	minItems, _ := n["minItems"].(float64)
	if minItems >= 1 {
		return c.define(base, `"[" ws `+inner+` "]" ws`), nil
	}
	return c.define(base, `"[" ws (`+inner+`)? "]" ws`), nil
}

func (c *schemaConverter) compileObject(n map[string]any, base string) (string, error) {
	props, _ := n["properties"].(map[string]any)
	if len(props) == 0 {
		if ap, ok := n["additionalProperties"].(bool); ok && !ap {
			return c.define(base, `"{" ws "}" ws`), nil
		}
		return "object", nil
	}
	// Extra keys alongside declared properties cannot be constrained
	// per-property, so an explicit opt-in to extras keeps the generic rule.
	if ap, ok := n["additionalProperties"]; ok && ap != false {
		return "object", nil
	}

	required := map[string]bool{}
	if reqList, ok := n["required"].([]any); ok {
		for _, r := range reqList {
			if s, ok := r.(string); ok {
				required[s] = true
			}
		}
	}

	propNames := make([]string, 0, len(props))
	for name := range props {
		propNames = append(propNames, name)
	}
	sort.Strings(propNames)

	var requiredKVs, optionalKVs []string
	for _, prop := range propNames {
		sub, err := c.compileNode(props[prop], base+"-"+sanitizeRuleName(prop))
		if err != nil {
			return "", err
		}
		kv := grammarJSONStringLiteral(prop) + ` ws ":" ws ` + sub
		if required[prop] {
			requiredKVs = append(requiredKVs, kv)
		} else {
			optionalKVs = append(optionalKVs, kv)
		}
	}

	body := `"{" ws `
	if len(requiredKVs) > 0 {
		body += strings.Join(requiredKVs, ` "," ws `)
		for _, kv := range optionalKVs {
			body += ` ("," ws ` + kv + `)?`
		}
	} else {
		// All properties optional: the first present one has no leading comma.
		alts := make([]string, len(optionalKVs))
		for i := range optionalKVs {
			alt := optionalKVs[i]
			for _, kv := range optionalKVs[i+1:] {
				alt += ` ("," ws ` + kv + `)?`
			}
			alts[i] = alt
		}
		body += `(` + strings.Join(alts, " | ") + `)?`
	}
	body += ` "}" ws`
	return c.define(base, body), nil
}

// sanitizeRuleName maps arbitrary text onto the GBNF rule-name alphabet
// (letters, digits, dashes).
func sanitizeRuleName(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "x"
	}
	return out
}
