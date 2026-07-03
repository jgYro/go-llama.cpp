# go-llama.cpp

Go bindings for the modern [llama.cpp](https://github.com/ggml-org/llama.cpp) C API, plus a
grammar-constrained tool-calling loop (`toolchat`). The bindings are high level: the inference
loop, sampling, prompt caching, and context management live in C++ (`binding.cpp`), and Go sees
a small, safe surface.

Works with **GGUF** models only.

## Build

The repository tracks llama.cpp as a git submodule:

```bash
git clone --recurse-submodules https://github.com/jgYro/go-llama.cpp
cd go-llama.cpp
make libbinding.a
```

`make libbinding.a` builds the submodule with CMake and archives it together with `binding.o`.
Go builds then need the include/library paths:

```bash
C_INCLUDE_PATH=$PWD LIBRARY_PATH=$PWD go build ./...
C_INCLUDE_PATH=$PWD LIBRARY_PATH=$PWD go test ./...
```

Acceleration variants:

```bash
BUILD_TYPE=metal    make libbinding.a   # Apple GPU (Metal + Accelerate are on by default on macOS)
BUILD_TYPE=cublas   make libbinding.a   # NVIDIA CUDA
BUILD_TYPE=hipblas  make libbinding.a   # AMD ROCm
BUILD_TYPE=openblas make libbinding.a   # CPU BLAS
```

For GPU builds pass the matching linker flags through `CGO_LDFLAGS`
(e.g. `CGO_LDFLAGS="-lcublas -lcudart -L/usr/local/cuda/lib64/"`) and offload layers with
`llama.SetGPULayers(n)`.

## Quick start

```go
package main

import (
	"fmt"

	llama "github.com/go-skynet/go-llama.cpp"
)

func main() {
	model, err := llama.New(
		"model.gguf",
		llama.SetContext(4096),
		llama.SetGPULayers(32),
	)
	if err != nil {
		panic(err) // errors carry the cause, e.g. `unable to load model model.gguf`
	}
	defer model.Free()

	out, err := model.Predict(
		"The three laws of robotics are",
		llama.SetTokens(256),
		llama.SetTemperature(0.7),
		llama.SetStopWords("\n\n"),
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(out)
}
```

`SetTokens(0)` generates until the model emits end-of-generation (bounded by the context
window; when the context fills, the binding context-shifts and keeps going). The default
budget is 128 tokens.

## API tour

### Model options (`llama.New`)

```go
model, err := llama.New("model.gguf",
	llama.SetContext(8192),            // KV cache size in tokens
	llama.SetGPULayers(99),            // layers to offload; 0 = CPU only
	llama.SetNBatch(512),              // logical batch size
	llama.SetNUBatch(512),             // physical micro-batch
	llama.SetMMap(true),               // mmap the weights (default true)
	llama.EnableMLock,                 // pin weights in RAM
	llama.EnableEmbeddings,            // required before Embeddings()
	llama.SetMainGPU("0"),             // device index for the main GPU
	llama.SetTensorSplit("0.7,0.3"),   // multi-GPU weight split
	llama.SetFlashAttention(llama.FlashAttentionEnabled),
	llama.SetRopeScaling(llama.RopeScalingYarn),
	llama.SetPoolingType(llama.PoolingMean),
	llama.SetLoraAdapter("adapter.gguf"), // one LoRA adapter, applied at load
)
```

### Text generation

```go
out, err := model.Predict(prompt,
	llama.SetTokens(0),               // until end-of-generation
	llama.SetSeed(42),
	llama.SetThreads(8),
	llama.SetTemperature(0.8),
	llama.SetTopK(40),
	llama.SetTopP(0.95),
	llama.SetMinP(0.05),
	llama.SetPenalty(1.1),            // repetition penalty
	llama.SetFrequencyPenalty(0.2),
	llama.SetStopWords("User:", "###"), // matched anywhere, trimmed from output
	llama.IgnoreEOS,                  // never stop on EOS
)
```

Also available: typical-p, top-n-sigma, XTC (`SetXTC`), dynamic temperature
(`SetDynamicTemperature`), adaptive-p (`SetAdaptiveP`), Mirostat v1/v2 (`SetMirostat`,
`SetMirostatTAU`, `SetMirostatETA`), DRY (`SetDRY`, `SetDRYSequenceBreakers`), and token-level
logit bias (`SetLogitBiasToken(token, bias)` or `SetLogitBias("15043:-100")`).

### Streaming tokens

```go
_, err := model.Predict(prompt,
	llama.SetTokenCallback(func(token string) bool {
		fmt.Print(token)
		return true // return false to stop generation early
	}),
	llama.SetTokens(0),
)
```

### Chat templates

Formats messages with the model's own chat template (from GGUF metadata):

```go
prompt, err := model.ApplyChatTemplate([]llama.ChatMessage{
	{Role: "system", Content: "You are a terse assistant."},
	{Role: "user", Content: "Why is the sky blue?"},
}, true) // true = append the assistant generation prompt
out, err := model.Predict(prompt, llama.SetTokens(0))
```

`llama.BuiltinChatTemplates()` lists the template names llama.cpp ships as fallbacks.

### Grammar-constrained generation

Any prediction can be constrained by a GBNF grammar; a grammar that fails to parse returns an
explicit error instead of silently generating free-form text:

```go
out, err := model.Predict("Is the sky blue? ",
	llama.WithGrammar(`root ::= "yes" | "no"`))
```

To constrain output to a JSON Schema, convert it first:

```go
grammar, err := toolchat.SchemaGrammar(json.RawMessage(`{
	"type": "object",
	"properties": {
		"city": {"type": "string"},
		"unit": {"enum": ["celsius", "fahrenheit"]}
	},
	"required": ["city"]
}`))
out, err := model.Predict(prompt, llama.WithGrammar(grammar), llama.SetTokens(0))
```

Supported schema subset: `type` (including lists), object `properties` with
`required`/`additionalProperties: false`, arrays with `items`/`minItems`, `enum`, `const`,
`anyOf`/`oneOf`, and recursive local `$defs`/`definitions` refs. Constraints a grammar cannot
express (`pattern`, numeric bounds, ...) are ignored — validate parsed values in Go.

### Tool calling (`toolchat`)

A local tool loop: the model answers with JSON envelopes, tool calls are dispatched to Go
callbacks, and results are fed back until the model produces a final answer. Each tool's
`arguments` object is grammar-constrained by its schema.

```go
runner := toolchat.Runner{
	Model:      model,
	MaxTurns:   4,                          // default 4
	ToolChoice: toolchat.ToolChoiceAuto,    // or ToolChoiceRequired / ToolChoiceNone
	Tools: []toolchat.Tool{{
		Name:        "get_weather",
		Description: "Return current weather for a city.",
		Schema: json.RawMessage(`{
			"type": "object",
			"properties": {"city": {"type": "string"}},
			"required": ["city"]
		}`),
		Call: func(ctx context.Context, args json.RawMessage) (any, error) {
			var in struct{ City string `json:"city"` }
			if err := json.Unmarshal(args, &in); err != nil {
				return nil, err
			}
			return map[string]any{"city": in.City, "temp_c": 21, "condition": "clear"}, nil
		},
	}},
}

resp, err := runner.Generate(ctx, []llama.ChatMessage{
	{Role: "user", Content: "What's the weather in Boston?"},
})
fmt.Println(resp.Content)     // final answer
fmt.Println(resp.ToolCalls)   // every call the model made
fmt.Println(resp.ToolResults) // and what the tools returned
```

With `ToolChoiceRequired` the first turn must call a tool; later turns may answer. Tool errors
are fed back to the model as results unless `FailOnToolError: true`. Set `DisableGrammar: true`
to rely on prompting alone.

### Multimodal (images and audio)

Load a multimodal projector (mmproj GGUF) alongside the text model, then attach
media files to predictions. Wrap media prompts in the model's chat template with
the media marker inside the user message:

```go
model, err := llama.New("granite-vision-4.1-4b-Q4_K_M.gguf",
	llama.SetContext(8192),           // vision models need room for image tokens
	llama.SetNBatch(2048),
	llama.SetGPULayers(99),
	llama.SetMMProj("mmproj-model-f16.gguf"),
)

prompt, err := model.ApplyChatTemplate([]llama.ChatMessage{
	{Role: "user", Content: llama.MediaMarker() + "\nWhat is shown in this image?"},
}, true)

out, err := model.Predict(prompt, llama.WithMedia("photo.jpg"), llama.SetTokens(0))
```

One `llama.MediaMarker()` (`<__media__>`) positions each media file in the
prompt; when the prompt contains no marker, one is prepended per file. Images
(jpg/png/bmp/gif) and audio (wav/mp3/flac, for audio-capable models) are decoded
automatically; grammar constraints, streaming callbacks, and stop words all
apply to multimodal predictions too. `model.SupportsVision()` /
`model.SupportsAudio()` report projector capabilities. A media prediction
without a loaded projector returns an explicit error.

Ollama blobs work directly as model and projector paths — resolve the `model`
and `projector` layer digests from the manifest under
`~/.ollama/models/manifests/`.

### Embeddings

```go
model, err := llama.New("embedding-model.gguf",
	llama.EnableEmbeddings,
	llama.SetPoolingType(llama.PoolingMean),
)
vec, err := model.Embeddings("The quick brown fox")   // []float32
vec, err = model.TokenEmbeddings([]int{1, 2, 3})      // from raw token ids
```

### Tokenization

```go
count, tokens, err := model.TokenizeString("Hello, world") // int32 count + []int32 ids
text, err := model.Detokenize(tokens, true, false)         // ids back to text
```

### Prompt caching

Session files skip re-decoding shared prompt prefixes across calls — the main latency win for
multi-turn chat with a growing transcript:

```go
out, err := model.Predict(prompt,
	llama.SetPathPromptCache("session.bin"),
	llama.EnablePromptCacheAll, // also cache generated tokens
)
```

On the next call the longest matching token prefix is reused and only the new suffix is
decoded. `EnablePromptCacheRO` reads an existing cache without updating it.

### State save/load

Snapshots the full context (KV cache + logits), separate from prompt caching:

```go
err := model.SaveState("state.bin")
err = model.LoadState("state.bin")
```

### Model introspection

```go
info, err := model.ModelInfo()
fmt.Println(info.Description, info.Parameters, info.ContextTrain, info.ChatTemplate)
fmt.Println(info.Metadata["general.architecture"])
```

### Concurrency and multiple models

A single `*LLama` is safe to share across goroutines: context-mutating calls (`Predict`,
`Eval`, embeddings, state save/load) serialize on an internal lock, so concurrent calls queue
rather than race. For actual parallelism, load more instances — `Pool` manages them by name:

```go
pool := llama.NewPool()
defer pool.Free()

if _, err := pool.Load("chat", "chat-model.gguf", llama.SetContext(4096)); err != nil { /* ... */ }
if _, err := pool.Load("embed", "embed-model.gguf", llama.EnableEmbeddings); err != nil { /* ... */ }

// Different names run in parallel; same name serializes.
out, err := pool.Predict("chat", "Hello!", llama.SetTokens(64))
vec, err := pool.Embeddings("embed", "Hello!")
```

Loading the same GGUF under two names gives two independent contexts for parallel inference on
one model (each instance holds its own KV cache; weights are shared by the OS page cache when
mmap'd).

## Embedding models into Go binaries

llama.cpp loads models by file path, so an embedded model is materialized to disk once and
loaded from there. With `go:embed`:

```go
package main

import (
	_ "embed"
	"os"
	"path/filepath"

	llama "github.com/go-skynet/go-llama.cpp"
)

//go:embed models/smollm2-135m-q8.gguf
var embeddedModel []byte

// materializeModel writes the embedded GGUF next to the user cache once and
// reuses it on later runs.
func materializeModel() (string, error) {
	cacheDir, err := os.UserCacheDir()
	if err != nil {
		return "", err
	}
	path := filepath.Join(cacheDir, "myapp", "smollm2-135m-q8.gguf")

	if info, err := os.Stat(path); err == nil && info.Size() == int64(len(embeddedModel)) {
		return path, nil // already extracted
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, embeddedModel, 0o644); err != nil {
		return "", err
	}
	return path, os.Rename(tmp, path) // atomic: readers never see a partial file
}

func main() {
	path, err := materializeModel()
	if err != nil {
		panic(err)
	}
	model, err := llama.New(path, llama.SetContext(2048))
	if err != nil {
		panic(err)
	}
	defer model.Free()
	// ...
}
```

Practical notes:

- **Binary size grows by the model size.** `go:embed` stores the bytes in a read-only section;
  they are paged in lazily, so memory is not doubled at runtime, but distribution size is.
  This is reasonable up to a few hundred MB (small quantized models); for larger models prefer
  a first-run download with a checksum instead of embedding.
- The file must live inside the module at build time (`//go:embed` cannot reach outside it),
  and Git LFS is advisable for versioning it.
- The size check above is a cheap staleness guard; use a SHA-256 comparison if the model file
  name does not change between releases.
- Extraction costs one write of the model size per machine; afterwards the default `MMap`
  loading keeps startup fast because pages load on demand.

## Testing

```bash
C_INCLUDE_PATH=$PWD LIBRARY_PATH=$PWD go test ./...
```

Integration tests that exercise a real model are gated behind an env var:

```bash
TEST_GRANITE_MODEL=/path/to/granite.gguf go test -run TestGranite -v ./...
```

## Maintaining this fork

`origin` is the fork, `upstream` the original project. To move the bundled llama.cpp forward:

```bash
git -C llama.cpp fetch --tags origin
git -C llama.cpp checkout <tag-or-commit>
make clean && make libbinding.a
C_INCLUDE_PATH=$PWD LIBRARY_PATH=$PWD go test ./...
```

The Makefile does not patch the submodule. If an upstream update changes public APIs, port
`binding.cpp` first, then move the submodule pointer.

## Known gaps

- `SpeculativeSampling` is API-compatible but falls back to standard prediction with a stderr
  warning; the draft model is unused.
- Options retained for compatibility but currently inert: `SetTailFreeSamplingZ`,
  `SetPenalizeNL`, `EnableF16KV`, `SetNegativePrompt`/`SetNegativePromptScale`, `SetNDraft`.
- Multimodal predictions do not context-shift (media token positions cannot be slid), skip the
  prompt cache, and video input needs an `ffmpeg` binary in PATH at runtime.
- No reranking, no runtime LoRA management, no continuous batching, and no embedded HTTP
  server — this is a library wrapper, not `llama-server`.

## License

MIT
