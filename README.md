# [![Go Reference](https://pkg.go.dev/badge/github.com/go-skynet/go-llama.cpp.svg)](https://pkg.go.dev/github.com/go-skynet/go-llama.cpp) go-llama.cpp

[LLama.cpp](https://github.com/ggml-org/llama.cpp) golang bindings.

The go-llama.cpp bindings are high level, as such most of the work is kept into the C/C++ code to avoid any extra computational cost, be more performant and lastly ease out maintenance, while keeping the usage as simple as possible.

Check out [this](https://about.sourcegraph.com/blog/go/gophercon-2018-adventures-in-cgo-performance) and [this](https://www.cockroachlabs.com/blog/the-cost-and-complexity-of-cgo/) write-ups which summarize the impact of a low-level interface which calls C functions from Go.

If you are looking for an high-level OpenAI compatible API, check out [here](https://github.com/go-skynet/llama-cli).

## Attention!

Since https://github.com/go-skynet/go-llama.cpp/pull/180 is merged, now go-llama.cpp is not anymore compatible with `ggml` format, but it works ONLY with the new `gguf` file format. See also the upstream PR: https://github.com/ggml-org/llama.cpp/pull/2398.

If you need to use the `ggml` format, use the https://github.com/go-skynet/go-llama.cpp/releases/tag/pre-gguf tag.

## Usage

Note: This repository uses git submodules to keep track of [LLama.cpp](https://github.com/ggml-org/llama.cpp).

Clone the repository locally:

```bash
git clone --recurse-submodules https://github.com/jgYro/go-llama.cpp
```

To build the bindings locally, run:

```
cd go-llama.cpp
make libbinding.a
```

The Makefile builds the `llama.cpp` submodule with CMake, then combines the
static `llama.cpp` and `ggml` libraries with this project's `binding.cpp` into
`libbinding.a`.

On macOS, the Go bindings include the Darwin framework link flags needed by
the default Accelerate/Metal `llama.cpp` build, so plain `go test ./...` and
editor LSP builds should work after `libbinding.a` is built.

Now you can run the example with:

```
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "/model/path/here" -t 14
```

## Maintaining this fork

This fork uses `origin` for the fork and `upstream` for the original project:

```bash
origin   https://github.com/jgYro/go-llama.cpp.git
upstream https://github.com/go-skynet/go-llama.cpp
```

To update this fork from upstream:

```bash
git switch master
git fetch upstream
git merge upstream/master
go test ./...
git push origin master
```

To update the bundled `llama.cpp` version:

```bash
git -C llama.cpp fetch --tags origin
git -C llama.cpp checkout <tag-or-commit>
make clean
make libbinding.a
go test ./...
git add .gitmodules llama.cpp Makefile binding.cpp binding.h llama.go options.go
git commit -m "Update llama.cpp"
git push origin master
```

To add future local changes:

```bash
git switch master
go test ./...
git add <files>
git commit -m "Describe the change"
git push origin master
```

The Makefile does not patch the submodule in place. If an upstream
`llama.cpp` update changes public APIs, port the C++ binding first, then move
the submodule pointer and test the Go package.

## Granite / Ollama checks

This fork includes a Granite integration test that applies the model's native
chat template and verifies a JSON-only prompt/response path. `TEST_GRANITE_MODEL`
must point at a local Granite GGUF file or an Ollama blob containing that GGUF:

```bash
TEST_GRANITE_MODEL=/path/to/granite.gguf go test -run TestGraniteChatTemplateAndPredict -v
```

The `toolchat` package adds a local Go tool-calling loop on top of the
llama.cpp binding. It uses chat templates, grammar-constrained JSON envelopes,
OpenAI-shaped tool definitions, Go callbacks for tool execution, and tool-result
turns fed back into the model:

```go
runner := toolchat.Runner{
    Model: model,
    Tools: []toolchat.Tool{{
        Name:        "get_weather",
        Description: "Return weather for a city.",
        Schema:      json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}`),
        Call: func(ctx context.Context, args json.RawMessage) (any, error) {
            return map[string]any{"city": "Boston", "temperature_f": 72}, nil
        },
    }},
}

resp, err := runner.Generate(ctx, []llama.ChatMessage{
    {Role: "user", Content: "Use get_weather for Boston, then answer."},
})
```

To run the Granite end-to-end tool-loop check:

```bash
TEST_GRANITE_MODEL=/path/to/granite.gguf go test ./toolchat -run TestGraniteToolRunner -v
```

## Binding feature support

This fork targets the modern public `llama.cpp` C API and exposes the core
local-library path from Go:

- GGUF model loading, Metal/Accelerate on macOS, and configurable GPU layers.
- Context options for context size, batch/micro-batch size, max sequences,
  RoPE scaling, pooling type, attention type, and flash attention mode.
- Text generation with top-k, top-p, min-p, typical-p, top-n-sigma, XTC,
  dynamic temperature, greedy, Mirostat v1/v2, DRY, repetition/frequency/
  presence penalties, EOG suppression, GBNF grammar constraints, stop strings,
  token callbacks, and token-id logit bias.
- Prompt cache files through modern `llama_state_load_file` /
  `llama_state_save_file`.
- Embeddings, token embeddings, tokenization, detokenization, chat-template
  application, built-in chat-template listing, state save/load, and model
  metadata introspection.
- Single LoRA adapter loading at model initialization.
- Go-native tool calling via `toolchat`: JSON envelope prompting, GBNF grammar
  constraints, OpenAI-shaped tool descriptors, tool execution callbacks,
  tool-result turns, and parsing for local JSON/OpenAI-style tool-call payloads.

Known gaps compared with the full `llama.cpp` project:

- `llama-server` is not embedded here, so OpenAI/Anthropic HTTP routes,
  streaming SSE, slots, metrics, model routing, Web UI, MCP/tools, and server
  request parsing are out of scope for this library wrapper.
- `SpeculativeSampling` is still API-compatible but currently falls back to
  normal prediction; true draft-model speculation needs a separate pass.
- Multimodal image/audio/video, runtime LoRA adapter management, reranking
  endpoints, model-native `common_chat` tool template parsing, full
  JSON-schema-to-grammar conversion, and continuous batching are not yet
  exposed as Go APIs.

## Acceleration

### OpenBLAS

To build and run with OpenBLAS, for example:

```
BUILD_TYPE=openblas make libbinding.a
CGO_LDFLAGS="-lopenblas" LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run -tags openblas ./examples -m "/model/path/here" -t 14
```

### CuBLAS

To build with CuBLAS:

```
BUILD_TYPE=cublas make libbinding.a
CGO_LDFLAGS="-lcublas -lcudart -L/usr/local/cuda/lib64/" LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "/model/path/here" -t 14
```

### ROCM

To build with ROCM (HIPBLAS):

```
BUILD_TYPE=hipblas make libbinding.a
CC=/opt/rocm/llvm/bin/clang CXX=/opt/rocm/llvm/bin/clang++ CGO_LDFLAGS="-O3 --hip-link --rtlib=compiler-rt -unwindlib=libgcc -lrocblas -lhipblas" LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "/model/path/here" -ngl 64 -t 32
```

### OpenCL

```
BUILD_TYPE=clblas CLBLAS_DIR=... make libbinding.a
CGO_LDFLAGS="-lOpenCL -lclblast -L/usr/local/lib64/" LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "/model/path/here" -t 14
```


You should see something like this from the output when using the GPU:

```
ggml_opencl: selecting platform: 'Intel(R) OpenCL HD Graphics'
ggml_opencl: selecting device: 'Intel(R) Graphics [0x46a6]'
ggml_opencl: device FP16 support: true
```

## GPU offloading

### Metal (Apple Silicon)

```
make clean
BUILD_TYPE=metal make libbinding.a
go test ./...
LIBRARY_PATH=$PWD C_INCLUDE_PATH=$PWD go run ./examples -m "/model/path/here" -t 1 -ngl 1
```

Enjoy!

The documentation is available [here](https://pkg.go.dev/github.com/go-skynet/go-llama.cpp) and the full example code is [here](https://github.com/go-skynet/go-llama.cpp/blob/master/examples/main.go).

## License

MIT
