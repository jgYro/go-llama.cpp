package llama

// #cgo CXXFLAGS: -I${SRCDIR}/llama.cpp/include -I${SRCDIR}/llama.cpp/ggml/include -I${SRCDIR}/llama.cpp/ggml/src -I${SRCDIR}/llama.cpp/tools/mtmd -std=c++17
// #cgo LDFLAGS: -L${SRCDIR}/ -lbinding -lm
// #cgo linux LDFLAGS: -lstdc++
// #cgo darwin LDFLAGS: -lc++ -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
// #include "binding.h"
// #include <stdlib.h>
import "C"
import (
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"unsafe"
)

// LLama wraps a single model plus inference context. All methods are safe for
// concurrent use: operations that mutate the context (Predict, Eval,
// Embeddings, state save/load, ...) are serialized on an internal mutex, so
// concurrent calls on one instance queue rather than race. For parallel
// inference, load multiple instances (see Pool).
type LLama struct {
	mu          sync.Mutex
	state       unsafe.Pointer
	embeddings  bool
	contextSize int
}

// bindingError converts a C-side error message into a Go error and frees it.
// Falls back to the given generic message when the binding did not set one.
func bindingError(cmsg *C.char, fallback string) error {
	if cmsg != nil {
		defer C.free(unsafe.Pointer(cmsg))
		if msg := C.GoString(cmsg); msg != "" {
			return errors.New(msg)
		}
	}
	return errors.New(fallback)
}

type ChatMessage struct {
	Role    string
	Content string
}

type ModelInfo struct {
	Description           string            `json:"description"`
	Size                  uint64            `json:"size"`
	Parameters            uint64            `json:"parameters"`
	FType                 int               `json:"ftype"`
	FTypeName             string            `json:"ftype_name"`
	ContextTrain          int               `json:"context_train"`
	EmbeddingLength       int               `json:"embedding_length"`
	EmbeddingInputLength  int               `json:"embedding_input_length"`
	EmbeddingOutputLength int               `json:"embedding_output_length"`
	Layers                int               `json:"layers"`
	Heads                 int               `json:"heads"`
	HeadsKV               int               `json:"heads_kv"`
	VocabSize             int               `json:"vocab_size"`
	RopeType              int               `json:"rope_type"`
	HasEncoder            bool              `json:"has_encoder"`
	HasDecoder            bool              `json:"has_decoder"`
	IsRecurrent           bool              `json:"is_recurrent"`
	IsHybrid              bool              `json:"is_hybrid"`
	IsDiffusion           bool              `json:"is_diffusion"`
	ChatTemplate          string            `json:"chat_template"`
	Metadata              map[string]string `json:"metadata"`
}

type cPredictParams struct {
	ptr      unsafe.Pointer
	prompt   *C.char
	cstrings []*C.char
}

func (p *cPredictParams) cString(s string) *C.char {
	cs := C.CString(s)
	p.cstrings = append(p.cstrings, cs)
	return cs
}

func (p *cPredictParams) free() {
	if p.ptr != nil {
		C.llama_free_params(p.ptr)
		p.ptr = nil
	}
	for _, cs := range p.cstrings {
		C.free(unsafe.Pointer(cs))
	}
	p.cstrings = nil
}

func newCPredictParams(prompt string, po PredictOptions) *cPredictParams {
	params := &cPredictParams{}
	input := params.cString(prompt)
	params.prompt = input

	reverseCount := len(po.StopPrompts)
	reversePrompt := make([]*C.char, reverseCount)
	var pass **C.char
	for i, s := range po.StopPrompts {
		reversePrompt[i] = params.cString(s)
		pass = &reversePrompt[0]
	}

	logitBias := params.cString(formatLogitBias(po))
	dryBreakers := params.cString(strings.Join(po.DrySequenceBreakers, "\x1f"))
	params.ptr = C.llama_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.float(po.Penalty), C.int(po.Repeat),
		C.bool(po.IgnoreEOS), C.bool(po.F16KV),
		C.int(po.Batch), C.int(po.NKeep), pass, C.int(reverseCount),
		C.float(po.TailFreeSamplingZ), C.float(po.TypicalP), C.float(po.MinP), C.float(po.TopNSigma),
		C.float(po.XTCProbability), C.float(po.XTCThreshold), C.float(po.DynamicTempRange), C.float(po.DynamicTempExponent),
		C.float(po.AdaptivePTarget), C.float(po.AdaptivePDecay), C.float(po.DryMultiplier), C.float(po.DryBase),
		C.int(po.DryAllowedLength), C.int(po.DryPenaltyLastN), dryBreakers,
		C.float(po.FrequencyPenalty), C.float(po.PresencePenalty),
		C.int(po.Mirostat), C.float(po.MirostatETA), C.float(po.MirostatTAU), C.bool(po.PenalizeNL), logitBias,
		params.cString(po.PathPromptCache), C.bool(po.PromptCacheAll), C.bool(po.MLock), C.bool(po.MMap),
		params.cString(po.MainGPU), params.cString(po.TensorSplit),
		C.bool(po.PromptCacheRO),
		params.cString(po.Grammar),
		C.float(po.RopeFreqBase), C.float(po.RopeFreqScale), C.float(po.NegativePromptScale), params.cString(po.NegativePrompt),
		C.int(po.NDraft),
	)

	return params
}

func formatLogitBias(po PredictOptions) string {
	var parts []string
	if po.LogitBias != "" {
		parts = append(parts, po.LogitBias)
	}
	for _, bias := range po.LogitBiases {
		parts = append(parts, strconv.Itoa(bias.Token)+":"+strconv.FormatFloat(float64(bias.Bias), 'g', -1, 32))
	}
	return strings.Join(parts, ",")
}

func New(model string, opts ...ModelOption) (*LLama, error) {
	mo := NewModelOptions(opts...)
	modelPath := C.CString(model)
	defer C.free(unsafe.Pointer(modelPath))
	loraBase := C.CString(mo.LoraBase)
	defer C.free(unsafe.Pointer(loraBase))
	loraAdapter := C.CString(mo.LoraAdapter)
	defer C.free(unsafe.Pointer(loraAdapter))
	mainGPU := C.CString(mo.MainGPU)
	defer C.free(unsafe.Pointer(mainGPU))
	tensorSplit := C.CString(mo.TensorSplit)
	defer C.free(unsafe.Pointer(tensorSplit))

	MulMatQ := true

	if mo.MulMatQ != nil {
		MulMatQ = *mo.MulMatQ
	}

	var cerr *C.char
	result := C.load_model(modelPath,
		C.int(mo.ContextSize), C.int(mo.Seed),
		C.bool(mo.F16Memory), C.bool(mo.MLock), C.bool(mo.Embeddings), C.bool(mo.MMap), C.bool(mo.LowVRAM),
		C.int(mo.NGPULayers), C.int(mo.NBatch), mainGPU, tensorSplit, C.bool(mo.NUMA),
		C.float(mo.FreqRopeBase), C.float(mo.FreqRopeScale),
		C.int(mo.RopeScaling), C.int(mo.Pooling), C.int(mo.Attention), C.int(mo.FlashAttention), C.int(mo.NUBatch), C.int(mo.NSeqMax),
		C.bool(MulMatQ), loraAdapter, loraBase, C.bool(mo.Perplexity),
		&cerr,
	)

	if result == nil {
		return nil, fmt.Errorf("failed loading model: %w", bindingError(cerr, "unknown cause"))
	}

	if mo.MMProj != "" {
		mmproj := C.CString(mo.MMProj)
		defer C.free(unsafe.Pointer(mmproj))
		var mmerr *C.char
		if C.llama_binding_load_mmproj(result, mmproj, C.bool(mo.NGPULayers > 0), C.int(0), &mmerr) != 0 {
			C.llama_binding_free_model(result)
			return nil, fmt.Errorf("failed loading model: %w", bindingError(mmerr, "unknown cause"))
		}
	}

	ll := &LLama{state: result, contextSize: mo.ContextSize, embeddings: mo.Embeddings}
	return ll, nil
}

// MediaMarker is the placeholder that positions a media file inside a prompt,
// e.g. "<__media__>\nDescribe this image." One marker per WithMedia file.
func MediaMarker() string {
	return C.GoString(C.llama_binding_media_marker())
}

// SupportsVision reports whether a multimodal projector with image support is loaded.
func (l *LLama) SupportsVision() bool {
	return l.state != nil && C.llama_binding_supports_vision(l.state) != 0
}

// SupportsAudio reports whether a multimodal projector with audio support is loaded.
func (l *LLama) SupportsAudio() bool {
	return l.state != nil && C.llama_binding_supports_audio(l.state) != 0
}

func BuiltinChatTemplates() ([]string, error) {
	var out *C.char
	ret := C.llama_builtin_chat_templates_json(&out)
	if ret != 0 {
		return nil, fmt.Errorf("failed to load builtin chat templates")
	}
	if out == nil {
		return nil, fmt.Errorf("builtin chat templates returned no output")
	}
	defer C.free(unsafe.Pointer(out))

	var templates []string
	if err := json.Unmarshal([]byte(C.GoString(out)), &templates); err != nil {
		return nil, err
	}
	return templates, nil
}

func (l *LLama) Free() {
	l.mu.Lock()
	defer l.mu.Unlock()

	if l.state == nil {
		return
	}
	setCallback(l.state, nil)
	C.llama_binding_free_model(l.state)
	l.state = nil
}

func (l *LLama) LoadState(state string) error {
	d := C.CString(state)
	w := C.CString("rb")
	defer C.free(unsafe.Pointer(d)) // free allocated C string
	defer C.free(unsafe.Pointer(w)) // free allocated C string

	l.mu.Lock()
	defer l.mu.Unlock()

	result := C.load_state(l.state, d, w)
	if result != 0 {
		return fmt.Errorf("error while loading state")
	}

	return nil
}

func (l *LLama) SaveState(dst string) error {
	d := C.CString(dst)
	w := C.CString("wb")
	defer C.free(unsafe.Pointer(d)) // free allocated C string
	defer C.free(unsafe.Pointer(w)) // free allocated C string

	l.mu.Lock()
	defer l.mu.Unlock()

	if C.save_state(l.state, d, w) != 0 {
		return fmt.Errorf("error while saving state")
	}

	return nil
}

func (l *LLama) ModelInfo() (ModelInfo, error) {
	if l.state == nil {
		return ModelInfo{}, fmt.Errorf("model is not loaded")
	}

	var out *C.char
	ret := C.llama_model_info_json(l.state, &out)
	if ret != 0 {
		return ModelInfo{}, fmt.Errorf("model info failed")
	}
	if out == nil {
		return ModelInfo{}, fmt.Errorf("model info returned no output")
	}
	defer C.free(unsafe.Pointer(out))

	var info ModelInfo
	if err := json.Unmarshal([]byte(C.GoString(out)), &info); err != nil {
		return ModelInfo{}, err
	}
	return info, nil
}

func (l *LLama) embeddingSize() (int, error) {
	size := int(C.llama_embedding_size(l.state))
	if size <= 0 {
		return 0, fmt.Errorf("invalid embedding size %d", size)
	}
	return size, nil
}

// Token Embeddings
func (l *LLama) TokenEmbeddings(tokens []int, opts ...PredictOption) ([]float32, error) {
	if !l.embeddings {
		return []float32{}, fmt.Errorf("model loaded without embeddings")
	}

	po := NewPredictOptions(opts...)

	outSize, err := l.embeddingSize()
	if err != nil {
		return nil, err
	}
	floats := make([]float32, outSize)

	var myArray *C.int
	if len(tokens) > 0 {
		myArray = (*C.int)(C.malloc(C.size_t(len(tokens)) * C.sizeof_int))
		defer C.free(unsafe.Pointer(myArray))
		for i, v := range tokens {
			(*[1<<31 - 1]int32)(unsafe.Pointer(myArray))[i] = int32(v)
		}
	}

	params := newCPredictParams("", po)
	defer params.free()

	l.mu.Lock()
	defer l.mu.Unlock()

	var cerr *C.char
	ret := C.get_token_embeddings(params.ptr, l.state, myArray, C.int(len(tokens)), (*C.float)(&floats[0]), &cerr)
	if ret != 0 {
		return floats, fmt.Errorf("embedding inference failed: %w", bindingError(cerr, "unknown cause"))
	}
	return floats, nil
}

// Embeddings
func (l *LLama) Embeddings(text string, opts ...PredictOption) ([]float32, error) {
	if !l.embeddings {
		return []float32{}, fmt.Errorf("model loaded without embeddings")
	}

	po := NewPredictOptions(opts...)

	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	outSize, err := l.embeddingSize()
	if err != nil {
		return nil, err
	}
	floats := make([]float32, outSize)

	params := newCPredictParams(text, po)
	defer params.free()

	l.mu.Lock()
	defer l.mu.Unlock()

	var cerr *C.char
	ret := C.get_embeddings(params.ptr, l.state, (*C.float)(&floats[0]), &cerr)
	if ret != 0 {
		return floats, fmt.Errorf("embedding inference failed: %w", bindingError(cerr, "unknown cause"))
	}

	return floats, nil
}

func (l *LLama) Eval(text string, opts ...PredictOption) error {
	po := NewPredictOptions(opts...)

	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	params := newCPredictParams(text, po)
	defer params.free()

	l.mu.Lock()
	defer l.mu.Unlock()

	var cerr *C.char
	ret := C.eval(params.ptr, l.state, params.prompt, &cerr)
	if ret != 0 {
		return fmt.Errorf("eval failed: %w", bindingError(cerr, "unknown cause"))
	}

	return nil
}

// SpeculativeSampling is retained for API compatibility. True speculative
// decoding is not wired to the modern llama.cpp API yet: the draft model is
// unused and generation falls back to standard prediction on the target model.
func (l *LLama) SpeculativeSampling(ll *LLama, text string, opts ...PredictOption) (string, error) {
	po := NewPredictOptions(opts...)

	if po.TokenCallback != nil {
		setCallback(l.state, po.TokenCallback)
		defer setCallback(l.state, nil)
	}

	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	params := newCPredictParams(text, po)
	defer params.free()

	l.mu.Lock()
	defer l.mu.Unlock()

	var out *C.char
	var cerr *C.char
	ret := C.speculative_sampling(params.ptr, l.state, ll.state, &out, C.bool(po.DebugMode), &cerr)
	if ret != 0 {
		return "", fmt.Errorf("inference failed: %w", bindingError(cerr, "unknown cause"))
	}
	if out == nil {
		return "", fmt.Errorf("inference returned no output")
	}
	defer C.free(unsafe.Pointer(out))

	return cleanPredictionResult(C.GoString(out), po.StopPrompts), nil
}

func (l *LLama) Predict(text string, opts ...PredictOption) (string, error) {
	po := NewPredictOptions(opts...)

	if po.TokenCallback != nil {
		setCallback(l.state, po.TokenCallback)
		defer setCallback(l.state, nil)
	}

	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	// Media prompts route through the multimodal path. Each file needs one
	// marker in the prompt; prepend them when the prompt has none.
	if len(po.MediaPaths) > 0 {
		marker := MediaMarker()
		if !strings.Contains(text, marker) {
			text = strings.Repeat(marker+"\n", len(po.MediaPaths)) + text
		}
	}

	params := newCPredictParams(text, po)
	defer params.free()

	l.mu.Lock()
	defer l.mu.Unlock()

	var out *C.char
	var cerr *C.char
	var ret C.int
	if len(po.MediaPaths) > 0 {
		media := make([]*C.char, len(po.MediaPaths))
		for i, path := range po.MediaPaths {
			media[i] = params.cString(path)
		}
		ret = C.llama_predict_mtmd(params.ptr, l.state, (**C.char)(unsafe.Pointer(&media[0])), C.int(len(media)), &out, C.bool(po.DebugMode), &cerr)
	} else {
		ret = C.llama_predict(params.ptr, l.state, &out, C.bool(po.DebugMode), &cerr)
	}
	if ret != 0 {
		return "", fmt.Errorf("inference failed: %w", bindingError(cerr, "unknown cause"))
	}
	if out == nil {
		return "", fmt.Errorf("inference returned no output")
	}
	defer C.free(unsafe.Pointer(out))

	return cleanPredictionResult(C.GoString(out), po.StopPrompts), nil
}

func (l *LLama) ApplyChatTemplate(messages []ChatMessage, addGenerationPrompt bool) (string, error) {
	if l.state == nil {
		return "", fmt.Errorf("model is not loaded")
	}
	if len(messages) == 0 {
		return "", nil
	}

	roles := make([]*C.char, len(messages))
	contents := make([]*C.char, len(messages))
	for i, message := range messages {
		roles[i] = C.CString(message.Role)
		contents[i] = C.CString(message.Content)
	}
	defer func() {
		for i := range messages {
			C.free(unsafe.Pointer(roles[i]))
			C.free(unsafe.Pointer(contents[i]))
		}
	}()

	var out *C.char
	ret := C.llama_apply_chat_template(
		l.state,
		(**C.char)(unsafe.Pointer(&roles[0])),
		(**C.char)(unsafe.Pointer(&contents[0])),
		C.int(len(messages)),
		C.bool(addGenerationPrompt),
		&out,
	)
	if ret != 0 {
		return "", fmt.Errorf("chat template formatting failed")
	}
	if out == nil {
		return "", fmt.Errorf("chat template formatting returned no output")
	}
	defer C.free(unsafe.Pointer(out))

	return C.GoString(out), nil
}

func (l *LLama) Detokenize(tokens []int32, removeSpecial, unparseSpecial bool) (string, error) {
	if l.state == nil {
		return "", fmt.Errorf("model is not loaded")
	}
	if len(tokens) == 0 {
		return "", nil
	}

	cTokens := make([]C.int, len(tokens))
	for i, token := range tokens {
		cTokens[i] = C.int(token)
	}

	var out *C.char
	ret := C.llama_detokenize_tokens(
		l.state,
		(*C.int)(unsafe.Pointer(&cTokens[0])),
		C.int(len(cTokens)),
		C.bool(removeSpecial),
		C.bool(unparseSpecial),
		&out,
	)
	if ret != 0 {
		return "", fmt.Errorf("detokenize failed")
	}
	if out == nil {
		return "", fmt.Errorf("detokenize returned no output")
	}
	defer C.free(unsafe.Pointer(out))

	return C.GoString(out), nil
}

// cleanPredictionResult trims a trailing stop word from the result. The binding
// returns only generated text (no prompt echo), so leading characters are
// legitimate output and must be preserved.
func cleanPredictionResult(res string, stopPrompts []string) string {
	for _, s := range stopPrompts {
		if s != "" {
			res = strings.TrimSuffix(res, s)
		}
	}

	return res
}

// tokenize has an interesting return property: negative lengths (potentially) have meaning.
// Therefore, return the length seperate from the slice and error - all three can be used together
func (l *LLama) TokenizeString(text string, opts ...PredictOption) (int32, []int32, error) {
	po := NewPredictOptions(opts...)

	if po.Tokens == 0 {
		po.Tokens = 4096 // ???
	}
	out := make([]C.int, po.Tokens)

	params := newCPredictParams(text, po)
	defer params.free()

	tokRet := C.llama_tokenize_string(params.ptr, l.state, (*C.int)(unsafe.Pointer(&out[0]))) //, C.int(po.Tokens), true)

	if tokRet < 0 {
		return int32(tokRet), []int32{}, fmt.Errorf("llama_tokenize_string returned negative count %d", tokRet)
	}

	// TODO: Is this loop still required to unbox cgo to go?
	gTokRet := int32(tokRet)

	gLenOut := min(len(out), int(gTokRet))

	goSlice := make([]int32, gLenOut)
	for i := 0; i < gLenOut; i++ {
		goSlice[i] = int32(out[i])
	}

	return gTokRet, goSlice, nil
}

// CGo only allows us to use static calls from C to Go, we can't just dynamically pass in func's.
// This is the next best thing, we register the callbacks in this map and call tokenCallback from
// the C code. We also attach a finalizer to LLama, so it will unregister the callback when the
// garbage collection frees it.

// SetTokenCallback registers a callback for the individual tokens created when running Predict. It
// will be called once for each token. The callback shall return true as long as the model should
// continue predicting the next token. When the callback returns false the predictor will return.
// The tokens are just converted into Go strings, they are not trimmed or otherwise changed. Also
// the tokens may not be valid UTF-8.
// Pass in nil to remove a callback.
//
// It is save to call this method while a prediction is running.
func (l *LLama) SetTokenCallback(callback func(token string) bool) {
	setCallback(l.state, callback)
}

var (
	m         sync.RWMutex
	callbacks = map[uintptr]func(string) bool{}
)

//export tokenCallback
func tokenCallback(statePtr unsafe.Pointer, token *C.char) bool {
	m.RLock()
	defer m.RUnlock()

	if callback, ok := callbacks[uintptr(statePtr)]; ok {
		return callback(C.GoString(token))
	}

	return true
}

// setCallback can be used to register a token callback for LLama. Pass in a nil callback to
// remove the callback.
func setCallback(statePtr unsafe.Pointer, callback func(string) bool) {
	m.Lock()
	defer m.Unlock()

	if callback == nil {
		delete(callbacks, uintptr(statePtr))
	} else {
		callbacks[uintptr(statePtr)] = callback
	}
}
