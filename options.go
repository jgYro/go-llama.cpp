package llama

type ModelOptions struct {
	ContextSize    int
	Seed           int
	NBatch         int
	NUBatch        int
	NSeqMax        int
	F16Memory      bool
	MLock          bool
	MMap           bool
	LowVRAM        bool
	Embeddings     bool
	NUMA           bool
	NGPULayers     int
	MainGPU        string
	TensorSplit    string
	FreqRopeBase   float32
	FreqRopeScale  float32
	RopeScaling    RopeScalingType
	Pooling        PoolingType
	Attention      AttentionType
	FlashAttention FlashAttentionType
	MulMatQ        *bool
	LoraBase       string
	LoraAdapter    string
	Perplexity     bool
}

type PredictOptions struct {
	Seed, Threads, Tokens, TopK, Repeat, Batch, NKeep int
	TopP, Temperature, Penalty                        float32
	NDraft                                            int
	F16KV                                             bool
	DebugMode                                         bool
	StopPrompts                                       []string
	IgnoreEOS                                         bool

	TailFreeSamplingZ   float32
	TypicalP            float32
	MinP                float32
	TopNSigma           float32
	XTCProbability      float32
	XTCThreshold        float32
	DynamicTempRange    float32
	DynamicTempExponent float32
	AdaptivePTarget     float32
	AdaptivePDecay      float32
	DryMultiplier       float32
	DryBase             float32
	DryAllowedLength    int
	DryPenaltyLastN     int
	DrySequenceBreakers []string
	FrequencyPenalty    float32
	PresencePenalty     float32
	Mirostat            int
	MirostatETA         float32
	MirostatTAU         float32
	PenalizeNL          bool
	LogitBias           string
	LogitBiases         []LogitBias
	TokenCallback       func(string) bool

	PathPromptCache             string
	MLock, MMap, PromptCacheAll bool
	PromptCacheRO               bool
	Grammar                     string
	MainGPU                     string
	TensorSplit                 string

	// Rope parameters
	RopeFreqBase  float32
	RopeFreqScale float32

	// Negative prompt parameters
	NegativePromptScale float32
	NegativePrompt      string
}

type PoolingType int

const (
	PoolingUnspecified PoolingType = -1
	PoolingNone        PoolingType = 0
	PoolingMean        PoolingType = 1
	PoolingCLS         PoolingType = 2
	PoolingLast        PoolingType = 3
	PoolingRank        PoolingType = 4
)

type AttentionType int

const (
	AttentionUnspecified AttentionType = -1
	AttentionCausal      AttentionType = 0
	AttentionNonCausal   AttentionType = 1
)

type FlashAttentionType int

const (
	FlashAttentionAuto     FlashAttentionType = -1
	FlashAttentionDisabled FlashAttentionType = 0
	FlashAttentionEnabled  FlashAttentionType = 1
)

type RopeScalingType int

const (
	RopeScalingUnspecified RopeScalingType = -1
	RopeScalingNone        RopeScalingType = 0
	RopeScalingLinear      RopeScalingType = 1
	RopeScalingYarn        RopeScalingType = 2
	RopeScalingLongRope    RopeScalingType = 3
)

type LogitBias struct {
	Token int
	Bias  float32
}

type PredictOption func(p *PredictOptions)

type ModelOption func(p *ModelOptions)

var DefaultModelOptions ModelOptions = ModelOptions{
	ContextSize:    512,
	Seed:           0,
	F16Memory:      false,
	MLock:          false,
	Embeddings:     false,
	MMap:           true,
	LowVRAM:        false,
	NBatch:         512,
	RopeScaling:    RopeScalingUnspecified,
	Pooling:        PoolingUnspecified,
	Attention:      AttentionUnspecified,
	FlashAttention: FlashAttentionAuto,
	FreqRopeBase:   0,
	FreqRopeScale:  0,
}

var DefaultOptions PredictOptions = PredictOptions{
	Seed:                -1,
	Threads:             4,
	Tokens:              128,
	Penalty:             1.1,
	Repeat:              64,
	Batch:               512,
	NKeep:               64,
	TopK:                40,
	TopP:                0.95,
	TailFreeSamplingZ:   1.0,
	TypicalP:            1.0,
	MinP:                0.0,
	TopNSigma:           -1.0,
	XTCProbability:      0.0,
	XTCThreshold:        0.1,
	DynamicTempRange:    0.0,
	DynamicTempExponent: 1.0,
	AdaptivePTarget:     -1.0,
	AdaptivePDecay:      0.0,
	DryMultiplier:       0.0,
	DryBase:             1.75,
	DryAllowedLength:    2,
	DryPenaltyLastN:     -1,
	DrySequenceBreakers: []string{"\n", ":", "\"", "*"},
	Temperature:         0.8,
	FrequencyPenalty:    0.0,
	PresencePenalty:     0.0,
	Mirostat:            0,
	MirostatTAU:         5.0,
	MirostatETA:         0.1,
	MMap:                true,
	RopeFreqBase:        0,
	RopeFreqScale:       0,
}

func SetMulMatQ(b bool) ModelOption {
	return func(p *ModelOptions) {
		p.MulMatQ = &b
	}
}

func SetLoraBase(s string) ModelOption {
	return func(p *ModelOptions) {
		p.LoraBase = s
	}
}

func SetLoraAdapter(s string) ModelOption {
	return func(p *ModelOptions) {
		p.LoraAdapter = s
	}
}

// SetContext sets the context size.
func SetContext(c int) ModelOption {
	return func(p *ModelOptions) {
		p.ContextSize = c
	}
}

func WithRopeFreqBase(f float32) ModelOption {
	return func(p *ModelOptions) {
		p.FreqRopeBase = f
	}
}

func WithRopeFreqScale(f float32) ModelOption {
	return func(p *ModelOptions) {
		p.FreqRopeScale = f
	}
}

func SetModelSeed(c int) ModelOption {
	return func(p *ModelOptions) {
		p.Seed = c
	}
}

// SetMMap sets model memory mapping.
func SetMMap(b bool) ModelOption {
	return func(p *ModelOptions) {
		p.MMap = b
	}
}

// SetNBatch sets the logical maximum batch size.
func SetNBatch(n_batch int) ModelOption {
	return func(p *ModelOptions) {
		p.NBatch = n_batch
	}
}

// SetNUBatch sets the physical micro-batch size.
func SetNUBatch(n_ubatch int) ModelOption {
	return func(p *ModelOptions) {
		p.NUBatch = n_ubatch
	}
}

// SetNSeqMax sets the maximum number of parallel sequences for the context.
func SetNSeqMax(n_seq_max int) ModelOption {
	return func(p *ModelOptions) {
		p.NSeqMax = n_seq_max
	}
}

func SetRopeScaling(t RopeScalingType) ModelOption {
	return func(p *ModelOptions) {
		p.RopeScaling = t
	}
}

func SetPoolingType(t PoolingType) ModelOption {
	return func(p *ModelOptions) {
		p.Pooling = t
	}
}

func SetAttentionType(t AttentionType) ModelOption {
	return func(p *ModelOptions) {
		p.Attention = t
	}
}

func SetFlashAttention(t FlashAttentionType) ModelOption {
	return func(p *ModelOptions) {
		p.FlashAttention = t
	}
}

// SetTensorSplit sets the tensor split ratios for multi-GPU loading.
func SetTensorSplit(maingpu string) ModelOption {
	return func(p *ModelOptions) {
		p.TensorSplit = maingpu
	}
}

// SetMainGPU sets the main_gpu
func SetMainGPU(maingpu string) ModelOption {
	return func(p *ModelOptions) {
		p.MainGPU = maingpu
	}
}

// SetPredictionTensorSplit is retained for API compatibility. Tensor split is applied at model load time.
func SetPredictionTensorSplit(maingpu string) PredictOption {
	return func(p *PredictOptions) {
		p.TensorSplit = maingpu
	}
}

// SetPredictionMainGPU is retained for API compatibility. Main GPU is applied at model load time.
func SetPredictionMainGPU(maingpu string) PredictOption {
	return func(p *PredictOptions) {
		p.MainGPU = maingpu
	}
}

// SetRopeFreqBase is retained for API compatibility. RoPE parameters are applied at model load time.
func SetRopeFreqBase(rfb float32) PredictOption {
	return func(p *PredictOptions) {
		p.RopeFreqBase = rfb
	}
}

// SetRopeFreqScale is retained for API compatibility. RoPE parameters are applied at model load time.
func SetRopeFreqScale(rfs float32) PredictOption {
	return func(p *PredictOptions) {
		p.RopeFreqScale = rfs
	}
}

// SetNDraft is retained for API compatibility. True speculative decoding is not wired yet.
func SetNDraft(nd int) PredictOption {
	return func(p *PredictOptions) {
		p.NDraft = nd
	}
}

func SetPerplexity(b bool) ModelOption {
	return func(p *ModelOptions) {
		p.Perplexity = b
	}
}

// SetNegativePromptScale is retained for API compatibility. CFG guidance is not wired yet.
func SetNegativePromptScale(nps float32) PredictOption {
	return func(p *PredictOptions) {
		p.NegativePromptScale = nps
	}
}

// SetNegativePrompt is retained for API compatibility. CFG guidance is not wired yet.
func SetNegativePrompt(np string) PredictOption {
	return func(p *PredictOptions) {
		p.NegativePrompt = np
	}
}

var EnabelLowVRAM ModelOption = func(p *ModelOptions) {
	p.LowVRAM = true
}

var EnableNUMA ModelOption = func(p *ModelOptions) {
	p.NUMA = true
}

var EnableEmbeddings ModelOption = func(p *ModelOptions) {
	p.Embeddings = true
}

var EnableF16Memory ModelOption = func(p *ModelOptions) {
	p.F16Memory = true
}

var EnableF16KV PredictOption = func(p *PredictOptions) {
	p.F16KV = true
}

var Debug PredictOption = func(p *PredictOptions) {
	p.DebugMode = true
}

var EnablePromptCacheAll PredictOption = func(p *PredictOptions) {
	p.PromptCacheAll = true
}

var EnablePromptCacheRO PredictOption = func(p *PredictOptions) {
	p.PromptCacheRO = true
}

var EnableMLock ModelOption = func(p *ModelOptions) {
	p.MLock = true
}

// Create a new PredictOptions object with the given options.
func NewModelOptions(opts ...ModelOption) ModelOptions {
	p := DefaultModelOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}

var IgnoreEOS PredictOption = func(p *PredictOptions) {
	p.IgnoreEOS = true
}

// WithGrammar sets the grammar to constrain the output of the LLM response
func WithGrammar(s string) PredictOption {
	return func(p *PredictOptions) {
		p.Grammar = s
	}
}

// SetMlock sets the memory lock.
func SetMlock(b bool) PredictOption {
	return func(p *PredictOptions) {
		p.MLock = b
	}
}

// SetMemoryMap sets memory mapping.
func SetMemoryMap(b bool) PredictOption {
	return func(p *PredictOptions) {
		p.MMap = b
	}
}

// SetGPULayers sets the number of GPU layers to use to offload computation
func SetGPULayers(n int) ModelOption {
	return func(p *ModelOptions) {
		p.NGPULayers = n
	}
}

// SetTokenCallback sets the prompts that will stop predictions.
func SetTokenCallback(fn func(string) bool) PredictOption {
	return func(p *PredictOptions) {
		p.TokenCallback = fn
	}
}

// SetStopWords sets the prompts that will stop predictions.
func SetStopWords(stop ...string) PredictOption {
	return func(p *PredictOptions) {
		p.StopPrompts = stop
	}
}

// SetSeed sets the random seed for sampling text generation.
func SetSeed(seed int) PredictOption {
	return func(p *PredictOptions) {
		p.Seed = seed
	}
}

// SetThreads sets the number of threads to use for text generation.
func SetThreads(threads int) PredictOption {
	return func(p *PredictOptions) {
		p.Threads = threads
	}
}

// SetTokens sets the number of tokens to generate.
func SetTokens(tokens int) PredictOption {
	return func(p *PredictOptions) {
		p.Tokens = tokens
	}
}

// SetTopK sets the value for top-K sampling.
func SetTopK(topk int) PredictOption {
	return func(p *PredictOptions) {
		p.TopK = topk
	}
}

// SetTopP sets the value for nucleus sampling.
func SetTopP(topp float32) PredictOption {
	return func(p *PredictOptions) {
		p.TopP = topp
	}
}

// SetTemperature sets the temperature value for text generation.
func SetTemperature(temp float32) PredictOption {
	return func(p *PredictOptions) {
		p.Temperature = temp
	}
}

// SetPathPromptCache sets the session file to store the prompt cache.
func SetPathPromptCache(f string) PredictOption {
	return func(p *PredictOptions) {
		p.PathPromptCache = f
	}
}

// SetPenalty sets the repetition penalty for text generation.
func SetPenalty(penalty float32) PredictOption {
	return func(p *PredictOptions) {
		p.Penalty = penalty
	}
}

// SetRepeat sets the number of times to repeat text generation.
func SetRepeat(repeat int) PredictOption {
	return func(p *PredictOptions) {
		p.Repeat = repeat
	}
}

// SetBatch sets the batch size.
func SetBatch(size int) PredictOption {
	return func(p *PredictOptions) {
		p.Batch = size
	}
}

// SetKeep sets the number of tokens from initial prompt to keep.
func SetNKeep(n int) PredictOption {
	return func(p *PredictOptions) {
		p.NKeep = n
	}
}

// Create a new PredictOptions object with the given options.
func NewPredictOptions(opts ...PredictOption) PredictOptions {
	p := DefaultOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}

// SetTailFreeSamplingZ is retained for API compatibility. Tail-free sampling is not available in the modern sampler API.
func SetTailFreeSamplingZ(tfz float32) PredictOption {
	return func(p *PredictOptions) {
		p.TailFreeSamplingZ = tfz
	}
}

// SetTypicalP sets the typicality parameter, p_typical.
func SetTypicalP(tp float32) PredictOption {
	return func(p *PredictOptions) {
		p.TypicalP = tp
	}
}

func SetMinP(mp float32) PredictOption {
	return func(p *PredictOptions) {
		p.MinP = mp
	}
}

func SetTopNSigma(tns float32) PredictOption {
	return func(p *PredictOptions) {
		p.TopNSigma = tns
	}
}

func SetXTC(probability, threshold float32) PredictOption {
	return func(p *PredictOptions) {
		p.XTCProbability = probability
		p.XTCThreshold = threshold
	}
}

func SetDynamicTemperature(delta, exponent float32) PredictOption {
	return func(p *PredictOptions) {
		p.DynamicTempRange = delta
		p.DynamicTempExponent = exponent
	}
}

func SetAdaptiveP(target, decay float32) PredictOption {
	return func(p *PredictOptions) {
		p.AdaptivePTarget = target
		p.AdaptivePDecay = decay
	}
}

func SetDRY(multiplier, base float32, allowedLength, penaltyLastN int) PredictOption {
	return func(p *PredictOptions) {
		p.DryMultiplier = multiplier
		p.DryBase = base
		p.DryAllowedLength = allowedLength
		p.DryPenaltyLastN = penaltyLastN
	}
}

func SetDRYSequenceBreakers(breakers ...string) PredictOption {
	return func(p *PredictOptions) {
		p.DrySequenceBreakers = breakers
	}
}

// SetFrequencyPenalty sets the frequency penalty parameter, freq_penalty.
func SetFrequencyPenalty(fp float32) PredictOption {
	return func(p *PredictOptions) {
		p.FrequencyPenalty = fp
	}
}

// SetPresencePenalty sets the presence penalty parameter, presence_penalty.
func SetPresencePenalty(pp float32) PredictOption {
	return func(p *PredictOptions) {
		p.PresencePenalty = pp
	}
}

// SetMirostat sets the mirostat parameter.
func SetMirostat(m int) PredictOption {
	return func(p *PredictOptions) {
		p.Mirostat = m
	}
}

// SetMirostatETA sets the mirostat ETA parameter.
func SetMirostatETA(me float32) PredictOption {
	return func(p *PredictOptions) {
		p.MirostatETA = me
	}
}

// SetMirostatTAU sets the mirostat TAU parameter.
func SetMirostatTAU(mt float32) PredictOption {
	return func(p *PredictOptions) {
		p.MirostatTAU = mt
	}
}

// SetPenalizeNL is retained for API compatibility. The modern penalty sampler does not expose newline-specific handling.
func SetPenalizeNL(pnl bool) PredictOption {
	return func(p *PredictOptions) {
		p.PenalizeNL = pnl
	}
}

// SetLogitBias sets token-id logit bias entries in "token:bias,token:bias" form.
func SetLogitBias(lb string) PredictOption {
	return func(p *PredictOptions) {
		p.LogitBias = lb
	}
}

func SetLogitBiasToken(token int, bias float32) PredictOption {
	return func(p *PredictOptions) {
		p.LogitBiases = append(p.LogitBiases, LogitBias{Token: token, Bias: bias})
	}
}

func SetLogitBiases(biases ...LogitBias) PredictOption {
	return func(p *PredictOptions) {
		p.LogitBiases = append(p.LogitBiases, biases...)
	}
}
