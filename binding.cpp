#include "binding.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

struct llama_binding_params {
    std::string prompt;
    int seed;
    int threads;
    int tokens;
    int top_k;
    int repeat;
    int batch;
    int n_keep;
    float top_p;
    float temp;
    float repeat_penalty;
    float typical_p;
    float min_p;
    float top_n_sigma;
    float xtc_probability;
    float xtc_threshold;
    float dynatemp_range;
    float dynatemp_exponent;
    float adaptive_p_target;
    float adaptive_p_decay;
    float dry_multiplier;
    float dry_base;
    int dry_allowed_length;
    int dry_penalty_last_n;
    float frequency_penalty;
    float presence_penalty;
    int mirostat;
    float mirostat_eta;
    float mirostat_tau;
    bool ignore_eos;
    bool penalize_nl;
    std::string session_file;
    bool prompt_cache_all;
    bool prompt_cache_ro;
    std::string grammar;
    std::vector<std::string> antiprompt;
    std::vector<std::string> dry_sequence_breakers;
    std::vector<llama_logit_bias> logit_bias;
};

struct llama_binding_state {
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab = nullptr;
    std::vector<float> tensor_split;
    std::vector<llama_adapter_lora *> adapters;
};

static std::once_flag g_backend_once;

static void llama_binding_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level >= GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static void ensure_backend(bool numa) {
    std::call_once(g_backend_once, []() {
        llama_log_set(llama_binding_log_callback, nullptr);
        llama_backend_init();
        ggml_backend_load_all();
    });

    if (numa) {
        llama_numa_init(GGML_NUMA_STRATEGY_DISTRIBUTE);
    }
}

static int copy_result_string(const std::string & src, char ** result) {
    if (result == nullptr) {
        return 1;
    }

    *result = static_cast<char *>(malloc(src.size() + 1));
    if (*result == nullptr) {
        return 1;
    }

    memcpy(*result, src.c_str(), src.size() + 1);
    return 0;
}

// Copies a human-readable failure reason into *error_out (malloc'd, freed by
// the Go side) so callers get the cause instead of a bare failure code.
static void set_binding_error(char ** error_out, const std::string & message) {
    if (error_out == nullptr) {
        return;
    }
    *error_out = static_cast<char *>(malloc(message.size() + 1));
    if (*error_out != nullptr) {
        memcpy(*error_out, message.c_str(), message.size() + 1);
    }
}

static std::string trim_copy(const std::string & value) {
    size_t begin = 0;
    while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
        begin++;
    }

    size_t end = value.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        end--;
    }

    return value.substr(begin, end - begin);
}

static std::vector<std::string> split_string(const std::string & value, char delimiter) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= value.size()) {
        size_t end = value.find(delimiter, start);
        if (end == std::string::npos) {
            end = value.size();
        }
        std::string item = value.substr(start, end - start);
        if (!item.empty()) {
            out.push_back(item);
        }
        if (end == value.size()) {
            break;
        }
        start = end + 1;
    }
    return out;
}

static std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (unsigned char ch : value) {
        switch (ch) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (ch < 0x20) {
                char buf[7];
                snprintf(buf, sizeof(buf), "\\u%04x", ch);
                out += buf;
            } else {
                out.push_back(static_cast<char>(ch));
            }
            break;
        }
    }
    return out;
}

static bool parse_float(const std::string & value, float & out) {
    const std::string v = trim_copy(value);
    if (v == "-inf" || v == "-infinity" || v == "-Inf" || v == "-Infinity") {
        out = -INFINITY;
        return true;
    }
    if (v == "inf" || v == "+inf" || v == "infinity" || v == "+infinity" ||
        v == "Inf" || v == "+Inf" || v == "Infinity" || v == "+Infinity") {
        out = INFINITY;
        return true;
    }

    char * end = nullptr;
    errno = 0;
    float parsed = std::strtof(v.c_str(), &end);
    if (errno != 0 || end == v.c_str()) {
        return false;
    }
    while (end != nullptr && *end != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end))) {
            return false;
        }
        end++;
    }
    out = parsed;
    return true;
}

static std::vector<llama_logit_bias> parse_logit_biases(const char * spec) {
    std::vector<llama_logit_bias> biases;
    if (spec == nullptr || spec[0] == '\0') {
        return biases;
    }

    std::string normalized(spec);
    std::replace(normalized.begin(), normalized.end(), ';', ',');
    for (const std::string & raw_entry : split_string(normalized, ',')) {
        const std::string entry = trim_copy(raw_entry);
        if (entry.empty()) {
            continue;
        }

        size_t sep = entry.find(':');
        if (sep == std::string::npos) {
            sep = entry.find('=');
        }
        if (sep == std::string::npos) {
            continue;
        }

        const std::string token_text = trim_copy(entry.substr(0, sep));
        const std::string bias_text = trim_copy(entry.substr(sep + 1));
        char * token_end = nullptr;
        errno = 0;
        long token = std::strtol(token_text.c_str(), &token_end, 10);
        if (errno != 0 || token_end == token_text.c_str() || token < 0) {
            continue;
        }
        while (token_end != nullptr && *token_end != '\0') {
            if (!std::isspace(static_cast<unsigned char>(*token_end))) {
                token = -1;
                break;
            }
            token_end++;
        }
        if (token < 0) {
            continue;
        }

        float bias = 0.0f;
        if (!parse_float(bias_text, bias)) {
            continue;
        }
        biases.push_back(llama_logit_bias{static_cast<llama_token>(token), bias});
    }

    return biases;
}

static void append_json_string_field(std::string & json, const char * key, const std::string & value, bool comma = true) {
    if (comma) {
        json += ",";
    }
    json += "\"";
    json += key;
    json += "\":\"";
    json += json_escape(value);
    json += "\"";
}

static bool file_exists(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

// Find the earliest stop word occurrence that involves text appended at or after
// appended_from. Matches may start before appended_from when a stop word spans
// token pieces. Returns std::string::npos when no stop word matches.
static size_t find_stop_position(const std::string & text, size_t appended_from, const std::vector<std::string> & stops) {
    size_t best = std::string::npos;
    for (const auto & stop : stops) {
        if (stop.empty()) {
            continue;
        }
        const size_t from = appended_from >= stop.size() - 1 ? appended_from - (stop.size() - 1) : 0;
        const size_t pos = text.find(stop, from);
        if (pos < best) {
            best = pos;
        }
    }
    return best;
}

static bool parse_tensor_split(const char * tensor_split, std::vector<float> & result) {
    result.clear();
    if (tensor_split == nullptr || tensor_split[0] == '\0') {
        return true;
    }

    std::stringstream ss(tensor_split);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            continue;
        }
        float value = 0.0f;
        if (!parse_float(item, value)) {
            return false;
        }
        result.push_back(value);
    }
    return true;
}

static bool parse_main_gpu(const char * maingpu, int & result) {
    char * end = nullptr;
    errno = 0;
    long value = std::strtol(maingpu, &end, 10);
    if (errno != 0 || end == maingpu || value < 0 || value > INT_MAX) {
        return false;
    }
    while (end != nullptr && *end != '\0') {
        if (!std::isspace(static_cast<unsigned char>(*end))) {
            return false;
        }
        end++;
    }
    result = static_cast<int>(value);
    return true;
}

static std::string token_to_piece(const llama_vocab * vocab, llama_token token) {
    char stack_buf[256];
    int n = llama_token_to_piece(vocab, token, stack_buf, sizeof(stack_buf), 0, true);
    if (n >= 0) {
        return std::string(stack_buf, n);
    }

    std::vector<char> heap_buf(static_cast<size_t>(-n));
    n = llama_token_to_piece(vocab, token, heap_buf.data(), heap_buf.size(), 0, true);
    if (n < 0) {
        return "";
    }
    return std::string(heap_buf.data(), n);
}

static std::vector<llama_token> tokenize_text(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int n_tokens = -llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), nullptr, 0, add_special, true);
    if (n_tokens <= 0) {
        return {};
    }

    std::vector<llama_token> tokens(static_cast<size_t>(n_tokens));
    int actual = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()), tokens.data(), n_tokens, add_special, true);
    if (actual < 0) {
        return {};
    }

    tokens.resize(static_cast<size_t>(actual));
    return tokens;
}

static int decode_tokens(llama_context * ctx, llama_token * tokens, int count, int batch_size) {
    if (count <= 0) {
        return 0;
    }

    int offset = 0;
    const int ctx_batch = std::max(1, static_cast<int>(llama_n_batch(ctx)));
    const int limit = std::max(1, std::min(batch_size, ctx_batch));
    while (offset < count) {
        const int n = std::min(limit, count - offset);
        llama_batch batch = llama_batch_get_one(tokens + offset, n);
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            return ret;
        }
        offset += n;
    }
    return 0;
}

static llama_sampler * create_sampler(llama_binding_state * state, const llama_binding_params * params, char ** error_out) {
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(chain_params);

    const int n_vocab = llama_vocab_n_tokens(state->vocab);

    std::vector<llama_logit_bias> biases = params->logit_bias;
    if (params->ignore_eos) {
        for (llama_token token = 0; token < n_vocab; token++) {
            if (llama_vocab_is_eog(state->vocab, token)) {
                biases.push_back(llama_logit_bias{token, -INFINITY});
            }
        }
    }
    if (!biases.empty()) {
        llama_sampler_chain_add(sampler, llama_sampler_init_logit_bias(n_vocab, static_cast<int32_t>(biases.size()), biases.data()));
    }

    if (!params->grammar.empty()) {
        llama_sampler * grammar = llama_sampler_init_grammar(state->vocab, params->grammar.c_str(), "root");
        if (grammar == nullptr) {
            set_binding_error(error_out, "failed to parse GBNF grammar");
            fprintf(stderr, "%s: error: failed to parse grammar\n", __func__);
            llama_sampler_free(sampler);
            return nullptr;
        }
        llama_sampler_chain_add(sampler, grammar);
    }

    if (params->mirostat == 1) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params->temp));
        llama_sampler_chain_add(sampler, llama_sampler_init_mirostat(n_vocab, static_cast<uint32_t>(params->seed), params->mirostat_tau, params->mirostat_eta, 100));
        return sampler;
    }

    if (params->mirostat == 2) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params->temp));
        llama_sampler_chain_add(sampler, llama_sampler_init_mirostat_v2(static_cast<uint32_t>(params->seed), params->mirostat_tau, params->mirostat_eta));
        return sampler;
    }

    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(params->repeat, params->repeat_penalty, params->frequency_penalty, params->presence_penalty));

    if (params->dry_multiplier != 0.0f) {
        std::vector<const char *> breaker_ptrs;
        breaker_ptrs.reserve(params->dry_sequence_breakers.size());
        for (const auto & breaker : params->dry_sequence_breakers) {
            breaker_ptrs.push_back(breaker.c_str());
        }
        llama_sampler_chain_add(
            sampler,
            llama_sampler_init_dry(
                state->vocab,
                llama_model_n_ctx_train(state->model),
                params->dry_multiplier,
                params->dry_base,
                params->dry_allowed_length,
                params->dry_penalty_last_n,
                breaker_ptrs.empty() ? nullptr : breaker_ptrs.data(),
                breaker_ptrs.size()));
    }

    if (params->top_n_sigma >= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_n_sigma(params->top_n_sigma));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params->top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_typical(params->typical_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params->top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(params->min_p, 1));
    if (params->xtc_probability > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_xtc(params->xtc_probability, params->xtc_threshold, 1, static_cast<uint32_t>(params->seed)));
    }

    if (params->temp <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else if (params->adaptive_p_target >= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_adaptive_p(params->adaptive_p_target, params->adaptive_p_decay, static_cast<uint32_t>(params->seed)));
    } else {
        if (params->dynatemp_range > 0.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_temp_ext(params->temp, params->dynatemp_range, params->dynatemp_exponent));
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(params->temp));
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(static_cast<uint32_t>(params->seed)));
    }

    return sampler;
}

static int embedding_for_tokens(llama_binding_state * state, std::vector<llama_token> & tokens, llama_binding_params * params, float * res_embeddings, char ** error_out) {
    if (state == nullptr || state->ctx == nullptr || state->model == nullptr || res_embeddings == nullptr) {
        set_binding_error(error_out, "embeddings called without a loaded model");
        return 1;
    }

    if (tokens.empty()) {
        tokens.push_back(llama_vocab_bos(state->vocab));
    }

    llama_memory_clear(llama_get_memory(state->ctx), true);
    llama_set_embeddings(state->ctx, true);
    llama_set_n_threads(state->ctx, params->threads, params->threads);

    int ret = decode_tokens(state->ctx, tokens.data(), static_cast<int>(tokens.size()), params->batch > 0 ? params->batch : static_cast<int>(llama_n_batch(state->ctx)));
    if (ret != 0) {
        set_binding_error(error_out, "embedding decode failed with code " + std::to_string(ret));
        return ret;
    }

    const int n_embd = llama_model_n_embd_out(state->model);
    float * embeddings = llama_get_embeddings_ith(state->ctx, -1);
    if (embeddings == nullptr) {
        embeddings = llama_get_embeddings_seq(state->ctx, 0);
    }
    if (embeddings == nullptr) {
        embeddings = llama_get_embeddings(state->ctx);
    }
    if (embeddings == nullptr) {
        set_binding_error(error_out, "model returned no embeddings; was it loaded with embeddings enabled?");
        return 1;
    }

    for (int i = 0; i < n_embd; i++) {
        res_embeddings[i] = embeddings[i];
    }

    return 0;
}

void* load_model(const char *fname,
                 int n_ctx,
                 int n_seed,
                 bool memory_f16,
                 bool mlock,
                 bool embeddings,
                 bool mmap,
                 bool low_vram,
                 int n_gpu,
                 int n_batch,
                 const char *maingpu,
                 const char *tensorsplit,
                 bool numa,
                 float rope_freq_base,
                 float rope_freq_scale,
                 int rope_scaling_type,
                 int pooling_type,
                 int attention_type,
                 int flash_attention_type,
                 int n_ubatch,
                 int n_seq_max,
                 bool mul_mat_q,
                 const char *lora,
                 const char *lora_base,
                 bool perplexity,
                 char **error_out) {
    (void) n_seed;
    (void) memory_f16;
    (void) low_vram;
    (void) mul_mat_q;
    (void) lora_base;
    (void) perplexity;

    ensure_backend(numa);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu;
    model_params.use_mlock = mlock;
    model_params.use_mmap = mmap;

    std::vector<float> tensor_split;
    if (!parse_tensor_split(tensorsplit, tensor_split)) {
        set_binding_error(error_out, std::string("invalid tensor split \"") + tensorsplit + "\" (expected comma-separated numbers)");
        return nullptr;
    }
    if (!tensor_split.empty()) {
        model_params.tensor_split = tensor_split.data();
    }

    if (maingpu != nullptr && maingpu[0] != '\0') {
        int main_gpu = 0;
        if (!parse_main_gpu(maingpu, main_gpu)) {
            set_binding_error(error_out, std::string("invalid main GPU \"") + maingpu + "\" (expected a device index)");
            return nullptr;
        }
        model_params.main_gpu = main_gpu;
    }

    llama_model * model = llama_model_load_from_file(fname, model_params);
    if (model == nullptr) {
        set_binding_error(error_out, std::string("unable to load model ") + fname);
        return nullptr;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx > 0 ? static_cast<uint32_t>(n_ctx) : 0;
    ctx_params.n_batch = n_batch > 0 ? static_cast<uint32_t>(n_batch) : 512;
    ctx_params.n_ubatch = n_ubatch > 0 ? static_cast<uint32_t>(n_ubatch) : ctx_params.n_batch;
    if (n_seq_max > 0) {
        ctx_params.n_seq_max = static_cast<uint32_t>(n_seq_max);
    }
    ctx_params.embeddings = embeddings;
    ctx_params.no_perf = false;
    ctx_params.rope_freq_base = rope_freq_base;
    ctx_params.rope_freq_scale = rope_freq_scale;
    if (rope_scaling_type >= LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED && rope_scaling_type <= LLAMA_ROPE_SCALING_TYPE_MAX_VALUE) {
        ctx_params.rope_scaling_type = static_cast<enum llama_rope_scaling_type>(rope_scaling_type);
    }
    if (pooling_type >= LLAMA_POOLING_TYPE_UNSPECIFIED && pooling_type <= LLAMA_POOLING_TYPE_RANK) {
        ctx_params.pooling_type = static_cast<enum llama_pooling_type>(pooling_type);
    }
    if (attention_type >= LLAMA_ATTENTION_TYPE_UNSPECIFIED && attention_type <= LLAMA_ATTENTION_TYPE_NON_CAUSAL) {
        ctx_params.attention_type = static_cast<enum llama_attention_type>(attention_type);
    }
    if (flash_attention_type >= LLAMA_FLASH_ATTN_TYPE_AUTO && flash_attention_type <= LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        ctx_params.flash_attn_type = static_cast<enum llama_flash_attn_type>(flash_attention_type);
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        set_binding_error(error_out, std::string("failed to create context for ") + fname);
        llama_model_free(model);
        return nullptr;
    }

    llama_binding_state * state = new llama_binding_state;
    state->model = model;
    state->ctx = ctx;
    state->vocab = llama_model_get_vocab(model);
    state->tensor_split = tensor_split;

    if (lora != nullptr && lora[0] != '\0') {
        llama_adapter_lora * adapter = llama_adapter_lora_init(model, lora);
        if (adapter == nullptr) {
            set_binding_error(error_out, std::string("failed to load LoRA adapter ") + lora);
            llama_binding_free_model(state);
            return nullptr;
        }
        state->adapters.push_back(adapter);
        std::vector<float> scales(state->adapters.size(), 1.0f);
        if (llama_set_adapters_lora(ctx, state->adapters.data(), state->adapters.size(), scales.data()) != 0) {
            set_binding_error(error_out, std::string("failed to apply LoRA adapter ") + lora);
            llama_binding_free_model(state);
            return nullptr;
        }
    }

    return state;
}

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty,
                            int repeat_last_n, bool ignore_eos, bool memory_f16,
                            int n_batch, int n_keep, const char** antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float min_p, float top_n_sigma, float xtc_probability, float xtc_threshold, float dynatemp_range, float dynatemp_exponent, float adaptive_p_target, float adaptive_p_decay, float dry_multiplier, float dry_base, int dry_allowed_length, int dry_penalty_last_n, const char *dry_sequence_breakers, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *logit_bias, const char *session_file, bool prompt_cache_all, bool mlock, bool mmap, const char *maingpu, const char *tensorsplit,
                            bool prompt_cache_ro, const char *grammar, float rope_freq_base, float rope_freq_scale, float negative_prompt_scale, const char* negative_prompt,
                            int n_draft) {
    (void) memory_f16;
    (void) tfs_z;
    (void) mlock;
    (void) mmap;
    (void) maingpu;
    (void) tensorsplit;
    (void) rope_freq_base;
    (void) rope_freq_scale;
    (void) negative_prompt_scale;
    (void) negative_prompt;
    (void) n_draft;

    llama_binding_params * params = new llama_binding_params;
    params->prompt = prompt != nullptr ? prompt : "";
    params->seed = seed < 0 ? static_cast<int>(LLAMA_DEFAULT_SEED) : seed;
    params->threads = threads > 0 ? threads : 1;
    params->tokens = tokens;
    params->top_k = top_k;
    params->repeat = repeat_last_n;
    params->batch = n_batch;
    params->n_keep = n_keep;
    params->top_p = top_p;
    params->temp = temp;
    params->repeat_penalty = repeat_penalty;
    params->typical_p = typical_p;
    params->min_p = min_p;
    params->top_n_sigma = top_n_sigma;
    params->xtc_probability = xtc_probability;
    params->xtc_threshold = xtc_threshold;
    params->dynatemp_range = dynatemp_range;
    params->dynatemp_exponent = dynatemp_exponent;
    params->adaptive_p_target = adaptive_p_target;
    params->adaptive_p_decay = adaptive_p_decay;
    params->dry_multiplier = dry_multiplier;
    params->dry_base = dry_base;
    params->dry_allowed_length = dry_allowed_length;
    params->dry_penalty_last_n = dry_penalty_last_n;
    params->frequency_penalty = frequency_penalty;
    params->presence_penalty = presence_penalty;
    params->mirostat = mirostat;
    params->mirostat_eta = mirostat_eta;
    params->mirostat_tau = mirostat_tau;
    params->ignore_eos = ignore_eos;
    params->penalize_nl = penalize_nl;
    params->session_file = session_file != nullptr ? session_file : "";
    params->prompt_cache_all = prompt_cache_all;
    params->prompt_cache_ro = prompt_cache_ro;
    params->grammar = grammar != nullptr ? grammar : "";
    params->logit_bias = parse_logit_biases(logit_bias);

    if (dry_sequence_breakers != nullptr && dry_sequence_breakers[0] != '\0') {
        params->dry_sequence_breakers = split_string(dry_sequence_breakers, '\x1f');
    }

    for (int i = 0; i < antiprompt_count; i++) {
        if (antiprompt != nullptr && antiprompt[i] != nullptr) {
            params->antiprompt.push_back(antiprompt[i]);
        }
    }

    return params;
}

void llama_free_params(void* params_ptr) {
    delete static_cast<llama_binding_params *>(params_ptr);
}

void llama_binding_free_model(void* state_pr) {
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (state == nullptr) {
        return;
    }

    if (state->ctx != nullptr) {
        llama_free(state->ctx);
    }

    for (auto * adapter : state->adapters) {
        llama_adapter_lora_free(adapter);
    }

    if (state->model != nullptr) {
        llama_model_free(state->model);
    }

    delete state;
}

int llama_embedding_size(void* state_pr) {
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (state == nullptr || state->model == nullptr) {
        return 0;
    }
    return llama_model_n_embd_out(state->model);
}

int get_embeddings(void* params_ptr, void* state_pr, float * res_embeddings, char** error_out) {
    llama_binding_params * params = static_cast<llama_binding_params *>(params_ptr);
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (params == nullptr || state == nullptr) {
        set_binding_error(error_out, "embeddings called without a loaded model");
        return 1;
    }

    std::vector<llama_token> tokens = tokenize_text(state->vocab, params->prompt, true);
    return embedding_for_tokens(state, tokens, params, res_embeddings, error_out);
}

int get_token_embeddings(void* params_ptr, void* state_pr, int *tokens, int tokenSize, float * res_embeddings, char** error_out) {
    llama_binding_params * params = static_cast<llama_binding_params *>(params_ptr);
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (params == nullptr || state == nullptr || tokenSize < 0) {
        set_binding_error(error_out, "token embeddings called without a loaded model or with a negative token count");
        return 1;
    }

    std::vector<llama_token> token_vec;
    token_vec.reserve(static_cast<size_t>(tokenSize));
    for (int i = 0; i < tokenSize; i++) {
        token_vec.push_back(static_cast<llama_token>(tokens[i]));
    }

    return embedding_for_tokens(state, token_vec, params, res_embeddings, error_out);
}

int eval(void* params_ptr, void* state_pr, char *text, char** error_out) {
    llama_binding_params * params = static_cast<llama_binding_params *>(params_ptr);
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (params == nullptr || state == nullptr || state->ctx == nullptr) {
        set_binding_error(error_out, "eval called without a loaded model");
        return 1;
    }

    std::string input = text != nullptr ? text : "";
    std::vector<llama_token> tokens = tokenize_text(state->vocab, input, true);
    if (tokens.empty()) {
        tokens.push_back(llama_vocab_bos(state->vocab));
    }

    llama_memory_clear(llama_get_memory(state->ctx), true);
    llama_set_n_threads(state->ctx, params->threads, params->threads);
    int ret = decode_tokens(state->ctx, tokens.data(), static_cast<int>(tokens.size()), params->batch > 0 ? params->batch : static_cast<int>(llama_n_batch(state->ctx)));
    if (ret != 0) {
        set_binding_error(error_out, "eval decode failed with code " + std::to_string(ret));
    }
    return ret;
}

int llama_predict(void* params_ptr, void* state_pr, char** result, bool debug, char** error_out) {
    llama_binding_params * params = static_cast<llama_binding_params *>(params_ptr);
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (params == nullptr || state == nullptr || state->ctx == nullptr || result == nullptr) {
        set_binding_error(error_out, "prediction called without a loaded model");
        return 1;
    }

    std::vector<llama_token> prompt_tokens = tokenize_text(state->vocab, params->prompt, true);
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(llama_vocab_bos(state->vocab));
    }

    const int n_ctx = static_cast<int>(llama_n_ctx(state->ctx));
    if (static_cast<int>(prompt_tokens.size()) >= n_ctx) {
        set_binding_error(error_out, "prompt is too long (" + std::to_string(prompt_tokens.size()) + " tokens, context " + std::to_string(n_ctx) + ")");
        return 1;
    }

    llama_memory_t memory = llama_get_memory(state->ctx);

    // Prompt cache: reuse the longest token prefix shared between the saved
    // session and this prompt. An exact match reuses the restored logits and
    // skips decoding entirely; otherwise the mismatched KV tail is dropped and
    // only the remaining prompt suffix is decoded.
    bool exact_cache = false;
    size_t n_reused = 0;
    if (!params->session_file.empty() && file_exists(params->session_file)) {
        std::vector<llama_token> cache_tokens(static_cast<size_t>(n_ctx));
        size_t cache_count = 0;
        if (llama_state_load_file(state->ctx, params->session_file.c_str(), cache_tokens.data(), cache_tokens.size(), &cache_count)) {
            cache_tokens.resize(cache_count);
            while (n_reused < cache_tokens.size() && n_reused < prompt_tokens.size() &&
                   cache_tokens[n_reused] == prompt_tokens[n_reused]) {
                n_reused++;
            }
            exact_cache = n_reused == prompt_tokens.size() && cache_tokens.size() == prompt_tokens.size();
            if (!exact_cache && n_reused >= prompt_tokens.size()) {
                // The cache covers the whole prompt plus stale tokens; re-decode the
                // final prompt token so the logits match the prompt, not the tail.
                n_reused = prompt_tokens.size() - 1;
            }
            if (debug) {
                fprintf(stderr, "%s: loaded prompt cache %s (%zu tokens, reusing %zu, exact=%s)\n",
                        __func__,
                        params->session_file.c_str(),
                        cache_tokens.size(),
                        n_reused,
                        exact_cache ? "true" : "false");
            }
        } else if (debug) {
            fprintf(stderr, "%s: failed to load prompt cache %s\n", __func__, params->session_file.c_str());
        }
    }

    if (!exact_cache) {
        if (n_reused > 0) {
            llama_memory_seq_rm(memory, 0, static_cast<llama_pos>(n_reused), -1);
        } else {
            llama_memory_clear(memory, true);
        }
    }
    llama_set_embeddings(state->ctx, false);
    llama_set_n_threads(state->ctx, params->threads, params->threads);

    std::vector<llama_token> session_tokens = prompt_tokens;
    int ret = 0;
    if (!exact_cache) {
        int batch_size = params->batch > 0 ? params->batch : static_cast<int>(llama_n_batch(state->ctx));
        ret = decode_tokens(state->ctx, prompt_tokens.data() + n_reused, static_cast<int>(prompt_tokens.size() - n_reused), batch_size);
        if (ret != 0) {
            set_binding_error(error_out, "prompt decode failed with code " + std::to_string(ret));
            return ret;
        }

        if (!params->session_file.empty() && !params->prompt_cache_ro && !params->prompt_cache_all) {
            if (!llama_state_save_file(state->ctx, params->session_file.c_str(), session_tokens.data(), session_tokens.size()) && debug) {
                fprintf(stderr, "%s: failed to save prompt cache %s\n", __func__, params->session_file.c_str());
            }
        }
    }

    llama_sampler * sampler = create_sampler(state, params, error_out);
    if (sampler == nullptr) {
        return 1;
    }

    std::string output;
    int n_past = static_cast<int>(prompt_tokens.size());
    bool context_shifted = false;
    int n_predict = params->tokens <= 0 ? 128 : params->tokens;
    for (int i = 0; i < n_predict; i++) {
        llama_token token = llama_sampler_sample(sampler, state->ctx, -1);
        if (llama_vocab_is_eog(state->vocab, token)) {
            break;
        }

        std::string piece = token_to_piece(state->vocab, token);
        const size_t appended_from = output.size();
        output += piece;

        if (!tokenCallback(state, const_cast<char *>(piece.c_str()))) {
            break;
        }

        const size_t stop_pos = find_stop_position(output, appended_from, params->antiprompt);
        if (stop_pos != std::string::npos) {
            output.resize(stop_pos);
            break;
        }

        // Context shift: when the context fills up, discard half of the tokens
        // after n_keep and slide the rest back so generation can continue.
        if (n_past + 1 >= n_ctx) {
            const int n_keep = std::max(0, std::min(params->n_keep, n_past - 1));
            const int n_discard = (n_past - n_keep) / 2;
            if (n_discard <= 0 || !llama_memory_can_shift(memory)) {
                break;
            }
            llama_memory_seq_rm(memory, 0, n_keep, n_keep + n_discard);
            llama_memory_seq_add(memory, 0, n_keep + n_discard, n_past, -n_discard);
            n_past -= n_discard;
            context_shifted = true;
        }

        llama_batch batch = llama_batch_get_one(&token, 1);
        ret = llama_decode(state->ctx, batch);
        if (ret != 0) {
            set_binding_error(error_out, "token decode failed with code " + std::to_string(ret));
            llama_sampler_free(sampler);
            return ret;
        }
        n_past++;
        session_tokens.push_back(token);
    }

    llama_sampler_free(sampler);
    // Skip the save after a context shift: the KV positions no longer match a
    // linear token list, so a saved session would corrupt later cache loads.
    if (!params->session_file.empty() && !params->prompt_cache_ro && params->prompt_cache_all && !context_shifted) {
        if (!llama_state_save_file(state->ctx, params->session_file.c_str(), session_tokens.data(), session_tokens.size()) && debug) {
            fprintf(stderr, "%s: failed to save final prompt cache %s\n", __func__, params->session_file.c_str());
        }
    }
    if (debug) {
        llama_perf_context_print(state->ctx);
        llama_perf_context_reset(state->ctx);
    }
    return copy_result_string(output, result);
}

int speculative_sampling(void* params_ptr, void* target_model, void* draft_model, char** result, bool debug, char** error_out) {
    (void) draft_model;
    fprintf(stderr, "%s: warning: speculative decoding is not implemented in this binding; the draft model is unused and generation falls back to standard prediction\n", __func__);
    return llama_predict(params_ptr, target_model, result, debug, error_out);
}

int llama_tokenize_string(void* params_ptr, void* state_pr, int* result) {
    llama_binding_params * params = static_cast<llama_binding_params *>(params_ptr);
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (params == nullptr || state == nullptr || result == nullptr) {
        return -1;
    }

    int needed = -llama_tokenize(state->vocab, params->prompt.c_str(), static_cast<int32_t>(params->prompt.size()), nullptr, 0, true, true);
    if (needed <= 0) {
        return 0;
    }
    if (params->tokens > 0 && needed > params->tokens) {
        return -needed;
    }

    std::vector<llama_token> tokens(static_cast<size_t>(needed));
    int actual = llama_tokenize(state->vocab, params->prompt.c_str(), static_cast<int32_t>(params->prompt.size()), tokens.data(), needed, true, true);
    if (actual < 0) {
        return actual;
    }
    for (int i = 0; i < actual; i++) {
        result[i] = tokens[static_cast<size_t>(i)];
    }
    return actual;
}

int llama_detokenize_tokens(void* state_pr, const int* tokens, int token_count, bool remove_special, bool unparse_special, char** result) {
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (state == nullptr || state->vocab == nullptr || result == nullptr || token_count < 0) {
        return 1;
    }
    if (token_count == 0) {
        return copy_result_string("", result);
    }

    std::vector<llama_token> token_vec(static_cast<size_t>(token_count));
    for (int i = 0; i < token_count; i++) {
        token_vec[static_cast<size_t>(i)] = static_cast<llama_token>(tokens[i]);
    }

    int needed = -llama_detokenize(state->vocab, token_vec.data(), token_count, nullptr, 0, remove_special, unparse_special);
    if (needed < 0) {
        return 1;
    }

    std::vector<char> buffer(static_cast<size_t>(needed) + 1);
    int actual = llama_detokenize(state->vocab, token_vec.data(), token_count, buffer.data(), static_cast<int32_t>(buffer.size()), remove_special, unparse_special);
    if (actual < 0) {
        return 1;
    }
    return copy_result_string(std::string(buffer.data(), static_cast<size_t>(actual)), result);
}

int save_state(void *state_pr, char *dst, char *modes) {
    (void) modes;
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (state == nullptr || state->ctx == nullptr || dst == nullptr) {
        return 1;
    }

    const size_t size = llama_state_get_size(state->ctx);
    std::vector<uint8_t> data(size);
    size_t written = llama_state_get_data(state->ctx, data.data(), data.size());
    if (written != size) {
        return 1;
    }

    std::ofstream file(dst, std::ios::binary);
    if (!file) {
        return 1;
    }
    file.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    return file.good() ? 0 : 1;
}

int load_state(void *state_pr, char *statefile, char *modes) {
    (void) modes;
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (state == nullptr || state->ctx == nullptr || statefile == nullptr) {
        return 1;
    }

    std::ifstream file(statefile, std::ios::binary | std::ios::ate);
    if (!file) {
        return 1;
    }

    std::streamsize size = file.tellg();
    if (size < 0) {
        return 1;
    }
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
        return 1;
    }

    size_t read = llama_state_set_data(state->ctx, data.data(), data.size());
    return read == data.size() ? 0 : 1;
}

int llama_apply_chat_template(void* state_pr, const char** roles, const char** contents, int count, bool add_generation_prompt, char** result) {
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (state == nullptr || state->model == nullptr || result == nullptr || count < 0) {
        return 1;
    }

    std::vector<llama_chat_message> messages;
    messages.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; i++) {
        messages.push_back(llama_chat_message{roles[i], contents[i]});
    }

    const char * tmpl = llama_model_chat_template(state->model, nullptr);
    int needed = llama_chat_apply_template(tmpl, messages.data(), messages.size(), add_generation_prompt, nullptr, 0);
    if (needed < 0) {
        return 1;
    }

    std::vector<char> buffer(static_cast<size_t>(needed) + 1);
    int actual = llama_chat_apply_template(tmpl, messages.data(), messages.size(), add_generation_prompt, buffer.data(), buffer.size());
    if (actual < 0) {
        return 1;
    }
    if (actual > static_cast<int>(buffer.size())) {
        buffer.resize(static_cast<size_t>(actual) + 1);
        actual = llama_chat_apply_template(tmpl, messages.data(), messages.size(), add_generation_prompt, buffer.data(), buffer.size());
        if (actual < 0) {
            return 1;
        }
    }

    return copy_result_string(std::string(buffer.data(), static_cast<size_t>(actual)), result);
}

int llama_model_info_json(void* state_pr, char** result) {
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (state == nullptr || state->model == nullptr || state->vocab == nullptr || result == nullptr) {
        return 1;
    }

    llama_model * model = state->model;
    const llama_vocab * vocab = state->vocab;

    char desc[1024];
    int desc_len = llama_model_desc(model, desc, sizeof(desc));
    std::string description = desc_len > 0 ? std::string(desc, static_cast<size_t>(std::min<int>(desc_len, sizeof(desc) - 1))) : "";

    enum llama_ftype ftype = llama_model_ftype(model);
    const char * ftype_name = llama_ftype_name(ftype);
    const char * chat_template = llama_model_chat_template(model, nullptr);

    std::string json = "{";
    append_json_string_field(json, "description", description, false);
    json += ",\"size\":" + std::to_string(llama_model_size(model));
    json += ",\"parameters\":" + std::to_string(llama_model_n_params(model));
    json += ",\"ftype\":" + std::to_string(static_cast<int>(ftype));
    append_json_string_field(json, "ftype_name", ftype_name != nullptr ? ftype_name : "");
    json += ",\"context_train\":" + std::to_string(llama_model_n_ctx_train(model));
    json += ",\"embedding_length\":" + std::to_string(llama_model_n_embd(model));
    json += ",\"embedding_input_length\":" + std::to_string(llama_model_n_embd_inp(model));
    json += ",\"embedding_output_length\":" + std::to_string(llama_model_n_embd_out(model));
    json += ",\"layers\":" + std::to_string(llama_model_n_layer(model));
    json += ",\"heads\":" + std::to_string(llama_model_n_head(model));
    json += ",\"heads_kv\":" + std::to_string(llama_model_n_head_kv(model));
    json += ",\"vocab_size\":" + std::to_string(llama_vocab_n_tokens(vocab));
    json += ",\"rope_type\":" + std::to_string(static_cast<int>(llama_model_rope_type(model)));
    json += ",\"has_encoder\":" + std::string(llama_model_has_encoder(model) ? "true" : "false");
    json += ",\"has_decoder\":" + std::string(llama_model_has_decoder(model) ? "true" : "false");
    json += ",\"is_recurrent\":" + std::string(llama_model_is_recurrent(model) ? "true" : "false");
    json += ",\"is_hybrid\":" + std::string(llama_model_is_hybrid(model) ? "true" : "false");
    json += ",\"is_diffusion\":" + std::string(llama_model_is_diffusion(model) ? "true" : "false");
    append_json_string_field(json, "chat_template", chat_template != nullptr ? chat_template : "");
    json += ",\"metadata\":{";

    const int meta_count = llama_model_meta_count(model);
    bool first_meta = true;
    for (int i = 0; i < meta_count; i++) {
        char key_buf[512];
        char val_buf[4096];
        int key_len = llama_model_meta_key_by_index(model, i, key_buf, sizeof(key_buf));
        int val_len = llama_model_meta_val_str_by_index(model, i, val_buf, sizeof(val_buf));
        if (key_len < 0 || val_len < 0) {
            continue;
        }

        std::string key(key_buf, static_cast<size_t>(std::min<int>(key_len, sizeof(key_buf) - 1)));
        std::string val(val_buf, static_cast<size_t>(std::min<int>(val_len, sizeof(val_buf) - 1)));
        if (!first_meta) {
            json += ",";
        }
        first_meta = false;
        json += "\"";
        json += json_escape(key);
        json += "\":\"";
        json += json_escape(val);
        json += "\"";
    }

    json += "}}";
    return copy_result_string(json, result);
}

int llama_builtin_chat_templates_json(char** result) {
    if (result == nullptr) {
        return 1;
    }

    int count = llama_chat_builtin_templates(nullptr, 0);
    if (count < 0) {
        return 1;
    }
    if (count == 0) {
        return copy_result_string("[]", result);
    }

    std::vector<const char *> templates(static_cast<size_t>(count));
    int actual = llama_chat_builtin_templates(templates.data(), templates.size());
    if (actual < 0) {
        return 1;
    }

    std::string json = "[";
    for (int i = 0; i < actual; i++) {
        if (i > 0) {
            json += ",";
        }
        json += "\"";
        json += json_escape(templates[static_cast<size_t>(i)] != nullptr ? templates[static_cast<size_t>(i)] : "");
        json += "\"";
    }
    json += "]";
    return copy_result_string(json, result);
}
