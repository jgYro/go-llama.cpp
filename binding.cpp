#include "binding.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
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
    float frequency_penalty;
    float presence_penalty;
    int mirostat;
    float mirostat_eta;
    float mirostat_tau;
    bool ignore_eos;
    bool penalize_nl;
    std::string grammar;
    std::vector<std::string> antiprompt;
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

static bool ends_with_any(const std::string & text, const std::vector<std::string> & suffixes) {
    for (const auto & suffix : suffixes) {
        if (!suffix.empty() && text.size() >= suffix.size() &&
            text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return true;
        }
    }
    return false;
}

static std::vector<float> parse_tensor_split(const char * tensor_split) {
    std::vector<float> result;
    if (tensor_split == nullptr || tensor_split[0] == '\0') {
        return result;
    }

    std::stringstream ss(tensor_split);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            result.push_back(std::stof(item));
        }
    }
    return result;
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

static int decode_tokens(llama_context * ctx, std::vector<llama_token> & tokens, int batch_size) {
    if (tokens.empty()) {
        return 0;
    }

    int offset = 0;
    const int limit = std::max(1, batch_size);
    while (offset < static_cast<int>(tokens.size())) {
        const int n = std::min(limit, static_cast<int>(tokens.size()) - offset);
        llama_batch batch = llama_batch_get_one(tokens.data() + offset, n);
        int ret = llama_decode(ctx, batch);
        if (ret != 0) {
            return ret;
        }
        offset += n;
    }
    return 0;
}

static llama_sampler * create_sampler(llama_binding_state * state, const llama_binding_params * params) {
    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler * sampler = llama_sampler_chain_init(chain_params);

    const int n_vocab = llama_vocab_n_tokens(state->vocab);

    std::vector<llama_logit_bias> biases;
    if (params->ignore_eos) {
        biases.push_back(llama_logit_bias{llama_vocab_eos(state->vocab), -INFINITY});
    }
    if (!biases.empty()) {
        llama_sampler_chain_add(sampler, llama_sampler_init_logit_bias(n_vocab, static_cast<int32_t>(biases.size()), biases.data()));
    }

    if (!params->grammar.empty()) {
        llama_sampler * grammar = llama_sampler_init_grammar(state->vocab, params->grammar.c_str(), "root");
        if (grammar != nullptr) {
            llama_sampler_chain_add(sampler, grammar);
        }
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

    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params->top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params->top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_typical(params->typical_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(params->repeat, params->repeat_penalty, params->frequency_penalty, params->presence_penalty));

    if (params->temp <= 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params->temp));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(static_cast<uint32_t>(params->seed)));
    }

    return sampler;
}

static int embedding_for_tokens(llama_binding_state * state, std::vector<llama_token> & tokens, llama_binding_params * params, float * res_embeddings) {
    if (state == nullptr || state->ctx == nullptr || state->model == nullptr || res_embeddings == nullptr) {
        return 1;
    }

    if (tokens.empty()) {
        tokens.push_back(llama_vocab_bos(state->vocab));
    }

    llama_memory_clear(llama_get_memory(state->ctx), true);
    llama_set_embeddings(state->ctx, true);
    llama_set_n_threads(state->ctx, params->threads, params->threads);

    int ret = decode_tokens(state->ctx, tokens, params->batch > 0 ? params->batch : static_cast<int>(llama_n_batch(state->ctx)));
    if (ret != 0) {
        return ret;
    }

    const int n_embd = llama_model_n_embd(state->model);
    float * embeddings = llama_get_embeddings_ith(state->ctx, -1);
    if (embeddings == nullptr) {
        embeddings = llama_get_embeddings_seq(state->ctx, 0);
    }
    if (embeddings == nullptr) {
        embeddings = llama_get_embeddings(state->ctx);
    }
    if (embeddings == nullptr) {
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
                 bool mul_mat_q,
                 const char *lora,
                 const char *lora_base,
                 bool perplexity) {
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

    std::vector<float> tensor_split = parse_tensor_split(tensorsplit);
    if (!tensor_split.empty()) {
        model_params.tensor_split = tensor_split.data();
    }

    if (maingpu != nullptr && maingpu[0] != '\0') {
        model_params.main_gpu = std::stoi(maingpu);
    }

    llama_model * model = llama_model_load_from_file(fname, model_params);
    if (model == nullptr) {
        fprintf(stderr, "%s: error: unable to load model %s\n", __func__, fname);
        return nullptr;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx > 0 ? static_cast<uint32_t>(n_ctx) : 0;
    ctx_params.n_batch = n_batch > 0 ? static_cast<uint32_t>(n_batch) : 512;
    ctx_params.n_ubatch = ctx_params.n_batch;
    ctx_params.embeddings = embeddings;
    ctx_params.no_perf = false;
    ctx_params.rope_freq_base = rope_freq_base;
    ctx_params.rope_freq_scale = rope_freq_scale;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to create context for %s\n", __func__, fname);
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
            fprintf(stderr, "%s: error: failed to load LoRA adapter %s\n", __func__, lora);
            llama_binding_free_model(state);
            return nullptr;
        }
        state->adapters.push_back(adapter);
        std::vector<float> scales(state->adapters.size(), 1.0f);
        if (llama_set_adapters_lora(ctx, state->adapters.data(), state->adapters.size(), scales.data()) != 0) {
            fprintf(stderr, "%s: error: failed to apply LoRA adapter %s\n", __func__, lora);
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
                            float tfs_z, float typical_p, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *logit_bias, const char *session_file, bool prompt_cache_all, bool mlock, bool mmap, const char *maingpu, const char *tensorsplit,
                            bool prompt_cache_ro, const char *grammar, float rope_freq_base, float rope_freq_scale, float negative_prompt_scale, const char* negative_prompt,
                            int n_draft) {
    (void) memory_f16;
    (void) tfs_z;
    (void) logit_bias;
    (void) session_file;
    (void) prompt_cache_all;
    (void) mlock;
    (void) mmap;
    (void) maingpu;
    (void) tensorsplit;
    (void) prompt_cache_ro;
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
    params->frequency_penalty = frequency_penalty;
    params->presence_penalty = presence_penalty;
    params->mirostat = mirostat;
    params->mirostat_eta = mirostat_eta;
    params->mirostat_tau = mirostat_tau;
    params->ignore_eos = ignore_eos;
    params->penalize_nl = penalize_nl;
    params->grammar = grammar != nullptr ? grammar : "";

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
    return llama_model_n_embd(state->model);
}

int get_embeddings(void* params_ptr, void* state_pr, float * res_embeddings) {
    llama_binding_params * params = static_cast<llama_binding_params *>(params_ptr);
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (params == nullptr || state == nullptr) {
        return 1;
    }

    std::vector<llama_token> tokens = tokenize_text(state->vocab, params->prompt, true);
    return embedding_for_tokens(state, tokens, params, res_embeddings);
}

int get_token_embeddings(void* params_ptr, void* state_pr, int *tokens, int tokenSize, float * res_embeddings) {
    llama_binding_params * params = static_cast<llama_binding_params *>(params_ptr);
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (params == nullptr || state == nullptr || tokenSize < 0) {
        return 1;
    }

    std::vector<llama_token> token_vec;
    token_vec.reserve(static_cast<size_t>(tokenSize));
    for (int i = 0; i < tokenSize; i++) {
        token_vec.push_back(static_cast<llama_token>(tokens[i]));
    }

    return embedding_for_tokens(state, token_vec, params, res_embeddings);
}

int eval(void* params_ptr, void* state_pr, char *text) {
    llama_binding_params * params = static_cast<llama_binding_params *>(params_ptr);
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (params == nullptr || state == nullptr || state->ctx == nullptr) {
        return 1;
    }

    std::string input = text != nullptr ? text : "";
    std::vector<llama_token> tokens = tokenize_text(state->vocab, input, true);
    if (tokens.empty()) {
        tokens.push_back(llama_vocab_bos(state->vocab));
    }

    llama_memory_clear(llama_get_memory(state->ctx), true);
    llama_set_n_threads(state->ctx, params->threads, params->threads);
    return decode_tokens(state->ctx, tokens, params->batch > 0 ? params->batch : static_cast<int>(llama_n_batch(state->ctx)));
}

int llama_predict(void* params_ptr, void* state_pr, char** result, bool debug) {
    (void) debug;
    llama_binding_params * params = static_cast<llama_binding_params *>(params_ptr);
    llama_binding_state * state = static_cast<llama_binding_state *>(state_pr);
    if (params == nullptr || state == nullptr || state->ctx == nullptr || result == nullptr) {
        return 1;
    }

    std::vector<llama_token> prompt_tokens = tokenize_text(state->vocab, params->prompt, true);
    if (prompt_tokens.empty()) {
        prompt_tokens.push_back(llama_vocab_bos(state->vocab));
    }

    if (prompt_tokens.size() >= llama_n_ctx(state->ctx)) {
        fprintf(stderr, "%s: prompt is too long (%zu tokens, context %u)\n", __func__, prompt_tokens.size(), llama_n_ctx(state->ctx));
        return 1;
    }

    llama_memory_clear(llama_get_memory(state->ctx), true);
    llama_set_embeddings(state->ctx, false);
    llama_set_n_threads(state->ctx, params->threads, params->threads);

    int batch_size = params->batch > 0 ? params->batch : static_cast<int>(llama_n_batch(state->ctx));
    int ret = decode_tokens(state->ctx, prompt_tokens, batch_size);
    if (ret != 0) {
        fprintf(stderr, "%s: prompt decode failed: %d\n", __func__, ret);
        return ret;
    }

    llama_sampler * sampler = create_sampler(state, params);
    if (sampler == nullptr) {
        return 1;
    }

    std::string output;
    int n_predict = params->tokens <= 0 ? 128 : params->tokens;
    for (int i = 0; i < n_predict; i++) {
        llama_token token = llama_sampler_sample(sampler, state->ctx, -1);
        if (llama_vocab_is_eog(state->vocab, token)) {
            break;
        }

        std::string piece = token_to_piece(state->vocab, token);
        output += piece;

        if (!tokenCallback(state, const_cast<char *>(piece.c_str()))) {
            break;
        }

        if (ends_with_any(output, params->antiprompt)) {
            break;
        }

        llama_batch batch = llama_batch_get_one(&token, 1);
        ret = llama_decode(state->ctx, batch);
        if (ret != 0) {
            fprintf(stderr, "%s: token decode failed: %d\n", __func__, ret);
            llama_sampler_free(sampler);
            return ret;
        }
    }

    llama_sampler_free(sampler);
    return copy_result_string(output, result);
}

int speculative_sampling(void* params_ptr, void* target_model, void* draft_model, char** result, bool debug) {
    (void) draft_model;
    return llama_predict(params_ptr, target_model, result, debug);
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
