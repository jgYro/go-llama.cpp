#ifdef __cplusplus
#include <vector>
#include <string>
extern "C" {
#endif

#include <stdbool.h>

extern unsigned char tokenCallback(void *, char *);

int load_state(void *state, char *statefile, char*modes);

int eval(void* params_ptr, void *ctx, char*text, char** error_out);

int save_state(void *state, char *dst, char*modes);

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
                 bool mul_mat_q, const char *lora, const char *lora_base, bool perplexity,
                 char **error_out
                 );

int get_embeddings(void* params_ptr, void* state_pr, float * res_embeddings, char** error_out);

int get_token_embeddings(void* params_ptr, void* state_pr,  int *tokens, int tokenSize, float * res_embeddings, char** error_out);

int llama_embedding_size(void* state_pr);

void* llama_allocate_params(const char *prompt, int seed, int threads, int tokens,
                            int top_k, float top_p, float temp, float repeat_penalty, 
                            int repeat_last_n, bool ignore_eos, bool memory_f16, 
                            int n_batch, int n_keep, const char** antiprompt, int antiprompt_count,
                            float tfs_z, float typical_p, float min_p, float top_n_sigma, float xtc_probability, float xtc_threshold, float dynatemp_range, float dynatemp_exponent, float adaptive_p_target, float adaptive_p_decay, float dry_multiplier, float dry_base, int dry_allowed_length, int dry_penalty_last_n, const char *dry_sequence_breakers, float frequency_penalty, float presence_penalty, int mirostat, float mirostat_eta, float mirostat_tau, bool penalize_nl, const char *logit_bias, const char *session_file, bool prompt_cache_all, bool mlock, bool mmap, const char *maingpu, const char *tensorsplit,
                            bool prompt_cache_ro, const char *grammar, float rope_freq_base, float rope_freq_scale, float negative_prompt_scale, const char* negative_prompt,
                            int n_draft);

int speculative_sampling(void* params_ptr, void* target_model, void* draft_model, char** result, bool debug, char** error_out);

const char* llama_binding_media_marker(void);

int llama_binding_load_mmproj(void* state, const char* path, bool use_gpu, int threads, char** error_out);

int llama_binding_supports_vision(void* state);

int llama_binding_supports_audio(void* state);

int llama_predict_mtmd(void* params_ptr, void* state_pr, const char** media_paths, int media_count, char** result, bool debug, char** error_out);

void llama_free_params(void* params_ptr);

void llama_binding_free_model(void* state);

int llama_tokenize_string(void* params_ptr, void* state_pr, int* result);

int llama_detokenize_tokens(void* state_pr, const int* tokens, int token_count, bool remove_special, bool unparse_special, char** result);

int llama_predict(void* params_ptr, void* state_pr, char** result, bool debug, char** error_out);

int llama_apply_chat_template(void* state_pr, const char** roles, const char** contents, int count, bool add_generation_prompt, char** result);

int llama_model_info_json(void* state_pr, char** result);

int llama_builtin_chat_templates_json(char** result);

#ifdef __cplusplus
}
#endif
