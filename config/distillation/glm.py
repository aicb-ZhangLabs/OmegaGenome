from dataclasses import replace

from src.model.glm import GLMConfig


dna_bert_v2 = GLMConfig(
    model_name_or_path="zhihan1996/DNABERT-2-117M",
    num_labels=2,
)

# NT teacher configuration
nt_2b5 = GLMConfig(
    model_name_or_path="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
    num_labels=2,
    trust_remote_code=True,
    output_hidden_states=True,
    is_lora=True,
    base_model_path="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
)

# Caduceus teacher configuration
caduceus = GLMConfig(
    model_name_or_path="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16",
    num_labels=2,
    trust_remote_code=True,
    output_hidden_states=True,
)


# Enformer teacher configuration
enformer = GLMConfig(
    model_name_or_path="EleutherAI/enformer-official-rough",
    num_labels=2,
    trust_remote_code=True,
    output_hidden_states=True,
)

# Carbon teacher configs (HuggingFaceBio, autoregressive, Apache-2.0). HF-native; needs the
# "<dna>" prefix and add_special_tokens=False for its hybrid 6-mer tokenizer.
carbon_3b = GLMConfig(
    model_name_or_path="HuggingFaceBio/Carbon-3B",
    num_labels=2,
    trust_remote_code=True,
    output_hidden_states=True,
    input_prefix="<dna>",
    add_special_tokens=False,
    torch_dtype="bfloat16",  # 3B teacher: bf16 to fit memory + matches its LoRA fine-tune dtype
)
carbon_8b = replace(carbon_3b, model_name_or_path="HuggingFaceBio/Carbon-8B")
carbon_500m = replace(carbon_3b, model_name_or_path="HuggingFaceBio/Carbon-500M")

# Carbon-3B LoRA *teacher* config for distillation: loads the base model + a per-task LoRA adapter
# (the consolidated teachers). merge_lora=True for fast precompute; base_model_path explicit so
# loading doesn't rely on adapter_config auto-detection.
carbon_3b_lora = replace(
    carbon_3b,
    is_lora=True,
    base_model_path="HuggingFaceBio/Carbon-3B",
    merge_lora=True,
)
