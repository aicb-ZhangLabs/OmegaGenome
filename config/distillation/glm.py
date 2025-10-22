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
)
