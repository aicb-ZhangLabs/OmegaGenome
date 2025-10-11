from src.model.glm import GLMConfig


dna_bert2_config = GLMConfig(
    model_name_or_path="BAAI/bpt2-bert-256M",
    num_labels=2,
)
