from src.model.distillation import DistillationModelConfig

distillation_model_config = DistillationModelConfig(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.1,
    temperature=2.0,
    zscore=False,
)

vanilla_distillation_model_config = DistillationModelConfig(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.0,
    temperature=2.0,
    distill_method="vanilla",
)
logits_standardization_model_config = DistillationModelConfig(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.2,
    temperature=4.0,
    distill_method="logit_standard",
)
dkd_model_config = DistillationModelConfig(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.0,  # DKD doesn't use feature matching
    temperature=4.0,
    distill_method="dkd",
    dkd_alpha=1.0,
    dkd_beta=8.0,
)
dist_model_config = (
    DistillationModelConfig(
        weight_ce=0.5,
        weight_kl=0.5,
        weight_mse=0.0,
        temperature=4.0,
        distill_method="dist",
    ),
)

nt_different_size_model_config = DistillationModelConfig(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.2,
    temperature=4.0,
    distill_method="vanilla",
)
