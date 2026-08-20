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
    weight_mse=0.1,
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
dist_model_config = DistillationModelConfig(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.0,
    temperature=4.0,
    distill_method="dist",
)


nt_different_size_model_config = DistillationModelConfig(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.2,
    temperature=4.0,
    distill_method="vanilla",
)

# Carbon -> deploy_120k BPNet: the requested ce0.5/kl0.5/mse0.2 vanilla config. Two variants for the
# raw-vs-L2-norm-MSE comparison (only difference is mse_normalize; loss math otherwise identical).
carbon_vanilla_mse_raw = DistillationModelConfig(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.2,
    temperature=2.0,
    distill_method="vanilla",
    mse_normalize=False,
)
carbon_vanilla_mse_l2norm = DistillationModelConfig(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.2,
    temperature=2.0,
    distill_method="vanilla",
    mse_normalize=True,
)
