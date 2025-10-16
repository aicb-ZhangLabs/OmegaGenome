from src.model.distillation import DistillationModelConfig

distillation_model_config = DistillationModelConfig(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.1,
    temperature=2.0,
    zscore=False,
)
