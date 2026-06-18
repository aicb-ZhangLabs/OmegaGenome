from src.model.bpnet_classifier import BPNetClassifierConfig


bpnet_classifier_config = BPNetClassifierConfig(
    num_labels=2,  # this will be set to the number based on the task
    teacher_hidden_size=None,  # this will be set to the hidden size of the teacher model
)

original_bpnet_classifier_config = BPNetClassifierConfig(
    num_labels=2,
    model_type="bpnet",
    model_size="original",  # Default to original
)

# ~0.12M-deployment-param BPNet student (the Carbon distillation target). num_labels +
# teacher_hidden_size are overridden per-task by distill.py; teacher_projection_opt="down"
# projects the teacher hidden -> student channel dim for the MSE feature-matching term.
deploy_120k_bpnet_config = BPNetClassifierConfig(
    num_labels=2,
    model_type="bpnet",
    model_size="deploy_120k",
    teacher_projection_opt="down",
)
