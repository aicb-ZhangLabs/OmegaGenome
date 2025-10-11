from src.model.bpnet_classifier import BPNetClassifierConfig


bpnet_classifier_config = BPNetClassifierConfig(
    num_labels=2,  # this will be set to the number based on the task
    teacher_hidden_size=None,  # this will be set to the hidden size of the teacher model
)
