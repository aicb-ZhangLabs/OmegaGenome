from .experiments import dna_bert_v2
from .experiments import dna_bert_v2_hyperparam


configs = {
    **dna_bert_v2.experiment_configs,
}


hyperparam_configs = {
    **dna_bert_v2_hyperparam.experiment_configs,
}
