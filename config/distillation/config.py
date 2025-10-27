"""
Main configuration file that imports all experiment configs
"""

from .experiments import dna_bert_v2
from .experiments import dna_bert_v2_hyperparam
from .experiments import nt
from .experiments import nt_hyperparam
from .experiments import nt_different_size
from .experiments import caduceus
from .experiments import caduceus_hyperparam

# Combine all configs
configs = {
    **dna_bert_v2.experiment_configs,
    **nt.experiment_configs,
    **nt_different_size.experiment_configs,
    **caduceus.experiment_configs,
}

# Hyperparameter search configs
hyperparam_configs = {
    **dna_bert_v2_hyperparam.experiment_configs,
    **nt_hyperparam.experiment_configs,
    **caduceus_hyperparam.experiment_configs,
}
