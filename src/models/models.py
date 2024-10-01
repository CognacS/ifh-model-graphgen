from typing import Dict, Tuple, Optional

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.models.reinsertion.regressors import GNNRegressor
from src.models.reinsertion.empirical import EmpiricalSampler
from src.models.denoising.wrappers import ConditionalGraphTransformer

KEY_REINSERTION = 'reinsertion'
KEY_HALTING = 'halting'
KEY_DENOISING = 'denoising'

KEY_MODEL_NAME = 'name'
KEY_MODEL_PARAMS = 'params'
KEY_REQUESTED_DATASET_FIELDS = 'requested_dataset_fields'

REGISTERED_MODELS = {
    KEY_HALTING: {
        'gnn_regressor': GNNRegressor
    },
    KEY_REINSERTION: {
        'gnn_regressor': GNNRegressor,
        'empirical_distribution': EmpiricalSampler
    },
    KEY_DENOISING: {
        'graph_transformer': ConditionalGraphTransformer
    }
}


def is_model_name(
        which_type: str = KEY_REINSERTION
    ) -> bool:
    """Check if a model is registered.
    """
    return which_type in REGISTERED_MODELS


def get_model_from_config(
        config: Dict,
        which_type: str = KEY_REINSERTION,
        dataset_info: Optional[Dict] = None,
        dataset_requested_fields: Optional[Dict] = None,
        **kwargs
    ) -> nn.Module:

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    # get model and params
    model_name = config[KEY_MODEL_NAME]
    if KEY_MODEL_PARAMS in config:
        model_params = config[KEY_MODEL_PARAMS]
    else:
        model_params = {}

    # get dataset information if needed
    if KEY_REQUESTED_DATASET_FIELDS in config and dataset_info is not None:
        dataset_requested_fields = config[KEY_REQUESTED_DATASET_FIELDS]
        requested_dataset_info = {k: dataset_info[v] for k, v in dataset_requested_fields.items()}
    else:
        requested_dataset_info = {}

    model = REGISTERED_MODELS[which_type][model_name](
        **model_params,
        **requested_dataset_info,
        **kwargs
    )

    return model