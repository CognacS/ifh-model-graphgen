###############################  LOSS FUNCTIONS  ###############################

# Reinsertion losses
KEY_REINSERTION_LOSS = 'reinsertion_loss'
KEY_REINSERTION_LOSS_MSE = 'reinsertion_loss_mse'
KEY_REINSERTION_LOSS_KLDIV = 'reinsertion_loss_kldiv'

# Denoising losses
KEY_DENOISING_LOSS_X_CE = 'denoising_loss_x_ce'
KEY_DENOISING_LOSS_E_CE = 'denoising_loss_e_ce'
KEY_DENOISING_LOSS_EXT_E_CE = 'denoising_loss_ext_e_ce'
KEY_DENOISING_LOSS_TOTAL_CE = 'denoising_loss_total_ce'

# Halting losses
KEY_HALTING_LOSS_BCE = 'halting_loss_bce'

# Training time performance metrics
KEY_COMPUTATIONAL_TRAIN_TIME = 'computational_train_time'
KEY_COMPUTATIONAL_TRAIN_MEMORY = 'computational_train_memory'


#######################  SUPERVISED EVALUATION METRICS  ########################
# the following metrics are computed over batch of the dataset
# and do not require generating new samples with the model

# Reinsertion metrics
KEY_REINSERTION_ACC = 'reinsertion_acc'

# Denoising metrics
KEY_DENOISING_ACC_X = 'denoising_x_acc'
KEY_DENOISING_ACC_E = 'denoising_e_acc'
KEY_DENOISING_ACC_EXT_E = 'denoising_ext_e_acc'

# Halting metrics
KEY_HALTING_ACC = 'halting_acc'
KEY_HALTING_RECALL = 'halting_recall'
KEY_HALTING_PRECISION = 'halting_precision'
KEY_HALTING_PRIOR_EMD = 'halting_prior_emd'


##############################  SAMPLING METRICS  ##############################
# the following metrics are computed by generating new samples
# with the model, which can be time consuming

# input metric name: name of the metric function <fn>
# called after sampling
# output metric names: name of the metrics resulting
# from the metric function <fn>

# Molecular
KEY_MOLECULAR_METRIC_TYPE = 'molecular'

# input metric name
KEY_MOLECULAR_VUN = 'molecular_vun'
# requires a data partition (train)
# output metric names
KEY_MOLECULAR_VALIDITY = 'molecular_validity'
KEY_MOLECULAR_RELAXED_VALIDITY = 'molecular_relaxed_validity'
KEY_MOLECULAR_UNIQUENESS = 'molecular_uniqueness'
KEY_MOLECULAR_NOVELTY = 'molecular_novelty'

# input metric name
KEY_MOLECULAR_DISTRIBUTION = 'molecular_distribution'
# requires a data partition (valid/test) and a metrics_list (all)
# output metric names
KEY_MOLECULAR_FCD = 'molecular_fcd'
KEY_MOLECULAR_NSPDK = 'molecular_nspdk' # not really molecular, but still used

# Graph
KEY_GRAPH_METRIC_TYPE = 'graph'

# input metric name
KEY_GRAPH_VUN_IN = 'graph_vun'
# requires a data partition (train) and a graph_type
# output metric names
KEY_GRAPH_UNIQUE = 'graph_uniqueness'
KEY_GRAPH_UNIQUE_NOVEL = 'graph_uniqueness_novelty'
KEY_GRAPH_VUN = 'graph_vun'

# input metric name
KEY_GRAPH_STRUCTURE = 'graph_structure'
# requires a data partition (valid/test), a metrics_list (all),
# and whether to compute the EMD kernel
# output metric names
KEY_GRAPH_DEGREE = 'graph_degree'
KEY_GRAPH_SPECTRE = 'graph_spectre'
KEY_GRAPH_CLUSTERING = 'graph_clustering'
KEY_GRAPH_ORBIT = 'graph_orbit'

# input metric name
KEY_GRAPH_CONN_COMP = 'graph_conn_comp'
# requires nothing
# output metric names
KEY_GRAPH_CONN_COMP_MEAN = 'graph_conn_comp_mean'
KEY_GRAPH_CONN_COMP_MIN = 'graph_conn_comp_min'
KEY_GRAPH_CONN_COMP_MAX = 'graph_conn_comp_max'

# input metric name
KEY_GRAPH_STRUCTURE_ALT = 'graph_structure_alt'
# requires a data partition (valid/test), a metrics_list (all),
# and an optional additional cfg dict
# output metric names
KEY_GRAPH_GIN = 'graph_gin'

# Computational
KEY_PROCESS_METRIC_TYPE = 'process'

KEY_COMPUTATIONAL_TIME = 'computational_time'
KEY_COMPUTATIONAL_MEMORY = 'computational_memory'


# output metrics are usually of the form:
# <phase>_<model>/<metric_name>
# where <phase> is one of 'train', 'valid', 'test', 'sampling'
# and <model> is one of 'reinsertion', 'denoising', 'halting'

# an input configuration is organized as follows:
# cfg = {
#     m_list.KEY_MOLECULAR_METRIC_TYPE: {
#         m_list.KEY_MOLECULAR_VUN: {
#             'data_partition': 'train',
#         },
#         m_list.KEY_MOLECULAR_DISTRIBUTION: {
#             'data_partition': 'valid',
#             'metrics_list': ['molecular_fcd', 'molecular_nspdk'],
#         }
#     }
# }
