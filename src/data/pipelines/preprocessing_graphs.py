from typing import List

import src.data.transforms as base_t
from src.data.transforms import (
    preprocessing as prep_t,
    splitting as split_t,
    conditions as cond_t,
    graphs as grph_t,
    molecular as chem_t,
    qm9 as qm9_t
)
import src.data.dataset as ds


from torch_geometric.datasets.qm9 import conversion
import numpy as np

from src.data.pipelines.download_graphs import _SPECTRE_DATASET_NAME_TO_FILENAME, _PICKLED_DATASET_NAME_TO_FILENAME


def get_len(d: List) -> int:
    return len(d)
    
def get_shape_1(y) -> int:
    return y.shape[1]

class SpectreDatasetPreprocessingPipeline:

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.filename = _SPECTRE_DATASET_NAME_TO_FILENAME[dataset_name]
    

    def __call__(
            self,
            tr_val_test_split: bool=True,
            valid_split: int|float=0.2,
            test_split: int|float=0.2,
            compute_one_hot: bool = True
        ) -> base_t.DFPipeline:


        splits, out_files = split_t.get_outputdfs_files_splits(
            split_test = tr_val_test_split,
            split_valid = tr_val_test_split and valid_split is not None,
            outdf_file_ids_exts=[
                (ds.KEY_DATA_CONTAINER, 'data', 'pt'),
                (ds.KEY_DATA_INFO, 'info', 'json'),
            ],
            file_format_string= self.dataset_name + '{split}_{file_id}.{ext}'
        )
        

        ############################################################################
        #                              PIPELINE START                              #
        ############################################################################
        pipeline = base_t.DFPipeline(

            input_files = {
                'data_file': self.filename, # pipeline input parameter
            },

            output_files = out_files,
            
            transforms = [
            
                #########################  FOLDERS SETUP  #########################
                base_t.DFAddDatafield('raw_path',       ds.STD_FOLDER_RAW),
                base_t.DFAddDatafield('processed_path', ds.STD_FOLDER_PROCESSED),

                base_t.DFSPecializePath([ds.KEY_ROOT, 'raw_path'], 'raw_path'),
                base_t.DFSPecializePath([ds.KEY_ROOT, 'processed_path'], 'processed_path'),

                base_t.DFLocalizeFiles(
                    root_df = 'raw_path',
                    file_dfs = [
                        'data_file',
                    ]
                ),
                base_t.DFLocalizeFiles(
                    root_df = 'processed_path',
                    file_dfs = list(out_files.keys())
                ),

                base_t.DFCreateFolder(
                    destination_df =    'processed_path'
                ),

                ########################  DATASET LOADING  #########################
                grph_t.DFLoadGraphListTorch(
                    load_path_df =      'data_file',
                    graph_list_df =     'data_list'
                ),

                base_t.DFUnpackDatafield(
                    src_df =     'data_list',
                    dst_dfs =    [
                        'adjmats_list',
                        None, None, # eigenvalues and eigenvectors are not needed
                        None, # n_nodes not needed
                        None, None, # max_eigval and min_eigval not needed
                        None, None # same_sample and n_max not needed
                    ]
                ),

                grph_t.DFAdjMatsToGraphs(
                    list_of_adjmats_df =    'adjmats_list',
                    list_of_graphs_df =     'data_list',
                    to_one_hot =            compute_one_hot
                ),

                base_t.DFAddDatafield('num_cls_nodes', 1),
                base_t.DFAddDatafield('num_cls_edges', 1),
                base_t.DFAddDatafield('dim_targets', 0),

                ###########  IF SHOULD SPLIT, SPLIT INTO DATASET SPLITS  ###########
                base_t.guarded_include(
                    if_ = tr_val_test_split, # pipeline input parameter
                    do_ = split_t.DFSplitTrainTestValid(
                        data_list_df =		'data_list',
                        test_fraction =		test_split,     # pipeline input parameter
                        valid_fraction =	valid_split,    # pipeline input parameter
                        test_df =			'graphs_test',
                        valid_df =			'graphs_valid',
                        train_df =			'graphs_train'
                    )
                ),

                #######################  SPLITTING PIPELINE  #######################
                split_t.DFForEachSplit(
                    split_names =   splits,
                    dont_split = not tr_val_test_split,

                    transform =     base_t.DFCompose([
                        ###################  SAVE GRAPHS DATASET  ##################
                        grph_t.DFSaveGraphListTorch(
                            graph_list_df =     'graphs{split}',
                            save_path_df =      ds.KEY_DATA_CONTAINER + '{split}'
                        ),

                        ###################  SAVE DATASET INFOS  ###################
                        # collect min and max number of nodes
                        grph_t.DFCollectGraphNodesStatistics(
                            list_of_graphs_df = 'graphs{split}',
                            df_stats_dict = {
                                'num_nodes_min': np.min,
                                'num_nodes_max': np.max
                            },
                            histogram_df =      'num_nodes_hist'
                        ),
                        # collect number of molecules
                        base_t.DFCustomTransform(
                            src_datafield =     'graphs{split}',
                            dst_datafield =     'num_graphs',
                            free_transform = 	get_len
                        ),
                        # save the above statistics to file as
                        # a dictionary
                        base_t.DFSaveToFile(
                            save_path_df =      ds.KEY_DATA_INFO + '{split}',
                            datafields = [
                                'num_nodes_min',
                                'num_nodes_max',
                                'num_nodes_hist',
                                'num_cls_nodes',
                                'num_cls_edges',
                                'dim_targets',
                                'num_graphs',
                            ]
                        )
                    ]),
                )
            ]
        )

        return pipeline
    



class PickledDatasetPreprocessingPipeline:

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.filename = _PICKLED_DATASET_NAME_TO_FILENAME[dataset_name]
    

    def __call__(
            self,
            tr_val_test_split: bool=True,
            valid_split: int|float=0.2,
            test_split: int|float=0.2,
            compute_one_hot: bool = True
        ) -> base_t.DFPipeline:


        splits, out_files = split_t.get_outputdfs_files_splits(
            split_test = tr_val_test_split,
            split_valid = tr_val_test_split and valid_split is not None,
            outdf_file_ids_exts=[
                (ds.KEY_DATA_CONTAINER, 'data', 'pt'),
                (ds.KEY_DATA_INFO, 'info', 'json'),
            ],
            file_format_string= self.dataset_name + '{split}_{file_id}.{ext}'
        )
        

        ############################################################################
        #                              PIPELINE START                              #
        ############################################################################
        pipeline = base_t.DFPipeline(

            input_files = {
                'data_file': self.filename, # pipeline input parameter
            },

            output_files = out_files,
            
            transforms = [
            
                #########################  FOLDERS SETUP  #########################
                base_t.DFAddDatafield('raw_path',       ds.STD_FOLDER_RAW),
                base_t.DFAddDatafield('processed_path', ds.STD_FOLDER_PROCESSED),

                base_t.DFSPecializePath([ds.KEY_ROOT, 'raw_path'], 'raw_path'),
                base_t.DFSPecializePath([ds.KEY_ROOT, 'processed_path'], 'processed_path'),

                base_t.DFLocalizeFiles(
                    root_df = 'raw_path',
                    file_dfs = [
                        'data_file',
                    ]
                ),
                base_t.DFLocalizeFiles(
                    root_df = 'processed_path',
                    file_dfs = list(out_files.keys())
                ),

                base_t.DFCreateFolder(
                    destination_df =    'processed_path'
                ),

                ########################  DATASET LOADING  #########################
                grph_t.DFLoadGraphListPickle(
                    load_path_df =      'data_file',
                    graph_list_df =     'data_list'
                ),

                grph_t.DFNetworkxToGraphs(
                    list_of_nxgraphs_df =    'data_list',
                    list_of_graphs_df =     'data_list',
                    to_one_hot =            compute_one_hot,
                    remove_self_loops =     True
                ),

                base_t.DFAddDatafield('num_cls_nodes', 1),
                base_t.DFAddDatafield('num_cls_edges', 1),
                base_t.DFAddDatafield('dim_targets', 0),

                ###########  IF SHOULD SPLIT, SPLIT INTO DATASET SPLITS  ###########
                base_t.guarded_include(
                    if_ = tr_val_test_split, # pipeline input parameter
                    do_ = split_t.DFSplitTrainTestValid(
                        data_list_df =		'data_list',
                        test_fraction =		test_split,     # pipeline input parameter
                        valid_fraction =	valid_split,    # pipeline input parameter
                        test_df =			'graphs_test',
                        valid_df =			'graphs_valid',
                        train_df =			'graphs_train'
                    )
                ),

                #######################  SPLITTING PIPELINE  #######################
                split_t.DFForEachSplit(
                    split_names =   splits,
                    dont_split = not tr_val_test_split,

                    transform =     base_t.DFCompose([
                        ###################  SAVE GRAPHS DATASET  ##################
                        grph_t.DFSaveGraphListTorch(
                            graph_list_df =     'graphs{split}',
                            save_path_df =      ds.KEY_DATA_CONTAINER + '{split}'
                        ),

                        ###################  SAVE DATASET INFOS  ###################
                        # collect min and max number of nodes
                        grph_t.DFCollectGraphNodesStatistics(
                            list_of_graphs_df = 'graphs{split}',
                            df_stats_dict = {
                                'num_nodes_min': np.min,
                                'num_nodes_max': np.max
                            },
                            histogram_df =      'num_nodes_hist'
                        ),
                        # collect number of molecules
                        base_t.DFCustomTransform(
                            src_datafield =     'graphs{split}',
                            dst_datafield =     'num_graphs',
                            free_transform = 	get_len
                        ),
                        # save the above statistics to file as
                        # a dictionary
                        base_t.DFSaveToFile(
                            save_path_df =      ds.KEY_DATA_INFO + '{split}',
                            datafields = [
                                'num_nodes_min',
                                'num_nodes_max',
                                'num_nodes_hist',
                                'num_cls_nodes',
                                'num_cls_edges',
                                'dim_targets',
                                'num_graphs',
                            ]
                        )
                    ]),
                )
            ]
        )

        return pipeline