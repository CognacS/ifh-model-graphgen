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
from src.data.pipelines.preprocessing_common import graph_list_to_one_hot_transform
import src.data.dataset as ds


from torch_geometric.datasets.qm9 import conversion
import numpy as np


def get_len(d: List) -> int:
    return len(d)
    
def get_shape_1(y) -> int:
    return y.shape[1]



def qm9_preprocessing_pipeline(
        sanitize: bool=True,
        kekulize: bool=True,
        remove_hydrogens: bool=True,
        remove_nonsmiles: bool=True,
        tr_val_test_split: bool=True,
        valid_split: int = 10000,
        test_split: int = 10000,
        compute_one_hot: bool = True
    ) -> base_t.DFPipeline:


    # SPLIT CASE
    # each output datafield is associated with its file
    # e.g. 'dataset_path_train' -> 'qm9_train_data.pt'
    # e.g. 'smiles_file_test' -> 'qm9_test_smiles.json'
    # NON-SPLIT CASE
    # each output datafield is associated with its file
    # e.g. 'dataset_path' -> 'qm9_data.pt'

    splits, out_files = split_t.get_outputdfs_files_splits(
        split_test = tr_val_test_split,
        split_valid = tr_val_test_split and valid_split is not None,
        outdf_file_ids_exts=[
            (ds.KEY_DATA_CONTAINER, 'data', 'pt'),
            (ds.KEY_DATA_INFO, 'info', 'json'),
            ('smiles_file', 'smiles', 'json'),
        ],
        file_format_string='qm9{split}_{file_id}.{ext}'
    )
    

    ############################################################################
    #                              PIPELINE START                              #
    ############################################################################
    pipeline = base_t.DFPipeline(

        input_files = {
            'mols_file': 'gdb9.sdf',
            'targets_csv_file': 'gdb9.sdf.csv',
            'skip_csv_file': '3195404'
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
                    'mols_file',
                    'targets_csv_file',
                    'skip_csv_file'
                ]
            ),
            base_t.DFLocalizeFiles(
                root_df = 'processed_path',
                file_dfs = list(out_files.keys())
            ),

            base_t.DFCreateFolder(
                destination_df =    'processed_path'
            ),

            #####################  TARGETS PREPROCESSING  #####################
            prep_t.DFReadCSV(
                datafield =         'targets_csv_file',
                columns_df =        'targets_columns',
                index_df =          'targets_index',
                table_df =          'targets',
                delimiter =         ','
            ),
            chem_t.DFApplyUnitConversion(
                datafield =         'targets',
                conversion_factors =conversion
            ),

            ######################  SETUP SKIP MOLECULES  ######################
            base_t.DFCustomTransform(
                src_datafield =     'skip_csv_file',
                dst_datafield =     'skip_list',
                free_transform =    qm9_t.prepare_skip_list
            ),

            ####################  MOLECULES PREPROCESSING  ####################
            base_t.DFAddDatafield('mol_graphs', []),
            base_t.DFAddDatafield('smiles_list', []),
            chem_t.DFReadMolecules(
                datafield =         'mols_file',
                new_datafield =     'mols',
                sanitize =          sanitize,           # pipeline input parameter
                remove_hydrogens =  remove_hydrogens    # pipeline input parameter
            ),
            base_t.DFIterateOver(
                datafield =         'mols',
                iter_idx_df =       'curr_idx',
                iter_elem_df =      'curr_mol',

                transform=base_t.DFCompose([

                    ################  FILTER CORRECT MOLECULES  ################
                    base_t.DFConditional(
                        
                        #######################  FILTER  #######################
                        condition = cond_t.ManyConditions([
                            cond_t.CondNotNone(
                                obj_to_check_df =       'curr_mol',
                            ),
                            cond_t.CondInList(
                                obj_to_check_df =       'curr_idx',
                                check_list_df =         'skip_list',
                                check =                 'not in'
                            )
                        ]),

                        #########  RDKIT MOLECULE TO TORCH GEOM GRAPH  #########
                        transform=base_t.DFCompose([

                            ############  APPLY PREPROC TO MOL (KEKULIZE)  ############
                            chem_t.DFMoleculePreprocess(
                                molecule_df =	'curr_mol',
                                kekulize =		kekulize
                            ),
                
                            #################  MOL TO SMILES  #################
                            chem_t.DFMolToSmiles(
                                mol_df =				'curr_mol',
                                smiles_df =				'full_smiles',
                                sanitize_smiles =       not kekulize
                            ),
                            #base_t.DFAppendElement(
                            #    elem_df =				'curr_smiles',
                            #    datalist_df =			'smiles_list'
                            #),

                            #base_t.DFPrintDatafield('full_smiles'),

                            ###############  MOL TO TORCH GRAPH  ###############
                            chem_t.DFAtomsData(
                                molecule_df =           'curr_mol',
                                result_nodes_df =       'curr_nodes',
                                atom_types_df =         'atom_types'
                            ),
                            chem_t.DFBondsData(
                                molecule_df =           'curr_mol',
                                result_edge_index_df =  'curr_edge_index',
                                result_edge_attr_df =   'curr_edge_attr',
                                bond_types_df =         'bond_types'
                            ),
                            base_t.DFIndexElement(
                                datalist_df =           'targets',
                                index_df =              'curr_idx',
                                result_df =             'curr_targets'
                            ),
                            chem_t.DFToGraph(
                                result_df =             'res_graph',
                                nodes_df =              'curr_nodes',
                                edge_index_df =         'curr_edge_index',
                                edge_attr_df =          'curr_edge_attr',
                                targets_df =            'curr_targets'
                            ),
                            #################  GRAPH TO SMILES  ################
                            chem_t.DFGraphToMol(
                                graph_df =              'res_graph',
                                result_mol_df =         'res_mol',
                                atom_decoder_df =       'atom_types',
                                bond_decoder_df =       'bond_types'
                            ),
                            chem_t.DFMolToSmiles(
                                mol_df =				'res_mol',
                                smiles_df =				'res_smiles',
                                sanitize_smiles =       True
                            ),

                            ################  FILTER INVALIDS  #################
                            # the filter is added only if remove_nonsmiles is True
                            # otherwise, the conditional is always True
                            base_t.DFConditional(
                                condition = base_t.guarded_include(
                                    if_ = remove_nonsmiles, # pipeline input parameter
                                    do_ = chem_t.CondValidSMILES('res_smiles')
                                ),
                                transform = base_t.DFCompose([
                                    # compute molecular properties on the actual molecule
                                    # computed from the graph
                                    chem_t.DFComputeMolecularProperties(
                                        mol_df =                'res_mol',
                                        graph_df =              'res_graph',
                                        properties = [
                                            'qed',
                                            'plogp'
                                        ]
                                    ),
                                    base_t.DFAppendElement(
                                        elem_df =				'full_smiles', # use correct smiles
                                        datalist_df =			'smiles_list'
                                    ),
                                    base_t.DFAppendElement(
                                        elem_df =				'res_graph',
                                        datalist_df =			'mol_graphs'
                                    )
                                ])
                            )

                        ])
                    )
                ])
            ),

            # collect number of atom types
            base_t.DFCustomTransform(
                src_datafield =     'atom_types',
                dst_datafield =     'num_cls_nodes',
                free_transform = 	get_len
            ),
            # collect number of bond types
            base_t.DFCustomTransform(
                src_datafield =     'bond_types',
                dst_datafield =     'num_cls_edges',
                free_transform = 	get_len
            ),
            # collect number of target components
            base_t.DFCustomTransform(
                src_datafield =     'targets',
                dst_datafield =     'dim_targets',
                free_transform = 	get_shape_1
            ),

            # transform all graphs features to one-hot
            # if enabled (compute_onehot = True)
            graph_list_to_one_hot_transform(
                graph_list_df =         'mol_graphs',
                num_classes_node_df =   'num_cls_nodes',
                num_classes_edge_df =   'num_cls_edges',
                enable =                compute_one_hot
            ),

            ###########  IF SHOULD SPLIT, SPLIT INTO DATASET SPLITS  ###########
            base_t.guarded_include(
                if_ = tr_val_test_split, # pipeline input parameter
                do_ = split_t.DFSplitTrainTestValid(
                    data_list_df =		['mol_graphs',   'smiles_list'],
                    test_fraction =		test_split,     # pipeline input parameter
                    valid_fraction =	valid_split,    # pipeline input parameter
                    test_df =			['graphs_test',  'smiles_test'],
                    valid_df =			['graphs_valid', 'smiles_valid'],
                    train_df =			['graphs_train', 'smiles_train']
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
                        dst_datafield =     'num_molecules',
                        free_transform = 	get_len
                    ),
                    # collect number of SMILES
                    base_t.DFCustomTransform(
                        src_datafield =     'smiles{split}',
                        dst_datafield =     'num_smiles',
                        free_transform = 	get_len
                    ),
                    # save the above statistics to file as
                    # a dictionary
                    base_t.DFSaveToFile(
                        save_path_df =      ds.KEY_DATA_INFO + '{split}',
                        datafields = [
                            'atom_types',
                            'bond_types',
                            'num_nodes_min',
                            'num_nodes_max',
                            'num_nodes_hist',
                            'num_cls_nodes',
                            'num_cls_edges',
                            'dim_targets',
                            'num_molecules',
                            'num_smiles'
                        ]
                    ),
                    base_t.DFSaveToFile(
                        save_path_df =      'smiles_file{split}',
                        datafields = 		['smiles{split}']
                    )
                ]),
            )
        ]
    )

    return pipeline

"""
def qm9_preprocessing_pipeline(
        sanitize: bool=True,
        kekulize: bool=True,
        remove_hydrogens: bool=True,
        remove_nonsmiles: bool=True,
        tr_val_test_split: bool=True,
        valid_split: int = 10000,
        test_split: int = 10000,
        compute_one_hot: bool = True,
        num_workers: int = 1
    ) -> base_t.DFPipeline:


    # SPLIT CASE
    # each output datafield is associated with its file
    # e.g. 'dataset_path_train' -> 'qm9_train_data.pt'
    # e.g. 'smiles_file_test' -> 'qm9_test_smiles.json'
    # NON-SPLIT CASE
    # each output datafield is associated with its file
    # e.g. 'dataset_path' -> 'qm9_data.pt'

    splits, out_files = split_t.get_outputdfs_files_splits(
        split_test = tr_val_test_split,
        split_valid = tr_val_test_split and valid_split is not None,
        outdf_file_ids_exts=[
            (ds.KEY_DATA_CONTAINER, 'data', 'pt'),
            (ds.KEY_DATA_INFO, 'info', 'json'),
            ('smiles_file', 'smiles', 'json'),
        ],
        file_format_string='qm9{split}_{file_id}.{ext}'
    )
    

    ############################################################################
    #                              PIPELINE START                              #
    ############################################################################
    pipeline = base_t.DFPipeline(

        input_files = {
            'mols_file': 'gdb9.sdf',
            'targets_csv_file': 'gdb9.sdf.csv',
            'skip_csv_file': '3195404'
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
                    'mols_file',
                    'targets_csv_file',
                    'skip_csv_file'
                ]
            ),
            base_t.DFLocalizeFiles(
                root_df = 'processed_path',
                file_dfs = list(out_files.keys())
            ),

            base_t.DFCreateFolder(
                destination_df =    'processed_path'
            ),

            #####################  TARGETS PREPROCESSING  #####################
            prep_t.DFReadCSV(
                datafield =         'targets_csv_file',
                columns_df =        'targets_columns',
                index_df =          'targets_index',
                table_df =          'targets',
                delimiter =         ','
            ),
            chem_t.DFApplyUnitConversion(
                datafield =         'targets',
                conversion_factors =conversion
            ),

            ######################  SETUP SKIP MOLECULES  ######################
            base_t.DFCustomTransform(
                src_datafield =     'skip_csv_file',
                dst_datafield =     'skip_list',
                free_transform =    qm9_t.prepare_skip_list
            ),

            ####################  MOLECULES PREPROCESSING  ####################
            chem_t.DFReadMolecules(
                datafield =         'mols_file',
                new_datafield =     'mols',
                sanitize =          sanitize,           # pipeline input parameter
                remove_hydrogens =  remove_hydrogens    # pipeline input parameter
            ),

            ###################  GATHER INFO AND PREPROCESS  ###################
            base_t.DFMultiProcessMap(
                input_list_df =     'mols',
                iter_idx_df =       'curr_idx',
                iter_elem_df =      'curr_mol',
                iter_ret_dfs =      ['ret_smiles', 'ret_graph'],
                output_list_dfs =   ['smiles_list', 'mol_graphs'],
                num_workers =       num_workers,
                transform=base_t.DFCompose([

                    ################  FILTER CORRECT MOLECULES  ################
                    base_t.DFConditional(
                        
                        #######################  FILTER  #######################
                        condition = cond_t.ManyConditions([
                            cond_t.CondNotNone(
                                obj_to_check_df =       'curr_mol',
                            ),
                            cond_t.CondNotInSkipList(
                                obj_to_check_df =       'curr_idx',
                                skip_list_df =          'skip_list'
                            )
                        ]),

                        #########  RDKIT MOLECULE TO TORCH GEOM GRAPH  #########
                        transform=base_t.DFCompose([

                            ############  APPLY PREPROC TO MOL (KEKULIZE)  ############
                            chem_t.DFMoleculePreprocess(
                                molecule_df =	'curr_mol',
                                kekulize =		kekulize
                            ),
                
                            #################  MOL TO SMILES  #################
                            chem_t.DFMolToSmiles(
                                mol_df =				'curr_mol',
                                smiles_df =				'full_smiles',
                                sanitize_smiles =       not kekulize
                            ),
                            #base_t.DFAppendElement(
                            #    elem_df =				'curr_smiles',
                            #    datalist_df =			'smiles_list'
                            #),

                            #base_t.DFPrintDatafield('full_smiles'),

                            ###############  MOL TO TORCH GRAPH  ###############
                            chem_t.DFAtomsData(
                                molecule_df =           'curr_mol',
                                result_nodes_df =       'curr_nodes',
                                atom_types_df =         'atom_types'
                            ),
                            chem_t.DFBondsData(
                                molecule_df =           'curr_mol',
                                result_edge_index_df =  'curr_edge_index',
                                result_edge_attr_df =   'curr_edge_attr',
                                bond_types_df =         'bond_types'
                            ),
                            base_t.DFIndexElement(
                                datalist_df =           'targets',
                                index_df =              'curr_idx',
                                result_df =             'curr_targets'
                            ),
                            chem_t.DFToGraph(
                                result_df =             'res_graph',
                                nodes_df =              'curr_nodes',
                                edge_index_df =         'curr_edge_index',
                                edge_attr_df =          'curr_edge_attr',
                                targets_df =            'curr_targets'
                            ),
                            #################  GRAPH TO SMILES  ################
                            chem_t.DFGraphToMol(
                                graph_df =              'res_graph',
                                result_mol_df =         'res_mol',
                                atom_decoder_df =       'atom_types',
                                bond_decoder_df =       'bond_types'
                            ),
                            chem_t.DFMolToSmiles(
                                mol_df =				'res_mol',
                                smiles_df =				'res_smiles',
                                sanitize_smiles =       True
                            ),

                            ################  FILTER INVALIDS  #################
                            # the filter is added only if remove_nonsmiles is True
                            # otherwise, the conditional is always True
                            base_t.DFConditional(
                                condition = base_t.guarded_include(
                                    if_ = remove_nonsmiles, # pipeline input parameter
                                    do_ = chem_t.CondValidSMILES('res_smiles')
                                ),
                                transform = base_t.DFCompose([
                                    # compute molecular properties on the actual molecule
                                    # computed from the graph
                                    chem_t.DFComputeMolecularProperties(
                                        mol_df =                'res_mol',
                                        graph_df =              'res_graph',
                                        properties = [
                                            'qed',
                                            'plogp'
                                        ]
                                    ),
                                    base_t.DFRenameDatafield(
                                        elem_df =				'full_smiles', # use correct smiles
                                        datalist_df =			'ret_smiles'
                                    ),
                                    base_t.DFRenameDatafield(
                                        elem_df =				'res_graph',
                                        datalist_df =			'ret_graph'
                                    )
                                ])
                            )

                        ])
                    )
                ])
            ),

            # collect number of atom types
            base_t.DFCustomTransform(
                src_datafield =     'atom_types',
                dst_datafield =     'num_cls_nodes',
                free_transform = 	get_len
            ),
            # collect number of bond types
            base_t.DFCustomTransform(
                src_datafield =     'bond_types',
                dst_datafield =     'num_cls_edges',
                free_transform = 	get_len
            ),
            # collect number of target components
            base_t.DFCustomTransform(
                src_datafield =     'targets',
                dst_datafield =     'dim_targets',
                free_transform = 	get_shape_1
            ),

            # transform all graphs features to one-hot
            # if enabled (compute_onehot = True)
            graph_list_to_one_hot_transform(
                graph_list_df =         'mol_graphs',
                num_classes_node_df =   'num_cls_nodes',
                num_classes_edge_df =   'num_cls_edges',
                enable =                compute_one_hot
            ),

            ###########  IF SHOULD SPLIT, SPLIT INTO DATASET SPLITS  ###########
            base_t.guarded_include(
                if_ = tr_val_test_split, # pipeline input parameter
                do_ = split_t.DFSplitTrainTestValid(
                    data_list_df =		['mol_graphs',   'smiles_list'],
                    test_fraction =		test_split,     # pipeline input parameter
                    valid_fraction =	valid_split,    # pipeline input parameter
                    test_df =			['graphs_test',  'smiles_test'],
                    valid_df =			['graphs_valid', 'smiles_valid'],
                    train_df =			['graphs_train', 'smiles_train']
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
                        dst_datafield =     'num_molecules',
                        free_transform = 	get_len
                    ),
                    # collect number of SMILES
                    base_t.DFCustomTransform(
                        src_datafield =     'smiles{split}',
                        dst_datafield =     'num_smiles',
                        free_transform = 	get_len
                    ),
                    # save the above statistics to file as
                    # a dictionary
                    base_t.DFSaveToFile(
                        save_path_df =      ds.KEY_DATA_INFO + '{split}',
                        datafields = [
                            'atom_types',
                            'bond_types',
                            'num_nodes_min',
                            'num_nodes_max',
                            'num_nodes_hist',
                            'num_cls_nodes',
                            'num_cls_edges',
                            'dim_targets',
                            'num_molecules',
                            'num_smiles'
                        ]
                    ),
                    base_t.DFSaveToFile(
                        save_path_df =      'smiles_file{split}',
                        datafields = 		['smiles{split}']
                    )
                ]),
            )
        ]
    )

    return pipeline"""

class ZincPreprocessingPipeline:

    def __init__(self, which_zinc: str='250k'):
        self.which_zinc = which_zinc

    def __call__(
            self,
            sanitize: bool=True,
            kekulize: bool=True,
            remove_nonsmiles: bool=True,
            tr_val_test_split: bool=True,
            valid_split: int = 0.1,
            test_split: int = 0.1,
            compute_one_hot: bool = True
        ) -> base_t.DFPipeline:


        splits, out_files = split_t.get_outputdfs_files_splits(
            split_test = tr_val_test_split,
            split_valid = tr_val_test_split and valid_split is not None,
            outdf_file_ids_exts=[
                (ds.KEY_DATA_CONTAINER, 'data', 'pt'),
                (ds.KEY_DATA_INFO, 'info', 'json'),
                ('smiles_file', 'smiles', 'json'),
            ],
            file_format_string='zinc{split}_{file_id}.{ext}'
        )
        

        ############################################################################
        #                              PIPELINE START                              #
        ############################################################################
        pipeline = base_t.DFPipeline(

            input_files = {
                'smiles_csv_file': f'zinc{self.which_zinc}.csv',
                'valid_idx_file': f'valid_idx_zinc{self.which_zinc}.json'
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
                        'smiles_csv_file',
                        'valid_idx_file'
                    ]
                ),
                base_t.DFLocalizeFiles(
                    root_df = 'processed_path',
                    file_dfs = list(out_files.keys())
                ),

                base_t.DFCreateFolder(
                    destination_df =    'processed_path'
                ),

                #####################  TARGETS PREPROCESSING  #####################
                # read smiles
                prep_t.DFReadCSV(
                    datafield =         'smiles_csv_file',
                    table_df =          'smiles_table',
                    astype =            'pandas',
                    index_col =         0
                ),
                # read index of smiles to use as tests set
                prep_t.DFReadJSON(
                    datafield =         'valid_idx_file',
                    output_df =         'test_idx',
                ),

                # preprocess smiles table
                prep_t.DFSelectColumnsDataframe(
                    table_df =          'smiles_table',
                    result_df =         'smiles_list_full',
                    selected_columns =  'smiles'
                ),

                prep_t.DFSelectColumnsDataframe(
                    table_df =          'smiles_table',
                    result_df =         'targets',
                    selected_columns =  slice(1, None)
                ),

                ####################  MOLECULES PREPROCESSING  ####################
                base_t.DFAddDatafield('mol_graphs', []),
                base_t.DFAddDatafield('smiles_list', []),
                base_t.DFAddDatafield('true_test_idx', []),
                base_t.DFIterateOver(
                    datafield =         'smiles_list_full',
                    iter_idx_df =       'curr_idx',
                    iter_elem_df =      'curr_smiles',

                    transform=base_t.DFCompose([
            
                        ############  APPLY PREPROC TO MOL (KEKULIZE)  ############
                        chem_t.DFSmilesToMol(
                            smiles_df =             'curr_smiles',
                            mol_df =                'curr_mol',
                            sanitize =              sanitize
                        ),

                        ################  FILTER CORRECT MOLECULES  ################
                        base_t.DFConditional(
                            
                            #######################  FILTER  #######################
                            condition = cond_t.ManyConditions([
                                cond_t.CondNotNone(
                                    obj_to_check_df =       'curr_mol',
                                )
                            ]),

                            #########  RDKIT MOLECULE TO TORCH GEOM GRAPH  #########
                            transform=base_t.DFCompose([

                                chem_t.DFMoleculePreprocess(
                                    molecule_df =           'curr_mol',
                                    kekulize =              kekulize
                                ),
                    
                                #################  MOL TO SMILES  #################
                                # chem_t.DFMolToSmiles(
                                #     mol_df =				'curr_mol',
                                #     smiles_df =				'full_smiles',
                                #     sanitize_smiles =       not kekulize
                                # ),
                                #base_t.DFAppendElement(
                                #    elem_df =				'curr_smiles',
                                #    datalist_df =			'smiles_list'
                                #),

                                ###############  MOL TO TORCH GRAPH  ###############
                                chem_t.DFAtomsData(
                                    molecule_df =           'curr_mol',
                                    result_nodes_df =       'curr_nodes',
                                    atom_types_df =         'atom_types'
                                ),
                                chem_t.DFBondsData(
                                    molecule_df =           'curr_mol',
                                    result_edge_index_df =  'curr_edge_index',
                                    result_edge_attr_df =   'curr_edge_attr',
                                    bond_types_df =         'bond_types'
                                ),
                                base_t.DFIndexElement(
                                    datalist_df =           'targets',
                                    index_df =              'curr_idx',
                                    result_df =             'curr_targets'
                                ),
                                chem_t.DFToGraph(
                                    result_df =             'res_graph',
                                    nodes_df =              'curr_nodes',
                                    edge_index_df =         'curr_edge_index',
                                    edge_attr_df =          'curr_edge_attr',
                                    targets_df =            'curr_targets'
                                ),
                                #################  GRAPH TO SMILES  ################
                                chem_t.DFGraphToMol(
                                    graph_df =              'res_graph',
                                    result_mol_df =         'res_mol',
                                    atom_decoder_df =       'atom_types',
                                    bond_decoder_df =       'bond_types'
                                ),
                                chem_t.DFMolToSmiles(
                                    mol_df =				'res_mol',
                                    smiles_df =				'res_smiles'
                                ),

                                ################  FILTER INVALIDS  #################
                                # the filter is added only if remove_nonsmiles is True
                                # otherwise, the conditional is always True
                                base_t.DFConditional(
                                    condition = base_t.guarded_include(
                                        if_ = remove_nonsmiles, # pipeline input parameter
                                        do_ = chem_t.CondValidSMILES('res_smiles')
                                    ),
                                    transform = base_t.DFCompose([
                                        # compute molecular properties on the actual molecule
                                        # computed from the graph
                                        chem_t.DFComputeMolecularProperties(
                                            mol_df =                'res_mol',
                                            graph_df =              'res_graph',
                                            properties = [
                                                'qed',
                                                'plogp'
                                            ]
                                        ),
                                        # save the current index for putting the datapoint
                                        base_t.DFCustomTransform(
                                            src_datafield =     'smiles_list',
                                            dst_datafield =     'curr_data_idx',
                                            free_transform = 	get_len
                                        ),
                                        # append smiles and graph to data lists
                                        base_t.DFAppendElement(
                                            elem_df =				'curr_smiles', # use correct smiles
                                            datalist_df =			'smiles_list'
                                        ),
                                        base_t.DFAppendElement(
                                            elem_df =				'res_graph',
                                            datalist_df =			'mol_graphs'
                                        ),
                                        # if datapoint is in test,
                                        # append the actual index to the test index list
                                        # because some datapoints are removed
                                        # not being valid
                                        base_t.DFConditional(
                                            condition = cond_t.CondInList(
                                                obj_to_check_df =       'curr_idx',
                                                check_list_df =         'test_idx',
                                                check =                 'in'
                                            ),
                                            transform = base_t.DFAppendElement(
                                                elem_df =				'curr_data_idx',
                                                datalist_df =			'true_test_idx'
                                            )
                                        )
                                    ])
                                )

                            ])
                        )
                    ])
                ),

                # collect number of atom types
                base_t.DFCustomTransform(
                    src_datafield =     'atom_types',
                    dst_datafield =     'num_cls_nodes',
                    free_transform = 	get_len
                ),
                # collect number of bond types
                base_t.DFCustomTransform(
                    src_datafield =     'bond_types',
                    dst_datafield =     'num_cls_edges',
                    free_transform = 	get_len
                ),
                # collect number of target components
                base_t.DFCustomTransform(
                    src_datafield =     'targets',
                    dst_datafield =     'dim_targets',
                    free_transform = 	get_shape_1
                ),

                # transform all graphs features to one-hot
                # if enabled (compute_onehot = True)
                graph_list_to_one_hot_transform(
                    graph_list_df =         'mol_graphs',
                    num_classes_node_df =   'num_cls_nodes',
                    num_classes_edge_df =   'num_cls_edges',
                    enable =                compute_one_hot
                ),

                ###########  IF SHOULD SPLIT, SPLIT INTO DATASET SPLITS  ###########
                base_t.guarded_include(
                    if_ = tr_val_test_split, # pipeline input parameter
                    do_ = base_t.DFCompose([
                        # split train+valid/test using list
                        split_t.DFSplitTrainTestFromList(
                            data_list_df =		['mol_graphs',   'smiles_list'],
                            test_idx_df =       'true_test_idx',
                            test_df =			['graphs_test',  'smiles_test'],
                            train_df =			['graphs_trval', 'smiles_trval']
                        ),
                        # split train/valid at random
                        split_t.DFSplitTrainTestValid(
                            data_list_df =		['graphs_trval',   'smiles_trval'],
                            test_fraction =		valid_split,     # pipeline input parameter
                            test_df =			['graphs_valid',  'smiles_valid'],
                            train_df =			['graphs_train', 'smiles_train']
                        )
                    ])
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
                            dst_datafield =     'num_molecules',
                            free_transform = 	get_len
                        ),
                        # collect number of SMILES
                        base_t.DFCustomTransform(
                            src_datafield =     'smiles{split}',
                            dst_datafield =     'num_smiles',
                            free_transform = 	get_len
                        ),
                        # save the above statistics to file as
                        # a dictionary
                        base_t.DFSaveToFile(
                            save_path_df =      ds.KEY_DATA_INFO + '{split}',
                            datafields = [
                                'atom_types',
                                'bond_types',
                                'num_nodes_min',
                                'num_nodes_max',
                                'num_nodes_hist',
                                'num_cls_nodes',
                                'num_cls_edges',
                                'dim_targets',
                                'num_molecules',
                                'num_smiles'
                            ]
                        ),
                        base_t.DFSaveToFile(
                            save_path_df =      'smiles_file{split}',
                            datafields = 		['smiles{split}']
                        )
                    ]),
                )
            ]
        )

        return pipeline
    


class ZincPreprocessingPipeline:

    def __init__(self, which_zinc: str='250k'):
        self.which_zinc = which_zinc

    def __call__(
            self,
            sanitize: bool=True,
            kekulize: bool=True,
            remove_nonsmiles: bool=True,
            tr_val_test_split: bool=True,
            valid_split: int = 0.1,
            test_split: int = 0.1,
            compute_one_hot: bool = True
        ) -> base_t.DFPipeline:


        splits, out_files = split_t.get_outputdfs_files_splits(
            split_test = tr_val_test_split,
            split_valid = tr_val_test_split and valid_split is not None,
            outdf_file_ids_exts=[
                (ds.KEY_DATA_CONTAINER, 'data', 'pt'),
                (ds.KEY_DATA_INFO, 'info', 'json'),
                ('smiles_file', 'smiles', 'json'),
            ],
            file_format_string='zinc{split}_{file_id}.{ext}'
        )
        

        ############################################################################
        #                              PIPELINE START                              #
        ############################################################################
        pipeline = base_t.DFPipeline(

            input_files = {
                'smiles_csv_file': f'zinc{self.which_zinc}.csv',
                'valid_idx_file': f'valid_idx_zinc{self.which_zinc}.json'
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
                        'smiles_csv_file',
                        'valid_idx_file'
                    ]
                ),
                base_t.DFLocalizeFiles(
                    root_df = 'processed_path',
                    file_dfs = list(out_files.keys())
                ),

                base_t.DFCreateFolder(
                    destination_df =    'processed_path'
                ),

                #####################  TARGETS PREPROCESSING  #####################
                # read smiles
                prep_t.DFReadCSV(
                    datafield =         'smiles_csv_file',
                    table_df =          'smiles_table',
                    astype =            'pandas',
                    index_col =         0
                ),
                # read index of smiles to use as tests set
                prep_t.DFReadJSON(
                    datafield =         'valid_idx_file',
                    output_df =         'test_idx',
                ),

                # preprocess smiles table
                prep_t.DFSelectColumnsDataframe(
                    table_df =          'smiles_table',
                    result_df =         'smiles_list_full',
                    selected_columns =  'smiles'
                ),

                prep_t.DFSelectColumnsDataframe(
                    table_df =          'smiles_table',
                    result_df =         'targets',
                    selected_columns =  slice(1, None)
                ),

                ####################  MOLECULES PREPROCESSING  ####################
                base_t.DFAddDatafield('mol_graphs', []),
                base_t.DFAddDatafield('smiles_list', []),
                base_t.DFIterateOver(
                    datafield =         'smiles_list_full',
                    iter_idx_df =       'curr_idx',
                    iter_elem_df =      'curr_smiles',

                    transform=base_t.DFCompose([
            
                        ############  APPLY PREPROC TO MOL (KEKULIZE)  ############
                        chem_t.DFSmilesToMol(
                            smiles_df =             'curr_smiles',
                            mol_df =                'curr_mol',
                            sanitize =              sanitize
                        ),

                        chem_t.DFMoleculePreprocess(
                            molecule_df =           'curr_mol',
                            kekulize =              kekulize
                        ),

                        ###############  MOL TO TORCH GRAPH  ###############
                        chem_t.DFAtomsData(
                            molecule_df =           'curr_mol',
                            result_nodes_df =       'curr_nodes',
                            atom_types_df =         'atom_types'
                        ),
                        chem_t.DFBondsData(
                            molecule_df =           'curr_mol',
                            result_edge_index_df =  'curr_edge_index',
                            result_edge_attr_df =   'curr_edge_attr',
                            bond_types_df =         'bond_types'
                        ),
                        base_t.DFIndexElement(
                            datalist_df =           'targets',
                            index_df =              'curr_idx',
                            result_df =             'curr_targets'
                        ),
                        chem_t.DFToGraph(
                            result_df =             'res_graph',
                            nodes_df =              'curr_nodes',
                            edge_index_df =         'curr_edge_index',
                            edge_attr_df =          'curr_edge_attr',
                            targets_df =            'curr_targets'
                        ),

                        # compute molecular properties on the actual molecule
                        # computed from the graph
                        chem_t.DFComputeMolecularProperties(
                            mol_df =                'curr_mol',
                            graph_df =              'res_graph',
                            properties = [
                                'qed',
                                'plogp'
                            ]
                        ),
                        # append smiles and graph to data lists
                        base_t.DFAppendElement(
                            elem_df =				'curr_smiles',
                            datalist_df =			'smiles_list'
                        ),
                        base_t.DFAppendElement(
                            elem_df =				'res_graph',
                            datalist_df =			'mol_graphs'
                        )
                    ])
                ),

                # collect number of atom types
                base_t.DFCustomTransform(
                    src_datafield =     'atom_types',
                    dst_datafield =     'num_cls_nodes',
                    free_transform = 	get_len
                ),
                # collect number of bond types
                base_t.DFCustomTransform(
                    src_datafield =     'bond_types',
                    dst_datafield =     'num_cls_edges',
                    free_transform = 	get_len
                ),
                # collect number of target components
                base_t.DFCustomTransform(
                    src_datafield =     'targets',
                    dst_datafield =     'dim_targets',
                    free_transform = 	get_shape_1
                ),

                # transform all graphs features to one-hot
                # if enabled (compute_onehot = True)
                graph_list_to_one_hot_transform(
                    graph_list_df =         'mol_graphs',
                    num_classes_node_df =   'num_cls_nodes',
                    num_classes_edge_df =   'num_cls_edges',
                    enable =                compute_one_hot
                ),

                ###########  IF SHOULD SPLIT, SPLIT INTO DATASET SPLITS  ###########
                base_t.guarded_include(
                    if_ = tr_val_test_split, # pipeline input parameter
                    do_ = base_t.DFCompose([
                        # split train+valid/test using list
                        split_t.DFSplitTrainTestFromList(
                            data_list_df =		['mol_graphs',   'smiles_list'],
                            test_idx_df =       'test_idx',
                            test_df =			['graphs_test',  'smiles_test'],
                            train_df =			['graphs_trval', 'smiles_trval']
                        ),
                        # split train/valid at random
                        split_t.DFSplitTrainTestValid(
                            data_list_df =		['graphs_trval',   'smiles_trval'],
                            test_fraction =		valid_split,     # pipeline input parameter
                            test_df =			['graphs_valid',  'smiles_valid'],
                            train_df =			['graphs_train', 'smiles_train']
                        )
                    ])
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
                            dst_datafield =     'num_molecules',
                            free_transform = 	get_len
                        ),
                        # collect number of SMILES
                        base_t.DFCustomTransform(
                            src_datafield =     'smiles{split}',
                            dst_datafield =     'num_smiles',
                            free_transform = 	get_len
                        ),
                        # save the above statistics to file as
                        # a dictionary
                        base_t.DFSaveToFile(
                            save_path_df =      ds.KEY_DATA_INFO + '{split}',
                            datafields = [
                                'atom_types',
                                'bond_types',
                                'num_nodes_min',
                                'num_nodes_max',
                                'num_nodes_hist',
                                'num_cls_nodes',
                                'num_cls_edges',
                                'dim_targets',
                                'num_molecules',
                                'num_smiles'
                            ]
                        ),
                        base_t.DFSaveToFile(
                            save_path_df =      'smiles_file{split}',
                            datafields = 		['smiles{split}']
                        )
                    ]),
                )
            ]
        )

        return pipeline