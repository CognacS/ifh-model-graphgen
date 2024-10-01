from typing import Dict

from torch_geometric.data import extract_zip

import src.data.transforms as base_t
from src.data.transforms import (
    downloads as down_t
)
import src.data.dataset as ds


class URLDatasetDownloadPipeline:

    def __init__(self, dataset_name: str, dataset_url: str=None, dataset_filenames_map: Dict=None):
        assert dataset_url is not None or dataset_filenames_map is not None, 'Both dataset_url and dataset_options must be provided'
        self.dataset_url = dataset_url
        self.filename = dataset_filenames_map[dataset_name]
    

    def __call__(self) -> base_t.DFPipeline:
        
        pipeline = base_t.DFPipeline(

            output_files = {
                'data_file': f'{self.filename}'
            },
            
            transforms=[
                #########################  FOLDERS SETUP  #########################
                base_t.DFAddDatafield('raw_path',       ds.STD_FOLDER_RAW),
                base_t.DFSPecializePath([ds.KEY_ROOT, 'raw_path'], 'raw_path'),
                
                base_t.DFCreateFolder(
                    destination_df =			'raw_path'
                ),

                ##################  DOWNLOAD AND UNZIP DATASETS  ##################
                down_t.DFDownloadFromURL(
                    download_to_df =	'raw_path',
                    url={
                        'data_file': self.dataset_url + self.filename
                    }
                )
            ]
        )

        return pipeline



_SPECTRE_DATASET_NAME_TO_FILENAME = {
    'community-20': 'community_12_21_100.pt',
    'planar': 'planar_64_200.pt',
    'sbm': 'sbm_200.pt'
}

_SPECTRE_DATASET_URL = 'https://github.com/KarolisMart/SPECTRE/raw/main/data/'


class SpectreDatasetDownloadPipeline(URLDatasetDownloadPipeline):

    def __init__(self, dataset_name: str):
        super().__init__(
            dataset_name=dataset_name,
            dataset_url=_SPECTRE_DATASET_URL,
            dataset_filenames_map=_SPECTRE_DATASET_NAME_TO_FILENAME
        )
    

_PICKLED_DATASET_NAME_TO_FILENAME = {
    'community-20': 'Community_small.pkl',
    'enzymes': 'ENZYMES.pkl',
    'ego': 'Ego.pkl',
    'ego-small': 'Ego_small.pkl',
    'grid': 'graphs.pkl'
}

_CDGS_DATASET_URL = 'https://github.com/GRAPH-0/GraphGDP/raw/main/data/raw/'

class CDGSDatasetDownloadPipeline(URLDatasetDownloadPipeline):

    def __init__(self, dataset_name: str):
        super().__init__(
            dataset_name=dataset_name,
            dataset_url=_CDGS_DATASET_URL,
            dataset_filenames_map=_PICKLED_DATASET_NAME_TO_FILENAME
        )


import numpy as np
import networkx as nx

class GRANDatasetGenerator:

    def __init__(self):
        pass


    def create_gran_grid_graphs(self):
        ### load datasets
        graphs = []
        # synthetic graphs
        for i in range(10, 20):
            for j in range(10, 20):
                graphs.append(nx.grid_2d_graph(i, j))

        num_nodes = [gg.number_of_nodes() for gg in graphs]
        num_edges = [gg.number_of_edges() for gg in graphs]
        print('max # nodes = {} || mean # nodes = {}'.format(max(num_nodes), np.mean(num_nodes)))
        print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges)))
            
        return graphs


    def __call__(self) -> base_t.DFPipeline:

        pipeline = base_t.DFPipeline(

            output_files = {
                'data_file': 'graphs.pkl'
            },
            
            transforms=[
                #########################  FOLDERS SETUP  #########################
                base_t.DFAddDatafield('raw_path',       ds.STD_FOLDER_RAW),
                base_t.DFSPecializePath([ds.KEY_ROOT, 'raw_path'], 'raw_path'),
                
                base_t.DFCreateFolder(
                    destination_df =			'raw_path'
                ),

                base_t.DFLocalizeFiles(
                    root_df = 'raw_path',
                    file_dfs = [
                        'data_file',
                    ]
                ),

                ##################  GENERATE AND SAVE GRAPHS  ##################
                base_t.DFCustomTransform(
                    src_datafield =     None,
                    dst_datafield =     'graphs',
                    free_transform =    self.create_gran_grid_graphs
                ),
                base_t.DFSaveToFile(
                    save_path_df =      'data_file',
                    datafields =        'graphs',
                    save_file_method =  'pickle'
                )
            ]
        )

        return pipeline