import torch
import torch.nn.functional as F

from typing import Optional, Callable, List, Union, Dict

import os.path as osp
import json

from torch_geometric.data import (
    Data,
    InMemoryDataset
)
from torch_geometric.data.separate import separate
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_undirected

from .transforms import DFPipeline


KEY_ROOT = 'root_path'

KEY_DATA_CONTAINER = 'dataset_path'
KEY_DATA_INFO = 'dataset_info_path'

STD_FOLDER_PROCESSED = 'processed'
STD_FOLDER_RAW = 'raw'

class GraphDataset(InMemoryDataset):
    """ This dataset object is similar to the ones found in torch_geometric,
    but tries to be a more general and informed version of InMemoryDataset,
    collecting classes and statistics to be used (the dataset information part).
    """

    def __init__(
            self,
            root: str,
            download_pipeline: Callable,
            runtimeprocessing_transform: Optional[Callable] = None,
            preprocessing_transform: Optional[Callable] = None,
            preprocessing_filter: Optional[Callable] = None,
            dataset_split: Optional[str] = None,
            only_load_info: Optional[bool] = False
        ):

        # before calling super, setup the download pipeline
        self.download_pipeline = download_pipeline

        super().__init__(
            root=root,
            transform=runtimeprocessing_transform,
            pre_transform=preprocessing_transform,
            pre_filter=preprocessing_filter
        )

        data_key = KEY_DATA_CONTAINER
        info_key = KEY_DATA_INFO

        if dataset_split is not None:
            data_key += '_' + dataset_split
            info_key += '_' + dataset_split
        else:
            if data_key not in self.processed_paths_dictionary:
                print('Warning: dataset not loaded in memory!')
                return


        if not only_load_info:
            # setup data and slices of data, i.e., how to index the dataset,
            # which is built as a big single graph to be more efficient
            self.data, self.slices = torch.load(
                self.processed_paths_dictionary[data_key]
            )
        

        with open(self.processed_paths_dictionary[info_key], 'r') as info_file:
            self.info = json.load(info_file)


    @property
    def raw_file_names(self) -> List[str]:
        if isinstance(self.download_pipeline, DFPipeline):
            return list(self.download_pipeline.get_output_files().values())
        else:
            return 'raw_data'

    @property
    def processed_file_names(self) -> List[str]:
        if isinstance(self.pre_transform, DFPipeline):
            return list(self.pre_transform.get_output_files().values())
        else:
            return 'data.pt'
        
    @property
    def raw_file_names_dictionary(self) -> Dict[str, str]:
        if isinstance(self.download_pipeline, DFPipeline):
            return self.download_pipeline.get_output_files()

    @property
    def processed_file_names_dictionary(self) -> Dict[str, str]:
        if isinstance(self.pre_transform, DFPipeline):
            return self.pre_transform.get_output_files()
        

    @property
    def raw_paths_dictionary(self) -> Dict[str, str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = self.raw_file_names_dictionary
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return {k: osp.join(self.raw_dir, f) for k, f in files.items()}

    @property
    def processed_paths_dictionary(self) -> Dict[str, str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        files = self.processed_file_names_dictionary
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return {k: osp.join(self.processed_dir, f) for k, f in files.items()}


    def download(self):
        # run the download pipeline
        self.download_pipeline({KEY_ROOT: self.root})


    def process(self):
        # run the preprocessing pipeline
        self.pre_transform({KEY_ROOT: self.root})


class SimplerToUndirected(BaseTransform):

    def __init__(self, reduce: str = "add"):
        self.reduce = reduce

    def __call__(self, data: Data) -> Data:

        data.edge_index, data.edge_attr = to_undirected(
            edge_index = data.edge_index,
            edge_attr =	 data.edge_attr,
            reduce =	 self.reduce
        )

        return data