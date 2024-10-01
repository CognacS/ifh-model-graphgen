from typing import Union, Dict, Optional, Any, Callable

import os.path as osp

import pytorch_lightning as pl
from torch_geometric.transforms import BaseTransform


from .dataset import GraphDataset
from .transforms import DFBaseTransform
from .dataloader import DataLoader
import src.data.pipelines as pplines

class GraphDataModule(pl.LightningDataModule):

    def __init__(
            self,
            root_dir: str,
            download_pl: Union[str, DFBaseTransform],
            preprocess_pl: Union[str, DFBaseTransform],
            runtimeprocess_pl: Optional[Union[str, DFBaseTransform, Dict[str, DFBaseTransform]]]=None,
            dataloader_config: Optional[Dict[str, Dict]]=None,
            download_pl_kwargs: Optional[Dict]=None,
            preprocess_pl_kwargs: Optional[Dict]=None
        ):

        super().__init__()
        self.root_dir = root_dir

        # set download pipeline
        if isinstance(download_pl, str):
            if download_pl_kwargs is None:
                download_pl_kwargs = {}
            self.download_pl = pplines.REGISTERED_DOWNLOAD_PIPELINES[download_pl](**download_pl_kwargs)
        else:
            self.download_pl = download_pl

        # set preprocess pipeline
        if isinstance(preprocess_pl, str):
            if preprocess_pl_kwargs is None:
                preprocess_pl_kwargs = {}
            self.preprocess_pl = pplines.REGISTERED_PREPROCESS_PIPELINES[preprocess_pl](**preprocess_pl_kwargs)
        else:
            self.preprocess_pl = preprocess_pl

        self.runtimeprocess_pl = runtimeprocess_pl
        if dataloader_config is None:
            self.dataloader_config = {}
        else:
            self.dataloader_config = dataloader_config

    
    def get_info(self, which_datasplit: str) -> Dict:
        
        temp_ds = GraphDataset(
            root =						self.root_dir,
            download_pipeline =			self.download_pl,
            preprocessing_transform =	self.preprocess_pl,
            dataset_split =				which_datasplit,
            only_load_info =			True
        )

        return temp_ds.info
    
    def get_filename(self, which_datasplit: str, file_key: str, include_path: bool=False) -> Any:
        
        temp_ds = GraphDataset(
            root =						self.root_dir,
            download_pipeline =			self.download_pl,
            preprocessing_transform =	self.preprocess_pl,
            dataset_split =				which_datasplit,
            only_load_info =			True
        )

        filename = temp_ds.processed_file_names_dictionary[file_key]

        if include_path:
            filename = osp.join(temp_ds.processed_dir, filename)

        return filename
    
    def load_file(self, which_datasplit: str, file_key: str, load_method):

        filepath = self.get_filename(which_datasplit, file_key, include_path=True)

        with open(filepath, 'r') as file:
            data = load_method(file)

        return data

    ############################################################################
    #                         DATA PREPARATION METHODS                         #
    ############################################################################

    def prepare_data(self):
        # download and preprocess
        # may get a warning for data splits, but that's expected
        GraphDataset(
            root = self.root_dir,
            download_pipeline = self.download_pl,
            preprocessing_transform = self.preprocess_pl
        )
        

    def setup(self, stage: str, disable_transform: bool = False):

        if not hasattr(self, 'datasets'):
            self.datasets = {}

        ###########################  TRAINING PHASE  ###########################
        if stage == 'fit' or stage == 'train':

            if 'train' not in self.datasets:
                self.datasets['train'] = self.__get_dataset('train', disable_transform)

            if 'valid' not in self.datasets:
                self.datasets['valid'] = self.__get_dataset('valid', disable_transform)

        ##########################  VALIDATION PHASE  ##########################
        elif stage == 'validate' or stage == 'valid':

            if 'valid' not in self.datasets:
                self.datasets['valid'] = self.__get_dataset('valid', disable_transform)

        #############################  TEST PHASE  #############################
        elif stage == 'test':

            if 'test' not in self.datasets:
                self.datasets['test'] = self.__get_dataset('test', disable_transform)

        else:
            raise NotImplementedError(f'Stage "{stage}" is not implemented!')
        


    def clear_datasets(self):
        self.datasets = {}


    ############################################################################
    #                            DATALOADER METHODS                            #
    ############################################################################

    def train_dataloader(self):
        return self.__get_dataloader('train')

    def val_dataloader(self):
        return self.__get_dataloader('valid')

    def test_dataloader(self):
        return self.__get_dataloader('test')
    
    def get_dataloader(self, which_datasplit: str):
        return self.__get_dataloader(which_datasplit)


    ############################################################################
    #                              HELPER METHODS                              #
    ############################################################################

    def __get_dataloader(self, which_datasplit: str) -> DataLoader:

        split_loader_config = get_config_datasplit(self.dataloader_config, which_datasplit)
        
        return DataLoader(
            dataset = self.datasets[which_datasplit],
            **split_loader_config
        )

    def __get_dataset(self, which_datasplit: str, disable_transform: bool = False) -> GraphDataset:

        if not disable_transform:
            if isinstance(self.runtimeprocess_pl, dict):
                runtimeprocess_pl = self.runtimeprocess_pl[which_datasplit]
            else:
                runtimeprocess_pl = self.runtimeprocess_pl

        else:
            runtimeprocess_pl = None

        return GraphDataset(
            root =                          self.root_dir,
            download_pipeline =             self.download_pl,
            preprocessing_transform =       self.preprocess_pl,
            runtimeprocessing_transform =   runtimeprocess_pl,
            dataset_split =                 which_datasplit
        )


def is_dict_of_dicts(d: Dict):
    return isinstance(next(iter(d.values())), dict)

def get_config_datasplit(config: Union[Dict, None], which_datasplit: str):
    if isinstance(config, dict):
        if is_dict_of_dicts(config):
            # case 1: config has different configs for each datasplit
            config = config[which_datasplit]
        else:
            # case 2: config is itself the config for all datasplits
            pass
    else:
        # case 3: there is no config, use default
        config = {}

    return config