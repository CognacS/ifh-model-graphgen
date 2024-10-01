from typing import Union, List, Dict, Callable, Optional, Tuple

import csv
import numpy as np
import torch

import random
import copy

from . import DFBaseTransform

def df_to_list(df: Union[str, List[str]]) -> List[str]:
    return [df] if isinstance(df, str) else df

def frac_to_int(fraction: Union[float, int], data_size: int) -> int:
    if isinstance(fraction, float):
        return int(fraction * data_size)
    elif isinstance(fraction, int):
        return fraction

class DFSplitTrainTestFromList(DFBaseTransform):
    def __init__(
            self,
            data_list_df: Union[str, List[str]],
            test_idx_df: str,
            train_df: Union[str, List[str]],
            test_df: Union[str, List[str]]
        ):
        self.test_idx_df = test_idx_df

        self.data_list_df = df_to_list(data_list_df)
        self.train_df = df_to_list(train_df)
        self.test_df = df_to_list(test_df)

    
    def __call__(self, data: Dict) -> Dict:

        # don't worry, its a shallow copy!
        # no copy of heavy data is done
        data_lists = [data[df] for df in self.data_list_df]
        data_size = len(data_lists[0])

        # generate boolean mask for test set
        is_test_arr = np.zeros(data_size, dtype=bool)
        test_idx = data[self.test_idx_df]
        is_test_arr[np.array(test_idx, dtype=int)] = True

        for train_df, test_df, data_list in zip(self.train_df, self.test_df, data_lists):
            # initialize lists
            data[train_df] = []
            data[test_df] = []
            # split data into train and test
            for i, is_test in enumerate(is_test_arr):
                if is_test:
                    data[test_df].append(data_list[i])
                else:
                    data[train_df].append(data_list[i])

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return self.data_list_df +[self.test_idx_df]
    
    @property
    def output_df_list(self) -> List[str]:
        sets = self.train_df + self.test_df
        return sets
    


class DFSplitTrainTestValid(DFBaseTransform):
    def __init__(
            self,
            data_list_df: Union[str, List[str]],
            test_fraction: Union[float, int],
            train_df: Union[str, List[str]],
            test_df: Union[str, List[str]],
            valid_fraction: Optional[Union[float, int]]=None,
            valid_df: Optional[Union[str, List[str]]]=None,
        ):
        self.test_fraction = test_fraction

        self.data_list_df = df_to_list(data_list_df)
        self.train_df = df_to_list(train_df)
        self.test_df = df_to_list(test_df)

        self.valid_fraction = valid_fraction
        self.valid_df = df_to_list(valid_df)

    
    def __call__(self, data: Dict) -> Dict:

        # don't worry, its a shallow copy!
        # no copy of heavy data is done
        data_lists = [data[df] for df in self.data_list_df]
        data_size = len(data_lists[0])

        # generate random shuffling
        idx = np.arange(data_size)
        np.random.shuffle(idx)

        # test sets extraction
        test_size = frac_to_int(self.test_fraction, data_size)
        # for each full dataset, partition test set and put it into the
        # corresponding test_df
        for test_df, data_list in zip(self.test_df, data_lists):
            data[test_df] = [data_list[i] for i in idx[:test_size]]

        # update already used data
        in_use_size = test_size

        if self.valid_fraction is not None and self.valid_df is not None:
            # validation sets extraction
            valid_size = frac_to_int(self.valid_fraction, data_size)
            # for each full dataset, partition valid set and put it into the
            # corresponding valid_df
            for valid_df, data_list in zip(self.valid_df, data_lists):
                data[valid_df] = [data_list[i] for i in idx[in_use_size:in_use_size+valid_size]]

            # update already used data
            in_use_size = in_use_size + valid_size

        # train sets extraction
        for train_df, data_list in zip(self.train_df, data_lists):
            data[train_df] = [data_list[i] for i in idx[in_use_size:]]

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return self.data_list_df
    
    @property
    def output_df_list(self) -> List[str]:
        sets = self.train_df + self.test_df
        if self.valid_fraction is not None and self.valid_df is not None:
            sets = sets + self.valid_df
        return sets
    
    def args_repr(self) -> str:
        test_label = ''
        if isinstance(self.test_fraction, float):
            test_label = 'test_fraction'
        elif isinstance(self.test_fraction, int):
            test_label = 'test_size'
        ret = f'{test_label}={self.test_fraction}'

        if self.valid_fraction is not None and self.valid_df is not None:
            valid_label = ''
            if isinstance(self.valid_fraction, float):
                valid_label = 'valid_fraction'
            elif isinstance(self.valid_fraction, int):
                valid_label = 'valid_size'
            ret += f'\n{valid_label}={self.valid_fraction}'
        return ret
    

class DFForEachSplit(DFBaseTransform):

    def __init__(
            self,
            split_names: List[str],
            transform: DFBaseTransform,
            dont_split: bool=True,
        ):
        self.split_names = split_names
        self.transform = transform
        self.dont_split = dont_split
        
    
    def __call__(self, data: Dict) -> Dict:
        split_names = [''] if self.dont_split else self.split_names

        # for each split
        for split_name in split_names:
            transform = copy.deepcopy(self.transform)
            transform.format_dfs({'split': split_name})
            data = transform(data)
        
        return data

    @property
    def input_df_list(self) -> List[str]:
        return self.transform.input_df_list
    @property
    def output_df_list(self) -> List[str]:
        return self.transform.output_df_list
    
    def args_repr(self) -> str:
        return (
            f'split_names={self.split_names}\n'
            f'dont_split={self.dont_split}\n'
            f'transform={self.transform}'
        )
    

def get_outputdfs_files_splits(
        split_test: bool, split_valid: bool,
        outdf_file_ids_exts: List[Tuple[str, str, str]],
        file_format_string: str = 'dataset{split}_{file_id}.{ext}'
    ) -> Dict[str, str]:

    # SPLIT CASE
    # each output datafield is associated with its file
    # e.g. 'dataset_path_train' -> 'qm9_train_data.pt'
    # e.g. 'smiles_file_test' -> 'qm9_test_smiles.json'
    # NON-SPLIT CASE
    # each output datafield is associated with its file
    # e.g. 'dataset_path' -> 'qm9_data.pt'

    if split_test:
        splits = ['_train', '_test']
        if split_valid:
            splits.append('_valid')
    else:
        splits = ['']

    out_files = {}
    # names of output datafields
    for outdf, file_id, ext in outdf_file_ids_exts:
        for split in splits:
            out_files[f'{outdf}{split}'] = file_format_string.format(
                split=split,
                file_id=file_id,
                ext=ext
            )

    return splits, out_files