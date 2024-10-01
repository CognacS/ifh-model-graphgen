from typing import Union, List, Dict, Callable, Optional

import csv
import numpy as np
import pandas as pd
import torch
import json

from . import DFBaseTransform

DF_PROCESSED_FOLDER = 'proc_folder'
DF_PROCESSED_FILE_NAMES = 'proc_file_names'

class PreprocessingException(Exception):
    pass


CSV_HEADER = 'csv_header'
CSV_SAMPLE_NAMES = 'csv_sample_names'
CSV_TABLE = 'csv_table'

class DFReadCSV(DFBaseTransform):

    def __init__(
            self,
            datafield: str,
            table_df: str,
            astype: str='numpy',
            columns_df: Optional[str]=None,
            index_df: Optional[str]=None,
            csvfields: Optional[List[str]]=None,
            **csvparams
        ):

        self.datafield = datafield
        self.columns_df = columns_df
        self.index_df = index_df
        self.table_df = table_df
        self.astype = astype
        self.csvfields = csvfields
        self.csvparams = csvparams

    def _header_indexing(self, full_header):
        h2i = {h: i for i, h in enumerate(full_header)}
        idx = [h2i[h] for h in self.csvfields]
        return idx
    
    def __call__(self, data: Dict) -> Dict:

        if self.astype == 'numpy':
            with open(data[self.datafield], 'r') as csvfile:

                csvrows = np.array(list(csv.reader(
                    csvfile,
                    **self.csvparams
                )))

            # get csv header
            # remove first column with names
            full_header = csvrows[0, 1:]

            # get indexing from full header to
            # selected header
            if self.csvfields is not None:
                header_idx = self._header_indexing(full_header)
                full_header = self.csvfields

            # get rest of table
            table = np.array(csvrows[1:, :])

            # get names of each sample
            sample_names = table[:, 0]

            # convert to a torch tensor
            table = torch.from_numpy(table[:, 1:].astype(float))

            # select only required fields
            if self.csvfields is not None:
                table = table[:, header_idx]

        elif self.astype == 'pandas':

            table = pd.read_csv(
                data[self.datafield],
                **self.csvparams
            )

            # get csv header
            full_header = table.columns

            # get indexing from full header to
            # selected header
            if self.csvfields is not None:
                table = table[self.csvfields]
                full_header = self.csvfields

            # get names of each sample
            sample_names = table.index


        data[self.table_df] = table
        if self.columns_df is not None:
            data[self.columns_df] = full_header
        if self.index_df is not None:
            data[self.index_df] = sample_names

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.datafield]
    @property
    def output_df_list(self) -> List[str]:
        return [self.columns_df, self.index_df, self.table_df]
    
    def args_repr(self) -> str:
        if self.csvfields is not None:
            fields = str(self.csvfields)
        else:
            fields = 'all those found'

        return (
            f'csvfields = {fields}\n'
            f'other parameters = {self.csvparams}'
        )
    

class DFReadJSON(DFBaseTransform):

    def __init__(
            self,
            datafield: str,
            output_df: str
        ):

        self.datafield = datafield
        self.output_df = output_df
    
    def __call__(self, data: Dict) -> Dict:

        with open(data[self.datafield], 'r') as jsonfile:
            data[self.output_df] = json.load(jsonfile)

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.datafield]
    @property
    def output_df_list(self) -> List[str]:
        return [self.output_df]
    

class DFSelectColumnsDataframe(DFBaseTransform):

    def __init__(
            self,
            table_df: str,
            result_df: str,
            selected_columns: List[str]|List[int]|slice|str|int
        ):

        self.table_df = table_df
        self.result_df = result_df
        if isinstance(selected_columns, (str, int)):
            selected_columns = [selected_columns]
        self.selected_columns = selected_columns

        # get column type
        col_type = type(selected_columns)
        if col_type is list:
            col_type = type(selected_columns[0])
        # select method for indexing
        self._loc = col_type is str


    def __call__(self, data: Dict) -> Dict:

        table = data[self.table_df]

        if self._loc:
            table = table.loc[:, self.selected_columns]
        else:
            table = table.iloc[:, self.selected_columns]

        res = table.to_numpy().squeeze()

        data[self.result_df] = res

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.table_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.result_df]
    
    def args_repr(self) -> str:
        return (
            f'selected_columns = {self.selected_columns}'
        )