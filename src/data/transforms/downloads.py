from typing import Union, List, Dict, Callable, Optional

import os

from torch_geometric.data import (
    download_url
)

from . import DFBaseTransform


DF_RAW_FOLDER = 'raw_folder'
DF_RAW_FILE_NAMES = 'raw_file_names'

class DownloadPipelineException(Exception):
    pass


class DFDownloadFromURL(DFBaseTransform):
            
    def __init__(
            self,
            url: Union[Dict[str, str], List[str], str],
            download_to_df: Optional[str]=None,
        ):
        self.download_to_df = download_to_df

        if isinstance(url, str):
            urls = {'download_path': url}
        elif isinstance(url, list):
            urls = {f'download_path_{i}': url for i, url in enumerate(url)}
        elif isinstance(url, dict):
            urls = url
        else:
            raise DownloadPipelineException(f'Url in {self.__class__.__name__} object has an invalid type. Should be: str, list, dict. Found: {type(url)}')
        
        self.urls = urls

    
    def __call__(self, data: Dict) -> Dict:

        raw_folder = data[self.download_to_df]

        for name, url in self.urls.items():
            file_path = download_url(url, raw_folder)
            data[name] = file_path

        return data
    
    @property
    def output_df_list(self) -> List[str]:
        return [self.download_to_df]

    def args_repr(self) -> str:
        urls_string = '\n'.join(list(self.urls.values()))
        return (
            f'urls=[\n{urls_string}\n]'
        )
    
    
KEY_EXTRACTED = '_extracted'

class DFExtract(DFBaseTransform):

    def __init__(
            self,
            extr_method: Callable[[str, str], None],
            datafield: Union[List[str], str],
            extract_path_df: Optional[str]=None,
        ):
        self.extract_path_df = extract_path_df
        self.extr_method = extr_method

        if isinstance(datafield, str):
            datafields = [datafield]
        elif isinstance(datafield, List):
            datafields = datafield
        else:
            raise DownloadPipelineException(f'datafield in {self.__class__.__name__} object has an invalid type. Should be: str, list. Found: {type(datafield)}')
        
        self.datafields = datafields

    def _curr_files(self, folder: str) -> List[str]:
        return os.listdir(folder)

    def _new_files(self, folder: str, prev_files: List[str]) -> Union[List[str], List[str]]:
        curr_files = self._curr_files(folder)
        new_files = [
            f for f in curr_files
            if f not in prev_files
        ]
        return new_files, curr_files

    def __call__(self, data: Dict) -> Dict:

        raw_folder = data[self.extract_path_df]

        prev_files = self._curr_files(raw_folder)

        for df in self.datafields:
            try:
                # get compressed file paths in data
                compr_file_path = data[df]
                # extract the file using the selected method
                self.extr_method(compr_file_path, raw_folder)

                # find new files and current files
                # to be used later
                new_files, prev_files = self._new_files(raw_folder, prev_files)

                # set the extracted files in data
                data[df + KEY_EXTRACTED] = new_files
            except Exception as ex:
                print('Something went wrong during the extraction of '\
                    +f'"{df}" at path "{data[df]}". Skipping '\
                    +f'this file. The thrown exception is: "{ex}"')
                
        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.extract_path_df] + self.datafields
    @property
    def output_df_list(self) -> List[str]:
        return [df + KEY_EXTRACTED for df in self.datafields]

    def args_repr(self) -> str:
        urls_string = ', '.join(self.datafields)
        return (
            f'urls=[\n{urls_string}\n]'
        )
