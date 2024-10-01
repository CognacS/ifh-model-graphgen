from typing import Callable, List, Dict, Optional, Any
from abc import ABC

import os, shutil
import os.path as osp
from tqdm import tqdm
from textwrap import indent
from itertools import chain
import json
import pickle

################################################################################
#                           BASE DATAFLOW TRANSFORMS                           #
################################################################################
# these transformations are highly inspired by:
# https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html
# but are meant for future extensions and customization. The main change is
# that data is contained in a dictionary

IND_CHAR = '\t'
DATAFIELD_SUFFIXES = ['_df', '_datafield']

class DFBaseTransform(ABC):
    
    def __call__(self, data: Dict) -> Dict:
        raise NotImplementedError
    
    #################  TO DEFINE  #################
    @property
    def input_df_list(self) -> List[str]:
        return [str(None)]
    @property
    def output_df_list(self) -> List[str]:
        return [str(None)]
    
    def args_repr(self) -> str:
        return ''
    
    ################  DF HANDLING  ###############

    def format_dfs(self, values: Dict):
        # for all attributes of the object with df
        # in the name, format its value with the
        # given values
        for attr_name in dir(self):
            if any([attr_name.endswith(suffix) for suffix in DATAFIELD_SUFFIXES]):
                attr = getattr(self, attr_name)
                if isinstance(attr, str):
                    setattr(self, attr_name, attr.format(**values))
    
    
    ##########  REPRESENTATION STRINGS  ##########
    def dataflow(self) -> str:
        try:
            input_args = ', '.join(list(set([str(df) for df in self.input_df_list])))
            output_args = ', '.join(list(set([str(df) for df in self.output_df_list])))
        except TypeError as ex:
            print(self.input_df_list)
            print(self.output_df_list)
            raise ex
        return f'{input_args} -> {output_args}'
    
    def __repr__(self) -> str:
        
        content = indent(
            f'{self.dataflow()}\n{self.args_repr()}',
            prefix=IND_CHAR
        )

        return (
            f'{self.__class__.__name__}(\n'
            f'{content}\n'
            f')'
        )


class DFCustomTransform(DFBaseTransform):

    def __init__(
            self,
            src_datafield: str,
            dst_datafield: str,
            free_transform: Callable
        ):
        self.src_datafield = src_datafield
        self.dst_datafield = dst_datafield
        self.free_transform = free_transform

    def __call__(
            self,
            data: Dict
        ) -> Dict:

        if self.src_datafield is not None:
            d = (data[self.src_datafield], )
        else:
            d = ()
        out = self.free_transform(*d)
        data[self.dst_datafield] = out

        return data	
    @property
    def input_df_list(self) -> List[str]:
        return [self.src_datafield]
    @property
    def output_df_list(self) -> List[str]:
        return [self.dst_datafield]
    
    def args_repr(self) -> str:
        return f'free_func={self.free_transform.__name__}'


class DFCompose(DFBaseTransform):

    def __init__(self, transforms: List[DFBaseTransform]):
        # check all transforms and remove Nones
        # from the list
        self.transforms = [
            transform for transform in transforms
            if transform is not None
        ]


    def __call__(
            self,
            data: Dict,
        ) -> Dict:
        
        for transform in self.transforms:
            data = transform(data)
        
        return data
    
    def format_dfs(self, values: Dict):
        super().format_dfs(values)
        for transform in self.transforms:
            transform.format_dfs(values)

    @property
    def input_df_list(self) -> List[str]:
        return list(set(chain(*[t.input_df_list for t in self.transforms])))
    @property
    def output_df_list(self) -> List[str]:
        return list(set(chain(*[t.output_df_list for t in self.transforms])))
    
    def args_repr(self) -> str:
        args = [f'{transform}' for transform in self.transforms]
        args_list = ',\n'.join(args)
        return f'transforms=[\n{args_list}\n]'



class DFPipeline(DFCompose):

    def __init__(
            self,
            transforms: List[DFBaseTransform],
            input_files: Optional[Dict]=None,
            output_files: Optional[Dict]=None,
        ):
        super().__init__(transforms=transforms)
        self.input_files = input_files
        self.output_files = output_files


    def get_input_files(self) -> Dict[str, str]:
        return self.input_files
    def get_output_files(self) -> Dict[str, str]:
        return self.output_files

    def __call__(
            self,
            data: Dict=None
        ) -> Dict:

        if data is None:
            data = {}

        if self.input_files is not None:
            data.update(self.input_files)

        if self.output_files is not None:
            data.update(self.output_files)

        # call Compose, that is, compute all transformations
        data = super().__call__(data)

        return data
    

    def args_repr(self) -> str:
        return (
            f'input_files={self.input_files}\n'
            f'output_files={self.output_files}\n'
            f'{super().args_repr()}'
        )



class DFAddDatafield(DFBaseTransform):

    def __init__(self, datafield: str, value: Any):
        self.datafield = datafield
        self.value = value


    def __call__(
            self,
            data: Dict,
        ) -> Dict:
        
        data[self.datafield] = self.value
        
        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [f'"{self.value}"']
    @property
    def output_df_list(self) -> List[str]:
        return [self.datafield]

class DFRenameDatafield(DFBaseTransform):
    
        def __init__(self, src_df: str, dst_df: str):
            self.src_df = src_df
            self.dst_df = dst_df
    
        def __call__(
                self,
                data: Dict,
            ) -> Dict:
            
            data[self.dst_df] = data[self.src_df]
            data.pop(self.src_df, None)
            
            return data
        
        @property
        def input_df_list(self) -> List[str]:
            return [self.src_df]
        @property
        def output_df_list(self) -> List[str]:
            return [self.dst_df]

class DFPrintDatafield(DFBaseTransform):
    
        def __init__(self, datafield: str):
            self.datafield = datafield
    
    
        def __call__(
                self,
                data: Dict,
            ) -> Dict:
            
            print(f'{self.datafield}: {data[self.datafield]}')
            
            return data
        
        @property
        def input_df_list(self) -> List[str]:
            return [self.datafield]


class DFUnpackDatafield(DFBaseTransform):
    
    def __init__(self, src_df: str, dst_dfs: List[str]):
        self.src_df = src_df
        self.dst_dfs = dst_dfs

    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        tuple_data = data[self.src_df]
        
        data.update({
            dst_df: tuple_data[i]
            for i, dst_df in enumerate(self.dst_dfs)
            if dst_df is not None
        })
        
        return data


class DFIterateOver(DFBaseTransform):

    def __init__(
            self,
            datafield: str,
            iter_idx_df: str,
            iter_elem_df: str,
            transform: DFBaseTransform
        ):

        self.datafield = datafield
        self.iter_idx_df = iter_idx_df
        self.iter_elem_df = iter_elem_df
        self.transform = transform

    def __call__(
            self,
            data: Dict
        ) -> Dict:

        # get iterable to iterate over
        iterable = data[self.datafield]

        # temporarily introduce iteration fields
        # into data
        data.update({
            self.iter_elem_df: None,
            self.iter_idx_df: None
        })

        # iterate
        for idx, elem in enumerate(tqdm(iterable)):

            # update elements
            data[self.iter_elem_df] = elem
            data[self.iter_idx_df] = idx

            # get output data from the transform
            out_data: Dict = self.transform(
                data
            )

        # remove temporary datafields from data
        data.pop(self.iter_elem_df, None),
        data.pop(self.iter_idx_df, None)

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.datafield]
    
    def args_repr(self) -> str:
        return (
            f'iterating on element = {self.iter_elem_df}\n'
            f'iteration index = {self.iter_idx_df}\n'
            f'transforms={self.transform}'
        )
    

from concurrent.futures import ProcessPoolExecutor

class DFMultiProcessMap(DFBaseTransform):

    def __init__(
            self,
            input_list_df: str,
            iter_idx_df: str,
            iter_elem_df: str,
            iter_ret_dfs: str|List[str],
            output_list_dfs: str|List[str],
            transform: DFBaseTransform,
            num_workers: int=1
        ):

        self.input_list_df = input_list_df
        self.iter_idx_df = iter_idx_df
        self.iter_elem_df = iter_elem_df
        self.iter_ret_dfs = iter_ret_dfs if isinstance(iter_ret_dfs, list) else [iter_ret_dfs]
        self.output_list_dfs = output_list_dfs if isinstance(output_list_dfs, list) else [output_list_dfs]
        self.transform = transform
        self.num_workers = num_workers

    def __call__(
            self,
            data: Dict
        ) -> Dict:

        # get iterable to iterate over
        iterable = data[self.input_list_df]

        # temporarily introduce iteration fields
        # into data
        data.update({
            self.iter_elem_df: None,
            self.iter_idx_df: None
        })
        data.update({
            df: None for df in self.iter_ret_dfs
        })

        # define the job
        def task(idx_elem):
            idx, elem = idx_elem
            # update elements
            data[self.iter_elem_df] = elem
            data[self.iter_idx_df] = idx

            # get output data from the transform
            out_data: Dict = self.transform(
                data
            )

            if isinstance(self.iter_ret_dfs, str):
                return out_data[self.iter_ret_dfs]
            else:
                return tuple(out_data[df] for df in self.iter_ret_dfs)

        total_iters = len(iterable)
        iterable = enumerate(iterable) # add also the iteration index

        ##################  JOB EXECUTION WITHOUT MULTIPROC  ###################
        if self.num_workers == 0:
            out_data_list = list(tqdm(map(task, iterable), total=total_iters))

        ####################  JOB EXECUTION WITH MULTIPROC  ####################
        else:            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                out_data_list = list(tqdm(executor.map(task, iterable), total=total_iters))

        #######################  OUTPUT LIST FORMATTING  #######################
        if isinstance(self.output_list_dfs, str):
            data[self.output_list_dfs] = out_data_list
        else:
            out_data_list = tuple(zip(*out_data_list))
            for out_df, out_data in zip(self.output_list_dfs, out_data_list):
                data[out_df] = out_data

        # remove temporary datafields from data
        data.pop(self.iter_elem_df, None)
        data.pop(self.iter_idx_df, None)
        for df in self.iter_ret_dfs:
            data.pop(df, None)

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.datafield]
    
    def args_repr(self) -> str:
        return (
            f'iterating on element = {self.iter_elem_df}\n'
            f'iteration index = {self.iter_idx_df}\n'
            f'transforms={self.transform}'
        )
    

class DFConditional(DFBaseTransform):

    def __init__(
            self,
            condition: Callable[[Dict], bool],
            transform: DFBaseTransform
        ):

        self.condition = condition
        self.transform = transform

    def __call__(
            self,
            data: Dict
        ) -> Dict:

        if self.condition is None or self.condition(data):
            data = self.transform(data)

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return self.transform.input_df_list
    @property
    def output_df_list(self) -> List[str]:
        return self.transform.output_df_list

    def args_repr(self) -> str:
        return (
            f'condition={self.condition}\n'
            f'transform={self.transform}'
        )


class DFAppendElement(DFBaseTransform):

    def __init__(
            self,
            elem_df: str,
            datalist_df: str
        ):

        self.elem_df = elem_df
        self.datalist_df = datalist_df

    def __call__(
            self,
            data: Dict
        ) -> Dict:

        # extract element from the list from data
        data[self.datalist_df].append(data[self.elem_df])

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.datalist_df, self.elem_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.datalist_df]


class DFIndexElement(DFBaseTransform):

    def __init__(
            self,
            datalist_df: str,
            index_df: str,
            result_df: str,
            index_input: bool=True # if False, output is indexed
        ):

        self.datalist_df = datalist_df
        self.index_df = index_df
        self.result_df = result_df
        self.index_input = index_input

    def __call__(
            self,
            data: Dict
        ) -> Dict:

        # get index
        idx = data[self.index_df]

        if self.index_input:
            # extract element from the list from data
            data[self.result_df] = data[self.datalist_df][idx]
        else:
            # write element at the index in the list of data
            data[self.datalist_df][idx] = data[self.result_df]

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.datalist_df, self.index_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.result_df]


class DFSPecializePath(DFBaseTransform):
    def __init__(
            self,
            root_dfs: List[str],
            result_df: str
        ):

        self.root_dfs = root_dfs
        self.result_df = result_df


    def __call__(self, data: Dict) -> Dict:

        concatenated_root = osp.join(*[
            data[root_df] for root_df in self.root_dfs
        ])

        data[self.result_df] = concatenated_root

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return self.root_dfs
    @property
    def output_df_list(self) -> List[str]:
        return [self.result_df]
    

class DFLocalizeFiles(DFBaseTransform):

    def __init__(
            self,
            root_df: str,
            file_dfs: List[str]|str
        ):

        self.root_df = root_df
        if isinstance(file_dfs, str):
            file_dfs = [file_dfs]
        self.file_dfs = file_dfs

    def __call__(self, data: Dict) -> Dict:

        root = data[self.root_df]

        for file_df in self.file_dfs:
            data[file_df] = osp.join(
                root, data[file_df]
            )

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.root_df]
    @property
    def output_df_list(self) -> List[str]:
        return self.file_dfs


class DFCreateFolder(DFBaseTransform):
    def __init__(
            self,
            destination_df: str
        ):

        self.destination_df = destination_df

    
    def __call__(self, data: Dict) -> Dict:

        folder = data[self.destination_df]

        create_folder(folder)
        
        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.destination_df]


class DFClearFolder(DFBaseTransform):
    def __init__(
            self,
            to_clear_df: str
        ):
        self.to_clear_df = to_clear_df

    
    def __call__(self, data: Dict) -> Dict:

        folder = data[self.to_clear_df]
        remove_all_files_in_dir(folder)

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.to_clear_df]
    

class DFCopyFile(DFBaseTransform):
    def __init__(
            self,
            source_df: str,
            destination_df: str,
            rename: Optional[str]=None
        ):

        self.source_df = source_df
        self.destination_df = destination_df
        self.rename = rename

    
    def __call__(self, data: Dict) -> Dict:

        source = data[self.source_df]
        destination = data[self.destination_df]

        shutil.copyfile(source, destination)

        if self.rename is not None and isinstance(self.rename, str):
            os.rename(destination, self.rename)

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.source_df, self.destination_df]

    

class DFSaveToFile(DFBaseTransform):
    def __init__(
            self,
            save_path_df,
            datafields: List[str]|str,
            save_file_method: str = 'json'
        ):
        self.save_path_df = save_path_df
        self.datafields = datafields
        self.save_file_method = save_file_method
        if self.save_file_method not in ['json', 'pickle']:
            raise ValueError(f'Unknown save file format {self.save_file_method}')

    
    def __call__(self, data: Dict) -> Dict:

        if isinstance(self.datafields, str):
            datastructure = data[self.datafields]
        elif len(self.datafields) == 1:
            datastructure = data[self.datafields[0]]
        else:
            datastructure = {df: data[df] for df in self.datafields}

        # save datastructure as a json file (for readability)
        write_how = 'w' if self.save_file_method == 'json' else 'wb'

        with open(data[self.save_path_df], write_how) as save_file:
            if self.save_file_method == 'json':
                json.dump(
                    datastructure, save_file,
                    indent='\t'
                )
            elif self.save_file_method == 'pickle':
                pickle.dump(
                    datastructure, save_file
                )

        return data
    
    def format_dfs(self, values: Dict):
        super().format_dfs(values)
        self.datafields = [df.format(**values) for df in self.datafields]
    
    @property
    def input_df_list(self) -> List[str]:
        return self.datafields
    
################################################################################
#                          PIPELINE FORMATION FUNCTIONS                        #
################################################################################

def guarded_include(if_: bool, do_: DFBaseTransform) -> DFBaseTransform|None:
    if if_:
        return do_
    else:
        return None


################################################################################
#                               UTILITY FUNCTIONS                              #
################################################################################


def create_folder(folder: str):

    # check existance
    if not os.path.isdir(folder):
        # create directory
        os.makedirs(folder)

def remove_all_files_in_dir(folder: str):

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))