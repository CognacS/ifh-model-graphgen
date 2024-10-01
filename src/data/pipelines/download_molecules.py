from torch_geometric.data import extract_zip

import src.data.transforms as base_t
from src.data.transforms import (
    downloads as down_t
)
import src.data.dataset as ds

def qm9_download_pipeline() -> base_t.DFPipeline:
    
    pipeline = base_t.DFPipeline(

        output_files = {
            'mols_file': 'gdb9.sdf',
            'targets_csv_file': 'gdb9.sdf.csv',
            'skip_csv_file': '3195404'
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
                    'qm9_data': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip',
                    'qm9_readme': 'https://ndownloader.figshare.com/files/3195404'
                }
            ),
            down_t.DFExtract(
                extr_method=extract_zip,
                datafield='qm9_data',
                extract_path_df='raw_path'
            )
        ]
    )

    return pipeline


class ZincDownloadPipeline:

    def __init__(self, which_zinc: str):
        self.which_zinc = which_zinc

    def __call__(
            self
        ) -> base_t.DFPipeline:
        
        pipeline = base_t.DFPipeline(

            output_files = {
                'smiles_file': f'zinc{self.which_zinc}.csv',
                'valid_idx_file': f'valid_idx_zinc{self.which_zinc}.json',
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
                        'smiles': f'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/zinc{self.which_zinc}.csv',
                        'valid_idx': f'https://raw.githubusercontent.com/divelab/DIG_storage/main/ggraph/valid_idx_zinc{self.which_zinc}.json'
                    }
                )
            ]
        )

        return pipeline