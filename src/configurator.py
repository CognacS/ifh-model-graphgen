from typing import Tuple, Dict, Callable, List, Union

# import python garbage collector
import gc

# logging utilities
from python_log_indenter import IndentedLoggerAdapter
import logging

# path utilities
import os.path as osp
from pathlib import Path
import glob

# configuration utilities
from omegaconf import OmegaConf, DictConfig

# experiment tracking utilities
import wandb

# data storing utilities
import json
import pickle as pkl

# miscellanous utilities
from datetime import datetime
from copy import deepcopy

# torch utilities
import torch
import torch.nn as nn

# pytorch lightning utilities
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import AdvancedProfiler

# runtime transforms
from torch_geometric.transforms import Compose
from src.datatypes.sparse import MyToUndirected

# datamodule utilities
import src.data.pipelines as ppl
from src.data.datamodule import GraphDataModule

# post transformations
from src.data.simple_transforms.molecular import GraphToMoleculeConverter

# model utilities
from src.models.generator import InsertDenoiseHaltModel
from src.models.models import is_model_name
from src.noise.data_transform.subgraph_sampler import SubgraphSampler
from src.noise.data_transform.subsequence_sampler import SubsequenceSampler, resolve_collater
from src.utils.modular_early_stopping import ModularEarlyStopping
from src.utils.modular_checkpointing import ModularModelCheckpoint, PREFIX_MODULE

# metrics utilities
from src.metrics.sampling import SamplingMetricsHandler
import src.metrics.metrics as m_list

# for RNG seeding purposes
import os
import random
import numpy as np

DATASETS_ROOT = 'datasets'
CHECKPOINT_PATH = 'checkpoints'
ALLOWED_EXECUTION_MODES = ['train', 'gen', 'eval', 'train+eval', 'validate_hparams']

# from https://github.com/Lightning-AI/lightning/issues/1565
def seed_everything(seed=191117):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision('medium')


class ContextException(Exception):
    """Exception raised for errors in the context.
    """
    pass


class PrepareData:
    """transform for preparing data for removal process"""
    def __init__(self, removal_process):

        self.removal_process = removal_process

    def __call__(self, batch):
            
        self.removal_process.prepare_data(datapoint=batch)
        return batch


class RunContext:

    def __init__(self):
        self.wandb_active = False

    
    @classmethod
    def from_config(cls, cfg: DictConfig) -> 'RunContext':
        """Factory method for creating a run context from a configuration.
        (see the configure(cfg) method for more details)

        Parameters
        ----------
        cfg : DictConfig
            full configuration of the run, usually coming from hydra

        Returns
        -------
        RunContext
            run context created from the configuration
        """

        context = cls()
        context.configure(cfg)
        return context


    def configure(self, cfg: DictConfig):
        """Method for configuring the run context. This will create the experiment
        directory, the datamodule, the model, the trainer, etc.

        Parameters
        ----------
        cfg : DictConfig
            full configuration of the run, usually coming from hydra
        """

        # preprocess configuration
        cfg = preprocess_config(cfg)

        # initialize logger
        self.logger = self.initialize_logger('context', cfg['verbosity'])

        # store execution arguments
        self.logger.info(f'Reading execution arguments...').push().add()
        self._configure_execution_args(cfg)
        self.logger.pop().info(f'Execution arguments read with success')

        # set seed
        self.logger.info(f'Setting seed to {self.seed}...').push().add()
        seed_everything(self.seed)
        self.logger.pop().info(f'Seed set with success')

        # check that context parameters are valid
        self.logger.info(f'Running checks on the context...').push().add()
        self.validate_context()
        self.logger.pop().info(f'Context validated with success')

        # checkpoints, run ids, etc..., are all contained in the run directory
        self.logger.info(f'Setting up run directory, version, id...').push().add()
        self.run_directory, self.version, self.run_id = self._setup_run_directory(self.config_name)
        self.logger.pop().info(f'Run directory: {self.run_directory}')
        self.logger.info(f'Run version: {self.version}')
        self.logger.info(f'Run id: {self.run_id}')

        # configuring datamodule
        self.logger.info(f'Configuring and loading datamodule...').push().add()
        self.datamodule = self._configure_datamodule(cfg['data'], cfg['model']['removal'])
        self.dataset_info = self.datamodule.get_info('train')
        self.logger.pop().info(f'Datamodule configured with success')
        
        # configuring model
        self.logger.info(f'Configuring and loading model...').push().add()
        datatype = ppl.REGISTERED_DATATYPES[cfg['data']['dataset']['name']]
        self.model = self._configure_model(datatype, cfg['model'], cfg['run']['training'], cfg['metric'], self.datamodule, self.dataset_info)
        self.logger.pop().info(f'Model configured with success')

        # configuring trainer
        self.logger.info(f'Configuring trainer...').push().add()
        self.trainer = self._configure_trainer(cfg['run'], cfg['platform'])
        self.logger.pop().info(f'Trainer configured with success')

        if self.enable_ckp and not self.load_ckp:
            self.logger.info(f'Creating run directory...')
            self.run_directory.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'Configuration completed')


    def execute(self):
        """Method for executing the run context. Depending on the execution mode,
        this will train the model, generate graphs, evaluate a model, etc.

        Returns
        -------
        results : Dict
            dictionary containing key-value pairs with keys being a metric name
            and values being the corresponding metric value. Will be returned only
            if the execution mode is 'validate_hparams' or 'eval' or 'train+eval'
        """

        results = {}

        # TRAINING: called for training a model
        # no evaluation is performed
        if self.mode == 'train':
            self.model.disable_generation()
            self.fit()
            self.model.enable_generation()

        # GENERATION: called for generating graphs
        # requires a pre-trained model
        # will produce a file with the generated graphs
        # by default, the file is stored in the run directory
        elif self.mode == 'gen':
            if not self.load_ckp:
                raise ContextException(f'To evaluate, must resume a checkpoint. Do it by setting load_ckp')
            ####### generate graphs #######
            kwargs = {}
            if 'how_many' in self.cfg:
                kwargs['how_many'] = self.cfg['how_many']
            graphs = self.generate(**kwargs)
            ######## store graphs ########
            kwargs = {}
            if 'gen_path' in self.cfg:
                kwargs['gen_path'] = self.cfg['gen_path']
            self.store_graphs(graphs, **kwargs)

        # VALIDATE HPARAMS: called during hyperparameter optimization
        # trains the model and returns the validation loss to be used by the optimizer
        elif self.mode == 'validate_hparams':
            self.model.disable_generation() # disable generation for hparams search
            self.fit()
            results = self.evaluate_best(validation=True)
            
        # EVALUATION: called for evaluating a model on the test set
        # requires a pre-trained model
        elif self.mode == 'eval':
            if not self.load_ckp:
                raise ContextException(f'To evaluate, must resume a checkpoint. Do it by setting load_ckp')
            results = self.evaluate_best(validation=False)

        # TRAIN+EVALUATION: called for evaluating a configuration on the test set
        # trains the model and returns the test loss
        # this is intended to be executed for the final evaluation on many seeds
        elif self.mode == 'train+eval':
            self.model.disable_generation()
            self.fit()
            self.model.enable_generation()
            results = self.evaluate_best(validation=False)

        # UNKNOWN MODE: raise an exception
        else:
            raise ContextException(f'Unknown mode {self.mode}')
        
        return results
    

    def __call__(self):
        return self.execute()


    ############################################################################
    #                            EXECUTION BRANCHES                            #
    ############################################################################
    def fit(self, ckpt='last'):
        return self.trainer.fit(
            model = 		self.model,
            datamodule = 	self.datamodule,
            ckpt_path =		ckpt
        )
    def validate(self, ckpt='last'):
        return self.trainer.validate(
            model = 		self.model,
            datamodule = 	self.datamodule,
            ckpt_path =		ckpt
        )
    def test(self, ckpt='last'):
        return self.trainer.test(
            model = 		self.model,
            datamodule = 	self.datamodule,
            ckpt_path =		ckpt
        )
    def generate(self, ckpt='last', how_many=128):
        self.load_checkpoint(ckpt)
        self.model.to(self.trainer.device)
        self.model.sample_batch(how_many)


    ############################################################################
    #                               INFO METHODS                               #
    ############################################################################

    def get_training_info(self):
        return self.trainer.current_epoch
    
    def get_dataset_info(self):
        return self.dataset_info
    
    def get_configuration(self, path: str=None):
        if path is None:
            return self.cfg
        else:
            return OmegaConf.select(self.cfg, path, throw_on_missing=True)

    ############################################################################
    #                     STORING AND CHECKPOINTING METHODS                    #
    ############################################################################

    def store_graphs(self, graphs: List, path: str=None):
        if path is None:
            # store graphs with date and time in name
            filename = f'generated_graphs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
            path = self.run_directory / filename
        # store graphs using pickle
        with open(path, 'wb') as f:
            pkl.dump(graphs, f)


    def load_checkpoint(self, checkpoint_name: str=None, strict: bool=True, load_stored_modules: bool=False):

        if not self.enable_ckp:
            self.logger.warning(f'Checkpointing is disabled, skipping...')
            return
        
        if load_stored_modules:
            # set last because the best modules will be loaded after
            checkpoint_name = 'last'

        self.logger.info(f'Loading checkpoint {checkpoint_name}...')
        if checkpoint_name is None:
            checkpoint_name = 'last'
        if not checkpoint_name.endswith('.ckpt'):
            checkpoint_name += '.ckpt'
        self.model = InsertDenoiseHaltModel.load_from_checkpoint(
            checkpoint_path =               str(self.run_directory / checkpoint_name),
            sampling_metrics =              None if 'sampling' not in self.model.losses else self.model.losses['sampling'],
            inference_samples_converter =   self.model.inference_samples_converter,
            console_logger =                self.model.console_logger,
            strict =                       strict
        )

        # load stored modules if requested, these are in the form of, e.g.:
        # model_reinsertion.pt, model_denoising.pt, etc..., and are stored
        # by the ModularModelCheckpoint callback
        if load_stored_modules:
            self.logger.info(f'Overriding modules with stored ones...')
            modules_list = glob.glob(str(self.run_directory / f'{PREFIX_MODULE}*.pt'))
            modules_found = []
            for module_path in modules_list:
                module_name = Path(module_path).name[len(PREFIX_MODULE):-3]
                # if it is a module, then load the stored state_dict
                if is_model_name(module_name):
                    modules_found.append(module_name)
                    self.load_module_from(self.run_directory, module_name, use_stored_module=True)
                    

            if len(modules_found) == 0:
                self.logger.info(f'No modules found')
            else:
                self.logger.info(f'Loaded modules: {", ".join(modules_found)}')

        self.logger.info(f'Succesfully loaded checkpoint {checkpoint_name}')


    def load_module_from(self, ckpt_path: str, module_name: str, use_stored_module: bool=False):

        if use_stored_module:
            loaded_module = torch.load(str(ckpt_path) + f'/{PREFIX_MODULE}{module_name}.pt')
            self.model.get_module(module_name).load_state_dict(loaded_module)

        else: # old way to do this
        
            other_model = InsertDenoiseHaltModel.load_from_checkpoint(
                checkpoint_path =               ckpt_path,
                sampling_metrics =              None if 'sampling' not in self.model.losses else self.model.losses['sampling'],
                inference_samples_converter =   self.model.inference_samples_converter,
                console_logger =                self.model.console_logger,
            )

            if module_name == 'reinsertion':
                self.model.reinsertion_model = other_model.reinsertion_model
            elif module_name == 'denoising':
                self.model.denoising_model = other_model.denoising_model


    ############################################################################
    #                            EVALUATION METHODS                            #
    ############################################################################
                
    def evaluate_ckpt(self, ckpt=None, use_modules: bool=False, validation=True):

        if ckpt is None:
            ckpt = 'best'

        if use_modules:
            self.load_checkpoint(ckpt, load_stored_modules=True)
            ckpt_to_use = None
        else:
            ckpt_to_use = ckpt

        self.logger.info(f'Current checkpoint: {ckpt}')

        # test the model using best checkpoint
        if validation:
            curr_metrics = self.validate(ckpt=ckpt_to_use)[0]
        else:
            curr_metrics = self.test(ckpt=ckpt_to_use)[0]

        curr_metrics['run'] = self.group_name
        curr_metrics['seed'] = self.seed
        curr_metrics['ckpt'] = ckpt

        return curr_metrics
    

    def evaluate_best(self, validation=True):
            
        # evaluate the model using best checkpoint
        curr_metrics = self.evaluate_ckpt('best', use_modules=True, validation=validation)

        return curr_metrics
    

    def evaluate_all_checkpoints(self, validation=True):

        dictionaries = []

        for ckpt in self.get_all_checkpoints(include_last=False):
                
            curr_metrics = self.evaluate_ckpt(ckpt, validation)
            dictionaries.append(curr_metrics)

        return dictionaries
    

    def log_dict_as_table(
            self,
            dictionary: Dict|List[Dict],
            name: str=None,
            save_table: bool=True,
            save_artifact: bool=True,
            save_file: bool=True
        ):
        
        if name is None:
            name = 'test_table'
        if save_file:
            with open(self.run_directory / f'{name}.json', 'w') as f:
                json.dump(dictionary, f)
        if save_table:
            curr_dict = dictionary if isinstance(dictionary, list) else [dictionary]
            columns = list(dictionary[0].keys())
            # each row has a run, each columns is a metric of the dict
            transposed_table = [[d[k] for k in columns] for d in curr_dict]
            table = wandb.Table(columns=columns, data=transposed_table)
            wandb.log({f"test/{name}": table})
        if save_artifact and save_table:
            table_art = wandb.Artifact(f'{name}_{self.run_id}', type='table')
            table_art.add(table, name)
            wandb.log_artifact(table_art)


    def get_all_checkpoints(self, include_path=True, include_last=False) -> List[Union[str, Path]]:
        checkpoints = list(self.run_directory.glob('*.ckpt'))

        if not include_last:
            checkpoints = [c for c in checkpoints if not c.name.startswith('last')]

        if not include_path:
            checkpoints = [c.name for c in checkpoints]

        return checkpoints


    def sample_batch(self, which_split: str='train'):
        self.datamodule.setup(which_split)
        batch = next(iter(self.datamodule.get_dataloader(which_split)))
        return batch
    

    def dry_run(self, which_split: str='train', num_steps: int=1, no_grad: bool=True):
        # save all relevant states
        grad_state = torch.is_grad_enabled()

        trn_curr_steps = self.trainer.limit_train_batches
        val_curr_steps = self.trainer.limit_val_batches
        tst_curr_steps = self.trainer.limit_test_batches
        curr_step = self.trainer.global_step
        curr_batch_step = self.trainer.fit_loop.epoch_loop.batch_progress.current.ready
        epoch_progress = deepcopy(self.trainer.fit_loop.epoch_progress.current)
        

        disable_generation = self.model._disable_generation
        debug_state = self.model.enable_logging
        
        try:    # run the desired split in safety

            if no_grad:
                torch.set_grad_enabled(False)
            self.model._disable_generation = True
            self.model.enable_logging = True

            if which_split == 'train':
                # temporarily update the trainer
                self.trainer.limit_train_batches = num_steps
                self.trainer.limit_val_batches = 0
                self.trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.optim_progress.optimizer.step.total.completed = 0
                self.trainer.fit_loop.epoch_loop.batch_progress.current.ready = 0
                self.trainer.fit_loop.epoch_progress.reset()

                # reset trainer to restart training
                # recall you cannot set global_step
                self.trainer.fit_loop.epoch_loop.batch_progress.current.reset()

                self.fit()

            elif which_split == 'valid':
                # temporarily update the trainer
                self.trainer.limit_val_batches = num_steps

                self.validate()

            elif which_split == 'test':
                # temporarily update the trainer
                self.trainer.limit_test_batches = num_steps

                self.test()
            
            elif which_split == 'gen':
                # temporarily update the trainer
                self.model._disable_generation = False

                self.model.sample_batch(64 * num_steps)

        finally:    # restore torch, trainer and model states

            if no_grad:
                torch.set_grad_enabled(grad_state)

            self.trainer.limit_train_batches = trn_curr_steps
            self.trainer.limit_val_batches = val_curr_steps
            self.trainer.limit_test_batches = tst_curr_steps
            self.trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.optim_progress.optimizer.step.total.completed = curr_step
            self.trainer.fit_loop.epoch_loop.batch_progress.current.ready = curr_batch_step
            self.trainer.fit_loop.epoch_progress.current = epoch_progress

            self.model._disable_generation = disable_generation
            self.model.enable_logging = debug_state


    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()

    
    def close(self):

        # turn off logger
        self.turn_off_logger(self.logger)
        self.turn_off_logger(self.model.console_logger)

        if self.wandb_active:
            wandb.finish()

    ############################################################################
    #                             UTILITY METHODS                              #
    ############################################################################

    def validate_context(self):
        pass

    def _setup_run_directory(self, config_name: str) -> Tuple[Path, int, str]:

        self.resume = False

        if self.enable_ckp:
            
            run_path = Path(CHECKPOINT_PATH, config_name)

            ####################  IF PATH HAS TO BE LOADED  ####################
            if self.load_ckp is not None:

                self.resume = True

                #####################  IF PATH IS A STRING  ####################
                if isinstance(self.load_ckp, str):
                    # load checkpoint from path
                    # path already exists and is already well-formed
                    run_path = Path(self.load_ckp)
                    if not run_path.exists():
                        raise ContextException(f'No run found matching the path {run_path}')
                    version = run_path.name.split('_')[0][1:]
                    run_id = run_path.name.split('_')[-1]

                #####################  IF PATH IS AN INT  ######################
                elif isinstance(self.load_ckp, int):
                    # load checkpoint from version
                    # path is taken from the configuration + version
                    version = self.load_ckp
                    # if the version is not specified, get the latest version
                    if version == -1:
                        # get latest version
                        matched_run_path = list(run_path.glob('v*'))
                        if len(matched_run_path) == 0:
                            raise ContextException(f'No runs found matching the name {config_name}')
                        version = max([int(p.name.split('_')[0][1:]) for p in matched_run_path])
                
                    # check that the run path exists (checking the prefix)
                    matched_run_path = list(run_path.glob(f'v{version}_*'))
                    if len(matched_run_path) == 0:
                        raise ContextException(f'No run found matching the version {version}')
                    if len(matched_run_path) > 1:
                        raise ContextException(f'Multiple runs found matching the version {version}')

                    # now we know the run exists, so we can get the run id
                    run_path = matched_run_path[0]
                    run_id = matched_run_path[0].name.split('_')[-1]

            ###################  IF PATH HAS TO BE CREATED  ####################
            else:
                # generate run id
                run_id = wandb.util.generate_id()
                version = 0
                
                # check that the run path exists, and if not, create it
                run_path.mkdir(parents=True, exist_ok=True)

                # list all directories in the run path and check the latest version
                matched_run_path = list(run_path.iterdir())
                if len(matched_run_path) > 0:
                    version = max([int(p.name.split('_')[0][1:]) for p in matched_run_path]) + 1
                
                # create the new run path
                run_path = Path(run_path, f'v{version}_{run_id}')

            return run_path, version, run_id
        
        else:
            return None, 0, wandb.util.generate_id()


    ############################################################################
    #                          CONFIGURATION METHODS                           #
    ############################################################################
        

    def _configure_execution_args(self, cfg: DictConfig):

        # store naming, useful for logging into wandb and for saving checkpoints
        self.config_name = cfg['config_name']
        self.logger.info(f'Configuring run "{self.config_name}"')

        # store grouping, useful for grouping runs in wandb
        if 'group_name' in cfg:
            self.group_name = cfg['group_name']
        else:
            self.group_name = self.config_name
        self.logger.info(f'Group name: {self.group_name}')

        # store execution mode: decides what to do in this execution
        # e.g. train, generate, evaluate, etc.
        self.mode = cfg['mode']
        if self.mode not in ALLOWED_EXECUTION_MODES:
            raise ContextException(
                f'Unknown execution mode {self.mode}, '
                f'must be one of {ALLOWED_EXECUTION_MODES}'
            )
        self.logger.info(f'Run mode: {self.mode}')
        
        # store enable flags
        self.debug = cfg['debug']
        self.enable_ckp = cfg['enable_ckp'] and not self.debug
        self.enable_log = cfg['enable_log'] and not self.debug
        self.profile = cfg['profile']

        add_msg = '. Logging to wandb and checkpointing are disabled.' if self.debug else ''
        self.logger.info(f'Debug mode: {self.debug}{add_msg}')
        self.logger.info(f'Checkpoints enabled: {self.enable_ckp}')
        self.logger.info(f'Wandb logging enabled: {self.enable_log}')
        self.logger.info(f'Profiling: {self.profile}')

        # store whether to load a checkpoint
        self.load_ckp = cfg['load_ckp']
        self.logger.info(f'Resuming run: {self.load_ckp}')

        # store seed to use
        self.seed = cfg['seed']
        self.logger.info(f'Seed: {self.seed}')

        # store entire configuration
        self.cfg = cfg
        


    def _configure_datamodule(
            self,
            cfg_data: DictConfig,
            cfg_removal: DictConfig
        ) -> GraphDataModule:

        ###########################  DATASET SETUP  ############################
        # get dataset name
        dataset_name = cfg_data['dataset']['name']
        dataset_directory = cfg_data['dataset']['root']
        dataset_path = osp.join(DATASETS_ROOT, dataset_directory)
        self.logger.info(f'Using dataset "{dataset_name}"')
        self.logger.info(f'Dataset will be stored at: "{dataset_path}"')

        ##########################  DATALOADER SETUP  ##########################
        # extract dataloader configuration as dict
        cfg_dataloader = OmegaConf.to_container(cfg_data['dataloader'])

        #  SUBSTITUTE COLLATER  #
        # setup how to collate data in the dataloader
        # i.e. how to merge multiple datapoints into a batch
        def setup_collater(cfg_dataloader):
            if 'collate_fn' in cfg_dataloader:
                # resolve collater return a Collater object given a string
                cfg_dataloader['collate_fn'] = resolve_collater(cfg_dataloader['collate_fn'])

        # if there are no splits
        if 'batch_size' in cfg_dataloader:
            bs = cfg_dataloader["batch_size"]
            setup_collater(cfg_dataloader)
            self.logger.info(f'Batch size: {bs}')
        # if there are splits (train, valid, test)
        else:
            bs = cfg_dataloader["train"]["batch_size"]
            for split in cfg_dataloader:
                setup_collater(cfg_dataloader[split])
                self.logger.info(f'Batch size for {split}: {cfg_dataloader[split]["batch_size"]}')

        #########################  DATATRANSFORM SETUP  ########################
        # setting up data transform with graph subsampling
        cfg_datatf = cfg_data['datatransform']

        # choose how to sample the subgraphs sequence:\
        # options: sample the whole sequence to train, or sample just one graph
        if cfg_datatf['sample_whole_sequence']:
            num_sequences = cfg_datatf['num_sequences']
            data_sampler = SubsequenceSampler.create_subsequence_sampler(
                process_config = cfg_removal,
                num_sequences = num_sequences
            )
            self.logger.info(f'Sampling {num_sequences} sequences of subgraphs from 1 datapoint')
        else:
            data_sampler = SubgraphSampler.create_subgraph_sampler(
                process_config = cfg_removal
            )
            self.logger.info(f'Sampling 1 random subgraphs from 1 datapoint')

        # create runtime process pipeline
        # will be executed during dataloading for each single graph, on CPU
        runtimeprocess_pl = Compose([
            # transform graph into undirected graph
            MyToUndirected(),
            # prepare data for removal process
            # e.g., compute ordering during dataloading on cpu
            PrepareData(data_sampler.removal_process)
        ])

        # get download and preprocess pipelines
        download_pl_kwargs = cfg_data['dataset']['download']
        if download_pl_kwargs is None:
            download_pl_kwargs = {}
        download_pl = ppl.REGISTERED_DOWNLOAD_PIPELINES[dataset_name](**download_pl_kwargs)

        preprocess_pl_kwargs = cfg_data['dataset']['preprocess']
        if preprocess_pl_kwargs is None:
            preprocess_pl_kwargs = {}
        preprocess_pl = ppl.REGISTERED_PREPROCESS_PIPELINES[dataset_name](**preprocess_pl_kwargs)

        #########################  CREATE DATAMODULE  ##########################
        # create datamodule
        datamodule = GraphDataModule(
            root_dir =          dataset_path,
            download_pl =       download_pl,
            preprocess_pl =     preprocess_pl,
            runtimeprocess_pl = runtimeprocess_pl,
            dataloader_config = cfg_dataloader
        )

        return datamodule
    

    def _configure_inference_samples_converter(self, datatype: str, dataset_info) -> Callable:

        if datatype == 'molecular':
            # get graph to molecule converter for transforming graphs into molecules
            return GraphToMoleculeConverter(
                atom_decoder = dataset_info['atom_types'],
                bond_decoder = dataset_info['bond_types']
            )
        
        else:
            return None


    def _configure_sampling_metrics(self, datatype: str, cfg_metrics: DictConfig, datamodule: GraphDataModule, inference_samples_converter: Callable) -> SamplingMetricsHandler:

        datamodule.setup('train', disable_transform=True)
        datamodule.setup('test', disable_transform=True)
        
        def configure_split_samp_metr(curr_cfg_metrics):

            metrics = SamplingMetricsHandler(
                datamodule =        datamodule,
                generation_cfg =    curr_cfg_metrics['generation'],
                metrics_cfg =       curr_cfg_metrics['metrics'],
                samples_converter = inference_samples_converter
            )
            
            return metrics

        
        if 'valid' in cfg_metrics or 'test' in cfg_metrics:
            metrics = nn.ModuleDict({
                f'_{split}': configure_split_samp_metr(curr_cfg) for split, curr_cfg in cfg_metrics.items()
            })
        else:
            metrics = configure_split_samp_metr(cfg_metrics)

        datamodule.clear_datasets()
        
        return metrics
    

    def _configure_model(
            self,
            datatype: str,
            cfg_model: DictConfig,
            cfg_training: DictConfig,
            cfg_metrics: DictConfig,
            datamodule: GraphDataModule,
            dataset_info: Dict
        ):

        # create torch graph to datatype converter, used as the final stage of generation, e.g.:
        # - get graph to molecule converter for transforming graphs into molecules
        # - get graph to nx-graph converter for transforming graphs into nx-graphs (not implemented)
        inference_samples_converter = self._configure_inference_samples_converter(
            datatype =      datatype,
            dataset_info =  dataset_info
        )
        self.logger.info(f'Generated graphs are transformed by: {type(inference_samples_converter)}')

        # get sampling metrics
        sampling_metrics = self._configure_sampling_metrics(
            datatype =                      datatype,
            cfg_metrics =                   cfg_metrics,
            datamodule =                    datamodule,
            inference_samples_converter =   inference_samples_converter
        )
        self.logger.info(f'Sampling metrics: {type(sampling_metrics).__name__}')

        # setup console logger for the model
        console_logger = self.initialize_logger('model', self.logger.logger.level)

        # create model
        model = InsertDenoiseHaltModel(
            architecture_config =           cfg_model['architecture'],
            diffusion_config =              cfg_model['diffusion'],
            removal_config =                cfg_model['removal'],
            dataset_info =                  dataset_info,
            run_config =                    cfg_training,
            sampling_metrics =              sampling_metrics,
            inference_samples_converter =   inference_samples_converter,
            console_logger =                console_logger,
            conditional_generator =         cfg_model['conditional'],
            enable_logging =                self.enable_log
        )

        try:
            model = torch.compile(model)
        except Exception as e:
            self.logger.warning(f'Could not compile the model, cause: {e}')

        return model
    
    
    def _configure_checkpoint(self, cfg_checkpoint: DictConfig) -> ModelCheckpoint:

        checkpoint_callback = []

        if self.enable_ckp:

            # get dictionary and see if there are modules configurations
            cfg_checkpoint = OmegaConf.to_container(cfg_checkpoint)
            module_monitors = cfg_checkpoint.pop('module_monitors', None)

            # if a checkpointing is specified for each module
            if module_monitors:
                for module_name, cfg_module in module_monitors.items():
                    if is_model_name(module_name) and cfg_module is not None:
                        # create a checkpoint callback for the module
                        curr_callback = ModularModelCheckpoint(
                            dirpath =       self.run_directory,
                            filename =      None,
                            **cfg_module,
                            **cfg_checkpoint
                        )
                        # include the module in the callback
                        curr_callback.include_module(module_name)
                        # append the callback to the list
                        checkpoint_callback.append(curr_callback)


            # create the general checkpoint callback
            checkpoint_callback.append(ModelCheckpoint(
                dirpath =       self.run_directory,
                filename =      None,
                save_last =     True,
                **cfg_checkpoint
            ))
        else:
            # callback that never saves anything
            # it is enough to disable saving last
            # and saving top k
            checkpoint_callback.append(ModelCheckpoint(
                dirpath =       self.run_directory,
                filename =      None,
                save_last =     False,
                save_top_k =    0
            ))
        
        return checkpoint_callback
    

    def _configure_logger(self, cfg_logger: DictConfig):
        if self.enable_log and not self.wandb_active:
            wandb.init(
                name =      self.config_name,
                resume =    self.resume,
                id =        self.run_id,
                config =    OmegaConf.to_container(self.cfg),
                **cfg_logger['wandb']
            )
            self.wandb_active = True


    def _configure_early_stopping(self, cfg_early_stopping: DictConfig):
        early_stopping_callback = ModularEarlyStopping(
            **cfg_early_stopping
        )

        return early_stopping_callback


    def _configure_trainer(
            self,
            cfg_run: DictConfig,
            cfg_platform: DictConfig
        ) -> Trainer:

        callbacks = []

        # if debug is activated or persistent is deactivated, checkpointing and logging are deactivated!

        if 'checkpoint' in cfg_run:
            checkpoint_callback = self._configure_checkpoint(cfg_run['checkpoint'])
            callbacks.extend(checkpoint_callback)

        if 'logger' in cfg_run:
            self._configure_logger(cfg_run['logger'])

        if 'early_stopping' in cfg_run:
            early_stopping_callback = self._configure_early_stopping(cfg_run['early_stopping'])
            callbacks.append(early_stopping_callback)

        # configure trainer
        gpus_ok = gpus_available(cfg_platform)
        gpus_num = cfg_platform['gpus'] if gpus_ok else 0

        self.logger.info(f'Using GPU: {gpus_ok}, N={gpus_num}')
        self.logger.info(f'Number of epochs: {cfg_run["trainer"]["max_epochs"]}')

        if self.profile:
            # remove file if exists
            for f in glob.glob('perf_logs*'):
                os.remove(f)
            profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
        else:
            profiler = None
        
        # build trainer
        trainer = Trainer(
            # training
            max_epochs =                cfg_run['trainer']['max_epochs'],

            # validation
            val_check_interval =		cfg_run['trainer']['val_check_interval'],
            check_val_every_n_epoch =	cfg_run['trainer']['check_val_every_n_epoch'],
            num_sanity_val_steps =		cfg_run['trainer']['num_sanity_val_steps'],

            # testing
            limit_train_batches =		20 if cfg_run['running_test'] else None,
            limit_val_batches =			20 if cfg_run['running_test'] else None,
            limit_test_batches =		20 if cfg_run['running_test'] else None,

            # computing devices
            accelerator =				'gpu' 		if gpus_ok else 'cpu',
            devices =					gpus_num 	if gpus_ok else None,
            strategy =					'ddp_find_unused_parameters_true' 		if gpus_num > 1 else 'auto',

            # visualization and debugging
            fast_dev_run = 				self.debug,
            enable_progress_bar =		cfg_run['trainer']['enable_progress_bar'],

            # logging and checkpointing
            logger =                    False,

            # callbacks
            callbacks =					callbacks,
            # for network debugging in case of NaNs
            profiler=profiler,
            detect_anomaly=False
        )
    
        return trainer
    

    def initialize_logger(self, name: str,  level: Union[str, int] = None) -> logging.Logger:

        level = level if level is not None else logging.INFO
        if isinstance(level, str) and not level.isupper():
            level = level.upper()

        logger = logging.getLogger(name)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.propagate = False
        logger.addHandler(handler)
        logger.setLevel(level)

        logger = IndentedLoggerAdapter(logger)

        return logger
    
    def turn_off_logger(self, logger: logging.Logger):
        if isinstance(logger, IndentedLoggerAdapter):
            logger = logger.logger
        
        logger.handlers = []
        logger.propagate = False


################################################################################
#                                UTILITY METHOD                                #
################################################################################

def preprocess_config(cfg: DictConfig):
    
    # resolve configuration interpolations which are using hydra choices
    choices = cfg.hydra.runtime.choices
    cfg.hydra = OmegaConf.create({'runtime': {'choices': choices}})
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # remove hydra configuration
    cfg.pop('hydra')
    cfg = OmegaConf.create(cfg)

    return cfg


################################################################################
#                            CONFIGURATION METHODS                             #
################################################################################

def setup_logger(logger: logging.Logger|int = None):
    if logger is None:
        level = logging.INFO
    elif isinstance(logger, int):
        level = logger
        logger = None
    elif isinstance(logger, logging.Logger):
        level = logger.level

    if logger is None:
        logger = logging.getLogger('configurator')
        logger.setLevel(level=level)
        
    # remove all handlers
    logger.handlers = []
    # set format of logger
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(level=level)
    ch.setFormatter(formatter)
    # add console handler to logger
    logger.addHandler(ch)

    return logger


def gpus_available(platform_config):
    return torch.cuda.is_available() and platform_config['gpus'] > 0
    