
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pathlib import Path


PREFIX_MODULE = 'model_'


class ModularModelCheckpoint(ModelCheckpoint):


    def include_module(self, module_name: str) -> None:
        """Include a module in the checkpointing process.
        """
        self._target_module = module_name


    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:

        # hijack the filepath to get the directory
        directory = Path(filepath).parent

        # generate new filepath for the single module
        new_filepath = directory / f'{PREFIX_MODULE}{self._target_module}.pt'

        # extract the module
        module = trainer.model.get_module(self._target_module)

        # save the module
        torch.save(module.state_dict(), new_filepath)