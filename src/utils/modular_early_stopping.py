from typing import Dict, List, Any

from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class ModularEarlyStopping(EarlyStopping):
    """This early stopping method monitor one metric
    for each submodule of the observed pl_module. It then
    decides to stop one module at a time when its metric
    stops improving. When all modules have stopped, the
    whole training stops. This is implemented by having
    an EarlyStopping object for each module.
    """

    def __init__(
            self,
            module_monitors: Dict[str, Dict[str, Any]],
            verbose: bool = False,
            **kwargs
        ):

        self.early_stoppings = {
            mod: EarlyStopping(
                **mod_kwargs,
                **kwargs
            ) for mod, mod_kwargs in module_monitors.items() if mod_kwargs is not None
        }

        super().__init__(monitor='none', verbose=verbose, **kwargs)

    @property
    def state_key(self) -> str:

        state_key_kwargs = {}

        for mod, early_stopping in self.early_stoppings.items():
            state_key_kwargs[f'{mod}_monitor'] = early_stopping.monitor
            state_key_kwargs[f'{mod}_mode'] = early_stopping.mode

        return self._generate_state_key(**state_key_kwargs)


    def _validate_condition_metric(self, logs: Dict[str, Tensor]) -> bool:
        
        # check if logs contains all monitored metrics
        for mod, early_stopping in self.early_stoppings.items():
            if not early_stopping._validate_condition_metric(logs):
                return False

        return True
    

    def state_dict(self) -> Dict[str, Any]:
        # save state dict for each module
        state_dict = {}
        for mod, early_stopping in self.early_stoppings.items():
            state_dict[mod] = early_stopping.state_dict()

        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # load state dict for each module
        for mod, early_stopping in self.early_stoppings.items():
            early_stopping.load_state_dict(state_dict[mod])


    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, pl_module)

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, pl_module)
    

    def _run_early_stopping_check(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        # disable early_stopping with fast_dev_run
        if trainer.fast_dev_run:
            return

        general_should_stop = True

    	# check early stopping for each module
        for mod, early_stopping in self.early_stoppings.items():

            # if training for this module is already disabled, skip it
            if not pl_module.training_enabled[mod]:
                continue

            if not early_stopping._validate_condition_metric(logs): # short circuit if metric not present
                return

            # extract current metric and check if early stopping is required
            current = logs[early_stopping.monitor].squeeze()
            should_stop, reason = early_stopping._evaluate_stopping_criteria(current)

            # stop training of the module for this module if required
            if should_stop:
                pl_module.training_enabled[mod] = False
                early_stopping.stopped_epoch = trainer.current_epoch
            if reason and self.verbose:
                self._log_info(trainer, f'{mod}: {reason}', self.log_rank_zero_only)

            general_should_stop = general_should_stop and should_stop

        # stop trainer if all models have stopped
        general_should_stop = trainer.strategy.reduce_boolean_decision(general_should_stop, all=False)
        trainer.should_stop = trainer.should_stop or general_should_stop

        # stop trainer if required
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)