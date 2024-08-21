import hydra
# for some reasons the pandas import is required,
# as it goes in conflict with torch_geometric
import pandas
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig

from src.configurator import RunContext


@hydra.main(version_base=None, config_path='configs', config_name='default')
def main(cfg: DictConfig):

    # make hydra config available in the current config
    # will be removed later
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.hydra = HydraConfig.get()


    # prepare context with data, model, trainer, etc.
    context: RunContext = RunContext.from_config(cfg)
    
    # execute context based on input arguments
    # may return a dictionary of results, containing the metrics values
    results = context.execute()

    #if results is not None:
        # log results to the logger
    #    context.log_dict_as_table(results)

    # close context (e.g. close wandb connection, garbage collection, etc.)
    context.close()

    # return results that can be used for hparams selection when using some
    # sweeper with hydra
    return results


if __name__ == '__main__':
    main()