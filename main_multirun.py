import hydra
# for some reasons the pandas import is required,
# as it goes in conflict with torch_geometric
import pandas
from omegaconf import DictConfig, OmegaConf, open_dict
from src.configurator import RunContext, preprocess_config
from hydra.core.hydra_config import HydraConfig
import torch

@hydra.main(version_base=None, config_path='configs', config_name='default')
def main(cfg: DictConfig):

    BIG_VALUE = 10000
    out_metrics = {
        'reinsertion_loss_kldiv': BIG_VALUE,
        'denoising_loss_total_ce': BIG_VALUE,
        'halting_prior_emd': BIG_VALUE
    }

    order_metrics = [
        'reinsertion_loss_kldiv',
        'denoising_loss_total_ce',
        'halting_prior_emd'
    ]

    context = None

    try:

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

        hparams_eval_metrics = {
            'valid_reinsertion/reinsertion_loss_kldiv': 'reinsertion_loss_kldiv',
            'valid_denoising/denoising_loss_total_ce': 'denoising_loss_total_ce',
            'valid_halting/halting_prior_emd': 'halting_prior_emd'
        }

        for m_in, m_out in hparams_eval_metrics.items():
            if m_in in results:
                val = results[m_in]
                if isinstance(val, torch.Tensor):
                    val = val.detach().cpu().item()
                out_metrics[m_out] = val


        # close context (e.g. close wandb connection, garbage collection, etc.)
        context.close()

        # return results that can be used for hparams selection when using some
        # sweeper with hydra

    except Exception as e:

        print(f"Job stopped due to exception: {e}")
        if context is not None:
            context.close()

    return (out_metrics[m] for m in order_metrics)


if __name__ == '__main__':
    main()