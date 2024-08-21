# IFH: a Diffusion Framework for Flexible Design of Graph Generative Models

This is the code which was used to run the experiments for the paper "IFH: a Diffusion Framework for Flexible Design of Graph Generative Models".

## Requirements

The code was tested with Python 3.11.7. The requirements can be installed by executing 
- Download anaconda/miniconda if needed
- Create a new environment through the given environment files with the following command:
    ```bash
    conda env create -f <env_file>.yml
    ```
    where \<env_file\> is the name of the environment file to use. It is possible to install dependencies for CPU with `environment_cpu.yml` or for GPU with `environment_cuda.yml`.
- Install this package with the following command:
    ```bash
    pip install -e .
    ```
    which will compile required cython code.
- Navigate the directory ./src/metrics/utils/orca and compile the orca executable with the following command:
    ```bash
    g++ -O2 -std=c++11 -o orca orca.cpp
    ```

## Experiments
For CUDA>=10.2, to run any experiments in a reproducible way, it is necessary to set the environment variable:
```bash
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
or in Windows, for cmd:
```bash
    set CUBLAS_WORKSPACE_CONFIG=:4096:8
```
for PowerShell:
```bash
    $env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```
and to remove:
```bash
    Remove-Item Env:\CUBLAS_WORKSPACE_CONFIG
```

Hyperparameters searches can be performed with the commands:
```bash
    python main_multirun.py +experiment/seq_degree=<degree> +experiment/task=<dataset> seed=<seed> +modsel=<hparams>
```
where \<degree\> is the sequentiality degree, found in experiment/seq_degree; \<dataset\> contains all the dataset task information; \<seed\> is the seed to use for the experiment; \<hparams\> contains the set of hyperparameters to search on, i.e., "bayesian_no_reins" for 1-node sequential models, "bayesian_15" for block generation, and "bayesian_no_halt_reins_15" for one-shot models.

All the configurations of the experiments can be found at ./src/configs/compact. These files also contain the hyperparameters found in the search procedure. The experiments can be run with different seeds with the following command:
```bash
    python main.py +compact/<dataset>=<experiment> seed=<seed>
```
where \<dataset\> is the name of the dataset, \<experiment\> is the name of the experiment and \<seed\> is the seed to use for the experiment. For example, to run the experiment "exp4_one_zinc" on the dataset "zinc" with the seed 2, the following command can be used:
```bash
    python main.py +compact/zinc=exp4_one_zinc seed=2
```

## Datasets
Datasets will be downloaded automatically to a new directory ./datasets when running the experiments. Current available datasets are QM9, ZINC250K, community, ego-small, enzymes, and ego. The datasets are then stored in the PyTorch Geometric format.

## Checkpoints and logging
Checkpoints are saved in a new directory ./checkpoints, and logging is done through WandB, which requires a free account to be used.