from typing import Dict, List, Tuple, Any, Set

import numpy as np
import torch
from torch import nn, Tensor
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from omegaconf import OmegaConf, DictConfig
import networkx as nx

from src.data.simple_transforms.molecular import mol2smiles, GraphToMoleculeConverter
from src.data.datamodule import GraphDataModule

import src.metrics.utils.molecular as molecular
import src.metrics.utils.synth as synth
import src.metrics.utils.graphgdp_metrics.evaluator as graphgdp_metrics


STATE_TORCH = 'torch'
STATE_NX = 'nx'
STATE_SMILES = 'smiles'
STATE_MOL = 'mol'
STATE_MOLNX = 'molnx'

def torch_to_nx(data: Data, to_undirected: bool=True) -> nx.Graph:
    return to_networkx(data, to_undirected=to_undirected)

def torch_to_mol(data: Data, g2m: GraphToMoleculeConverter) -> Chem.Mol:
    return g2m.graph_to_molecule(
        data,override_relaxed=True,
        override_post_hoc_mols_fix=True,
        override_post_hoc_mols_convert=True
    )

def mol_to_smiles(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol)

def mol_to_molnx(mol: Chem.Mol) -> nx.Graph:
    return molecular.mols_to_nx(mol)

TRANSFORMS = {
    (STATE_TORCH, STATE_NX): torch_to_nx,
    (STATE_TORCH, STATE_MOL): torch_to_mol,
    (STATE_MOL, STATE_SMILES): mol_to_smiles,
    (STATE_MOL, STATE_MOLNX): mol_to_molnx,
}

REQUIREMENTS = {
    STATE_TORCH: set(),
    STATE_NX: {STATE_TORCH},
    STATE_SMILES: {STATE_MOL},
    STATE_MOL: {STATE_TORCH},
    STATE_MOLNX: {STATE_MOL},
}

def transform(data: List[Any], required_states: Set[str], g2m: GraphToMoleculeConverter=None) -> Dict[str, Any]:
    
    out = {STATE_TORCH: data}
    # TODO: create generic transformation



        

class BaseSamplingMetric(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, data: Any, ret_logs: bool=True) -> Dict:
        raise NotImplementedError


