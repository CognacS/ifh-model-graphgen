from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch import nn, Tensor
import re
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from omegaconf import OmegaConf, DictConfig
import networkx as nx

from src.data.simple_transforms.molecular import mol2smiles, GraphToMoleculeConverter
from src.data.datamodule import GraphDataModule

import src.metrics.metrics as m_list
import src.metrics.utils.molecular as molecular
import src.metrics.utils.synth as synth
import src.metrics.utils.graphgdp_metrics.evaluator as graphgdp_metrics


class BaseSamplingMetric(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, data: Any, ret_logs: bool=True) -> Dict:
        raise NotImplementedError
    

class SamplingMetricsHandler(BaseSamplingMetric):

    def __init__(self, datamodule, generation_cfg, metrics_cfg, samples_converter=None):
        super().__init__()
        
        self.generation_cfg = generation_cfg
        self.metrics = {}

        for metric_type, curr_cfg_metrics in metrics_cfg.items():
            if metric_type == m_list.KEY_MOLECULAR_METRIC_TYPE:
                metrics = MoleculeSamplingMetrics(
                    datamodule = datamodule,
                    metrics = curr_cfg_metrics,
                    graph_to_mol_converter = samples_converter
                )

            elif metric_type == m_list.KEY_GRAPH_METRIC_TYPE:

                metrics = GraphSamplingMetrics(
                    datamodule = datamodule,
                    metrics = curr_cfg_metrics
                )

            elif metric_type == m_list.KEY_PROCESS_METRIC_TYPE:
                metrics = ProcessSamplingMetrics(
                    metrics = curr_cfg_metrics
                )

            elif metric_type == 'summarizing':
                metrics = SummarizingSamplingMetrics(
                    weights = curr_cfg_metrics
                )

            else:
                raise ValueError(f'Unknown metric_type "{metric_type}"')
            
            self.metrics[metric_type] = metrics

    
    def __call__(self, data: Dict, ret_logs: bool=True) -> Dict:
        
        logs = {}

        for metric_type, metric in self.metrics.items():
            if metric_type in [m_list.KEY_MOLECULAR_METRIC_TYPE, m_list.KEY_GRAPH_METRIC_TYPE]:
                logs.update(metric(data['generated_graphs'], ret_logs=ret_logs))
            elif metric_type == m_list.KEY_PROCESS_METRIC_TYPE:
                logs.update(metric(data, ret_logs=ret_logs))
            elif metric_type == 'summarizing':
                logs.update(metric(logs, ret_logs=ret_logs))
            else:
                raise ValueError(f'Unknown metric_type "{metric_type}"')
        
        return logs

################################################################################
#                          MOLECULAR SAMPLING METRICS                          #
################################################################################

class MoleculeVUNMetric(BaseSamplingMetric):

    NEEDED_PARTITION = ['train']

    def __init__(self, train_smiles: List[str], graph_to_mol_converter: GraphToMoleculeConverter):
        super().__init__()

        self.dataset_smiles_list = train_smiles

        self.graph_to_mol_converter = graph_to_mol_converter
    

    def compute_validity(
            self,
            generated_graphs: List[Data],
            relaxed: bool=False

        ) -> Tuple[List[str], float, np.ndarray, List[str]]:
        """ generated: list of couples (positions, atom_types)"""
        
        valid_smiles = []
        num_components = []
        all_smiles = []
        
        for graph in generated_graphs:
            
            # torch_geometric graph to RDKit molecule
            mol = self.graph_to_mol_converter(
                graph,
                override_relaxed=relaxed
            )

            # RDKit molecule to string (SMILES)
            smiles = mol2smiles(mol)

            try:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                num_components.append(len(mol_frags))
            except:
                pass
            if smiles is not None:
                try:
                    mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                    largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                    smiles = mol2smiles(largest_mol)
                    all_smiles.append(smiles)
                    if smiles != '' and smiles is not None:
                        valid_smiles.append(smiles)
                except Chem.rdchem.AtomValenceException:
                    print("Valence error in GetmolFrags")
                    all_smiles.append(None)
                except Chem.rdchem.KekulizeException:
                    print("Can't kekulize molecule")
                    all_smiles.append(None)
            else:
                all_smiles.append(None)

        return valid_smiles, len(valid_smiles) / len(generated_graphs), np.array(num_components), all_smiles
    

    def compute_uniqueness(
            self,
            valid_smiles: List[str]

        ) -> Tuple[List[str], float]:

        """ valid: list of SMILES strings."""
        return list(set(valid_smiles)), len(set(valid_smiles)) / len(valid_smiles)


    def compute_novelty(
            self,
            unique_smiles: List[str]
        ) -> Tuple[List[str], float]:

        num_novel = 0
        novel = []
        if self.dataset_smiles_list is None:
            print("Dataset smiles is None, novelty computation skipped")
            return [], 1
        for smiles in unique_smiles:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / len(unique_smiles)
    
    

    def __call__(self, generated_graphs, ret_logs: bool=True) -> Dict:
        """ generated: list of pairs (positions: n x 3, atom_types: n [int])
            the positions and atom types should already be masked. """
        
        # 1 - compute validity
        valid_smiles, validity, num_components, all_smiles = self.compute_validity(generated_graphs)
        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0
        print(f"Validity over {len(generated_graphs)} molecules: {validity * 100 :.2f}%")
        print(f"Number of connected components of {len(generated_graphs)} molecules: min:{nc_min:.2f} mean:{nc_mu:.2f} max:{nc_max:.2f}")

        # 2 - compute relaxed validity
        relaxed_valid_smiles, relaxed_validity, num_components, all_smiles = self.compute_validity(generated_graphs, relaxed=True)
        print(f"Relaxed validity over {len(generated_graphs)} molecules: {relaxed_validity * 100 :.2f}%")
        if relaxed_validity > 0:
            unique_smiles, uniqueness = self.compute_uniqueness(relaxed_valid_smiles)
            print(f"Uniqueness over {len(relaxed_valid_smiles)} valid molecules: {uniqueness * 100 :.2f}%")

            if self.dataset_smiles_list is not None:
                novel_smiles, novelty = self.compute_novelty(unique_smiles)
                print(f"Novelty over {len(unique_smiles)} unique valid molecules: {novelty * 100 :.2f}%")
            else:
                novelty = -1.0
        else:
            novelty = -1.0
            uniqueness = 0.0
            unique_smiles = []

        logs = {
            m_list.KEY_MOLECULAR_VALIDITY: validity,
            m_list.KEY_MOLECULAR_RELAXED_VALIDITY: relaxed_validity,
            m_list.KEY_MOLECULAR_UNIQUENESS: uniqueness,
            m_list.KEY_MOLECULAR_NOVELTY: novelty,
        }

        ret = {
            KEY_SAMPLE_METRICS: logs,
            KEY_UNIQUE_SMILES: unique_smiles,
            KEY_NUM_COMPONENTS: dict(nc_min=nc_min, nc_max=nc_max, nc_mu=nc_mu),
            KEY_ALL_SMILES: all_smiles
        }

        if ret_logs:
            return logs
        else:
            return ret


class MoleculeDistributionMetric(BaseSamplingMetric):

    NEEDED_PARTITION = ['test']

    def __init__(self, test_smiles: List[str], graph_to_mol_converter: GraphToMoleculeConverter, metrics_list='all'):
        super().__init__()

        self.test_smiles = test_smiles
        self.num_graphs_test = len(self.test_smiles)

        self.graph_to_mol_converter = graph_to_mol_converter

        if metrics_list == 'all':
            metrics_list = [
                m_list.KEY_MOLECULAR_NSPDK,
                m_list.KEY_MOLECULAR_FCD
            ]
        if metrics_list == 'all+cond':
            metrics_list = [
                m_list.KEY_MOLECULAR_NSPDK,
                m_list.KEY_MOLECULAR_FCD,
                'cond_mae'
            ]

        self.metrics_list = {}

        if m_list.KEY_MOLECULAR_NSPDK in metrics_list:
            test_mols = [Chem.MolFromSmiles(s) for s in self.test_smiles]
            self.test_graphs = molecular.mols_to_nx(test_mols)
            self.metrics_list[m_list.KEY_MOLECULAR_NSPDK] = molecular.get_nspdk_eval()

        if m_list.KEY_MOLECULAR_FCD in metrics_list:
            self.metrics_list[m_list.KEY_MOLECULAR_FCD] = molecular.get_FCDMetric(
                ref_smiles = self.test_smiles,
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            )

        if 'cond_mae' in metrics_list:
            self.metrics_list['cond_mae'] = molecular.get_mae_molecular_metric(
                which_properties = ['qed', 'plogp'],
                separated = True
            )



    def __call__(self, generated_graphs: List, ret_logs: bool=True) -> Dict:
        print(f"Computing distribution metrics between {len(generated_graphs)} generated graphs and {len(self.test_graphs)} reference graphs")
        
        # apply post-hoc molecule correction to compute valid molecules
        posthoc_molecules = self.graph_to_mol_converter(
            generated_graphs,
            override_relaxed=True,
            override_post_hoc_mols_fix=True,
            override_post_hoc_mols_convert=True
        )

        logs = {}

        if m_list.KEY_MOLECULAR_NSPDK in self.metrics_list:
            print("Computing nspdk stats..")

            # molecules -> graphs
            graphs: List[nx.Graph] = molecular.mols_to_nx(posthoc_molecules)

            nspdk = self.metrics_list[m_list.KEY_MOLECULAR_NSPDK](
                graph_ref_list = self.test_graphs,
                graph_pred_list = graphs
            )

            logs[m_list.KEY_MOLECULAR_NSPDK] = nspdk
            

        if m_list.KEY_MOLECULAR_FCD in self.metrics_list:
            print("Computing fcd stats...")

            # molecules -> smiles
            smiles = [mol2smiles(mol) for mol in posthoc_molecules if mol.GetNumAtoms() > 0]

            fcd = self.metrics_list[m_list.KEY_MOLECULAR_FCD](smiles)

            logs[m_list.KEY_MOLECULAR_FCD] = fcd


        if 'cond_mae' in self.metrics_list:
            print("Computing conditional mae stats...")

            # stack together the reference properties
            ys = [g.y for g in generated_graphs]
            reference_properties = torch.stack(ys, dim=0)

            # compute mae: compute properties on post-hoc molecules
            # and compare to true reference properties
            mae_per_prop = self.metrics_list['cond_mae'](posthoc_molecules, reference_properties)

            logs.update(mae_per_prop)


        ret = {KEY_SAMPLE_METRICS: logs}

        if ret_logs:
            return logs
        else:
            return ret



MOLECULAR_METRICS = {
    m_list.KEY_MOLECULAR_VUN: MoleculeVUNMetric,
    m_list.KEY_MOLECULAR_DISTRIBUTION: MoleculeDistributionMetric
}

import json

class MoleculeSamplingMetrics(BaseSamplingMetric):

    def __init__(self, datamodule: GraphDataModule, metrics: Dict[str, Dict], graph_to_mol_converter: GraphToMoleculeConverter):
        super().__init__()
        
        self.datamodule = datamodule
        self.graph_to_mol_converter = graph_to_mol_converter
        
        self.metrics = []
        stored_data_partitions = {}

        for metric_name, metric_kwargs in metrics.items():
            if metric_name in MOLECULAR_METRICS.keys():

                if isinstance(metric_kwargs, DictConfig):
                    metric_kwargs = OmegaConf.to_container(metric_kwargs)

                metric_args = []

                # provide metric with graph-to-molecule converter
                # this is done because each metric might need a different graph-to-molecule
                # converter process: e.g., post-hoc, relaxed, etc.
                metric_kwargs['graph_to_mol_converter'] = graph_to_mol_converter

                ###############  EXTRACT NEEDED DATA PARTITION  ################
                # the following replaces the data partitions (train/test/val) with the corresponding graphs
                # from that partition
                data_partition = metric_kwargs.pop('data_partition', None)

                if data_partition is None and hasattr(MOLECULAR_METRICS[metric_name], 'NEEDED_PARTITION'):
                    data_partition = MOLECULAR_METRICS[metric_name].NEEDED_PARTITION
                if isinstance(data_partition, str):
                    data_partition = [data_partition]
                if isinstance(data_partition, list):
                    for partition in data_partition:
                        if partition not in stored_data_partitions.keys():
                            stored_data_partitions[partition] = datamodule.load_file(partition, f'smiles_file_{partition}', json.load)
                        metric_args.append(stored_data_partitions[partition])

                ##############  APPEND METRIC TO LIST OF METRICS  ##############
                self.metrics.append(MOLECULAR_METRICS[metric_name](*metric_args, **metric_kwargs))
            else:
                raise ValueError(f"Metric {metric_name} not supported, choose one of {list(MOLECULAR_METRICS.keys())}")


    def __call__(self, generated_graphs: List, ret_logs: bool=True) -> Dict:

        if hasattr(generated_graphs, 'ptr'):
            generated_graphs = generated_graphs.to_data_list()

        logs = {}
        for metric in self.metrics:
            logs.update(metric(generated_graphs, ret_logs=ret_logs))
        return logs


KEY_SAMPLE_METRICS = 'sample_metrics'
KEY_UNIQUE_SMILES = 'unique'
KEY_NUM_COMPONENTS = 'num_components'
KEY_ALL_SMILES = 'all_smiles'


################################################################################
#                      GRAPH GENERATION SAMPLING METRICS                       #
################################################################################


class GraphVUNMetric(BaseSamplingMetric):

    NEEDED_PARTITION = ['train']

    def __init__(self, train_graphs: List[nx.Graph], graph_type: str):
        super().__init__()

        assert graph_type in synth.VALIDITY_FUNCTIONS.keys(), f"Graph type {graph_type} not supported, " \
                                                                f"choose one of {list(synth.VALIDITY_FUNCTIONS.keys())}"

        self.train_graphs = train_graphs
        self.validity_func = synth.VALIDITY_FUNCTIONS[graph_type]


    def __call__(self, generated_graphs: List[nx.Graph], ret_logs: bool=True) -> Dict:
        print(f"Computing VUN metrics between {len(generated_graphs)} generated graphs and {len(self.train_graphs)}")

        print("Computing all fractions...")
        uniqueness, uniqueness_novelty, vun = synth.eval_vun(
            fake_graphs = generated_graphs,
            train_graphs = self.train_graphs,
            validity_func = self.validity_func
        )

        logs = {
            m_list.KEY_GRAPH_UNIQUE: uniqueness,
            m_list.KEY_GRAPH_UNIQUE_NOVEL: uniqueness_novelty,
            m_list.KEY_GRAPH_VUN: vun
        }

        ret = {
            KEY_SAMPLE_METRICS: logs
        }

        if ret_logs:
            return logs
        else:
            return ret

from networkx import number_connected_components

class GraphConnCompMetric(BaseSamplingMetric):

    def __init__(self):
        super().__init__()


    def __call__(self, generated_graphs: List[nx.Graph], ret_logs: bool=True) -> Dict:
        
        conn_comps = [number_connected_components(g) for g in generated_graphs]

        logs = {
            m_list.KEY_GRAPH_CONN_COMP_MEAN: np.mean(conn_comps),
            m_list.KEY_GRAPH_CONN_COMP_MIN: np.min(conn_comps),
            m_list.KEY_GRAPH_CONN_COMP_MAX: np.max(conn_comps)
        }

        print(f"Connected components: {logs}")

        ret = {
            KEY_SAMPLE_METRICS: logs
        }

        if ret_logs:
            return logs
        else:
            return ret


class GraphStructureMetric(BaseSamplingMetric):

    NEEDED_PARTITION = ['test']

    def __init__(self, test_graphs: List[nx.Graph], metrics_list='all', compute_emd=True):
        super().__init__()

        self.test_graphs = test_graphs
        self.num_graphs_test = len(self.test_graphs)
        self.compute_emd = compute_emd
        if metrics_list == 'all':
            self.metrics_list = [
                m_list.KEY_GRAPH_DEGREE,
                m_list.KEY_GRAPH_SPECTRE,
                m_list.KEY_GRAPH_CLUSTERING,
                m_list.KEY_GRAPH_ORBIT
            ]
        else:
            self.metrics_list = metrics_list

    def __call__(self, generated_graphs: List, ret_logs: bool=True) -> Dict:
        print(f"Computing structural metrics between {len(generated_graphs)} generated graphs and {len(self.test_graphs)}"
                f" test graphs -- emd computation: {self.compute_emd}")

        logs = {}

        if m_list.KEY_GRAPH_DEGREE in self.metrics_list:
            print("Computing degree stats..")
            degree = synth.degree_stats(
                self.test_graphs,
                generated_graphs,
                is_parallel=True,
                compute_emd=self.compute_emd
            )

            logs[m_list.KEY_GRAPH_DEGREE] = degree
            

        if m_list.KEY_GRAPH_SPECTRE in self.metrics_list:
            print("Computing spectre stats...")
            spectre = synth.spectral_stats(
                self.test_graphs,
                generated_graphs,
                is_parallel=True,
                n_eigvals=-1,
                compute_emd=self.compute_emd
            )

            logs[m_list.KEY_GRAPH_SPECTRE] = spectre

        if m_list.KEY_GRAPH_CLUSTERING in self.metrics_list:
            print("Computing clustering stats...")
            clustering = synth.clustering_stats(
                self.test_graphs,
                generated_graphs,
                bins=100,
                is_parallel=True,
                compute_emd=self.compute_emd
            )
            
            logs[m_list.KEY_GRAPH_CLUSTERING] = clustering
        

        if m_list.KEY_GRAPH_ORBIT in self.metrics_list:
            print("Computing orbit stats...")
            orbit = synth.orbit_stats_all(
                self.test_graphs,
                generated_graphs,
                compute_emd=self.compute_emd
            )
            
            logs[m_list.KEY_GRAPH_ORBIT] = orbit

        ret = {KEY_SAMPLE_METRICS: logs}

        if ret_logs:
            return logs
        else:
            return ret
        

class GraphAlternativeMetric(BaseSamplingMetric):

    NEEDED_PARTITION = ['test']

    def __init__(self, test_graphs: List[nx.Graph], metrics_list='all', cfg=None):
        super().__init__()

        if cfg is None:
            cfg = OmegaConf.create({})

        self.test_graphs = test_graphs
        self.num_graphs_test = len(self.test_graphs)
        if metrics_list == 'all':
            metrics_list = ['structure', m_list.KEY_GRAPH_GIN]

        self.metrics_list = {}

        if 'structure' in metrics_list:
            self.metrics_list['structure'] = graphgdp_metrics.get_stats_eval(cfg)

        if m_list.KEY_GRAPH_GIN in metrics_list:
            self.metrics_list[m_list.KEY_GRAPH_GIN] = graphgdp_metrics.get_nn_eval(cfg)


    def __call__(self, generated_graphs: List, ret_logs: bool=True) -> Dict:
        print(f"Computing structural metrics between {len(generated_graphs)} generated graphs and {len(self.test_graphs)}")

        logs = {}

        if 'structure' in self.metrics_list:
            print("Computing structure stats..")
            structure_stats = self.metrics_list['structure'](
                test_dataset = self.test_graphs,
                pred_graph_list = generated_graphs
            )

            logs.update(structure_stats)
            

        if m_list.KEY_GRAPH_GIN in self.metrics_list:
            print("Computing nn stats...")
            nn_stats = self.metrics_list[m_list.KEY_GRAPH_GIN](
                test_dataset = self.test_graphs,
                pred_graph_list = generated_graphs
            )

            logs.update(nn_stats)

        ret = {KEY_SAMPLE_METRICS: logs}

        if ret_logs:
            return logs
        else:
            return ret


GRAPH_METRICS = {
    m_list.KEY_GRAPH_VUN_IN: GraphVUNMetric,
    m_list.KEY_GRAPH_STRUCTURE: GraphStructureMetric,
    m_list.KEY_GRAPH_STRUCTURE_ALT: GraphAlternativeMetric,
    m_list.KEY_GRAPH_CONN_COMP: GraphConnCompMetric
}

class GraphSamplingMetrics(BaseSamplingMetric):

    def __init__(self, datamodule, metrics: Dict[str, Dict]):
        super().__init__()
        self.datamodule = datamodule
        
        self.metrics = []
        stored_data_partitions = {}

        for metric_name, metric_kwargs in metrics.items():
            if metric_name in GRAPH_METRICS.keys():

                if isinstance(metric_kwargs, DictConfig):
                    metric_kwargs = OmegaConf.to_container(metric_kwargs)

                metric_args = []

                ###############  EXTRACT NEEDED DATA PARTITION  ################
                # the following replaces the data partitions (train/test/val) with the corresponding graphs
                # from that partition
                data_partition = metric_kwargs.pop('data_partition', None)

                if data_partition is None and hasattr(GRAPH_METRICS[metric_name], 'NEEDED_PARTITION'):
                    data_partition = GRAPH_METRICS[metric_name].NEEDED_PARTITION
                if isinstance(data_partition, str):
                    data_partition = [data_partition]
                if isinstance(data_partition, list):
                    for partition in data_partition:
                        if partition not in stored_data_partitions.keys():
                            stored_data_partitions[partition] = synth.dataloader_to_nx(datamodule.get_dataloader(partition))
                        metric_args.append(stored_data_partitions[partition])

                ##############  APPEND METRIC TO LIST OF METRICS  ##############
                self.metrics.append(GRAPH_METRICS[metric_name](*metric_args, **metric_kwargs))
            else:
                raise ValueError(f"Metric {metric_name} not supported, choose one of {list(GRAPH_METRICS.keys())}")


    def __call__(self, generated_graphs: List, ret_logs: bool=True) -> Dict:

        if hasattr(generated_graphs, 'ptr'):
            generated_graphs = generated_graphs.to_data_list()

        if isinstance(generated_graphs[0], Data):
            generated_graphs = [to_networkx(graph, to_undirected=True, remove_self_loops=True) for graph in generated_graphs]

        logs = {}
        for metric in self.metrics:
            logs.update(metric(generated_graphs, ret_logs=ret_logs))
        return logs
    

################################################################################
#                           PROCESS SAMPLING METRICS                           #
################################################################################

PROCESS_METRICS = [
    m_list.KEY_COMPUTATIONAL_TIME,
    m_list.KEY_COMPUTATIONAL_MEMORY
]

class ProcessSamplingMetrics(BaseSamplingMetric):

    def __init__(self, metrics: Dict[str, Dict]):
        super().__init__()

        self.metrics = {
            metric : metric in metrics.keys() for metric in PROCESS_METRICS
        }

    def __call__(self, data: Dict, ret_logs: bool=True) -> Dict:
            
        logs = {}

        m = m_list.KEY_COMPUTATIONAL_TIME
        if self.metrics[m]:
            logs[m] = data['time']['end'] - data['time']['start']

        m = m_list.KEY_COMPUTATIONAL_MEMORY
        if self.metrics[m]:
            logs[m] = data['memory']['peak']

        ret = {KEY_SAMPLE_METRICS: logs}

        if ret_logs:
            return logs
        else:
            return ret
        

################################################################################
#                         SUMMARIZING SAMPLING METRICS                         #
################################################################################

class SummarizingSamplingMetrics(BaseSamplingMetric):

    def __init__(self, weights: Dict[str, float]):
        super().__init__()

        self.weights = weights

    def __call__(self, logs: Dict, ret_logs: bool=True) -> Dict:

        value = sum([logs['sampling/'+k] * self.weights[k] for k in self.weights.keys()])
        
        logs = {'summarizing_metric': torch.tensor(value)}

        if ret_logs:
            return {'sampling/'+k: v for k, v in logs.items()}
        else:
            return logs