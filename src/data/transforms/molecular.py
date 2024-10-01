from typing import Union, List, Dict, Callable, Optional

from . import DFBaseTransform
from .conditions import Condition
from .preprocessing import (
    CSV_TABLE
)

import torch
from tqdm import tqdm

import rdkit
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

from src.datatypes.sparse import SparseGraph
from src.data.simple_transforms.molecular import (
    GraphToMoleculeConverter,
    mol2smiles,
    BOND_TYPES
)
from src.data.simple_transforms.chemical_props import (
    PROPS_NAME_TO_FUNC
)




class MolecularPipelineException(Exception):
    pass


class DFApplyUnitConversion(DFBaseTransform):

    def __init__(
            self,
            datafield: str,
            conversion_factors: torch.Tensor
        ):

        self.datafield = datafield
        self.conversion_factors = conversion_factors

    
    def __call__(self, data: Dict) -> Dict:

        # get table of csv datafield
        table = data[self.datafield]
        table = torch.cat([table[:, 3:], table[:, :3]], dim=-1)

        # apply conversions to each row
        table = table * self.conversion_factors.view(1, -1)

        data[self.datafield] = table

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.datafield]
    @property
    def output_df_list(self) -> List[str]:
        return [self.datafield]
    
    def args_repr(self) -> str:
        return (
            f'factors shape={self.conversion_factors.shape}'
        )


class DFReadMolecules(DFBaseTransform):

    def __init__(
            self,
            datafield: str,
            new_datafield: str,
            sanitize: bool=False,
            remove_hydrogens: bool=False
        ):

        self.datafield = datafield
        self.new_datafield = new_datafield
        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens

    
    def __call__(self, data: Dict) -> Dict:

        filename: str = data[self.datafield]

        ############  molecules file  ############
        if filename.endswith('.sdf'):

            suppl = Chem.SDMolSupplier(
                filename,
                removeHs=self.remove_hydrogens,
                sanitize=self.sanitize
            )

        #############  smiles file  ##############
        elif filename.endswith('.smi'):
            
            suppl = Chem.SmilesSupplier(
                filename,
                removeHs=self.remove_hydrogens,
                sanitize=self.sanitize
            )

        else:
            raise NotImplementedError(
                f'Molecules supplier for file {filename} not implemented'
            )
        
        data[self.new_datafield] = suppl

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.datafield]
    @property
    def output_df_list(self) -> List[str]:
        return [self.new_datafield]
    
    def args_repr(self) -> str:
        return (
            f'sanitize={self.sanitize}\n'
            f'remove_hydrogens={self.remove_hydrogens}'
        )
    

class DFAtomsPositions(DFBaseTransform):
    pass


class DFMoleculePreprocess(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            kekulize: bool=False
        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df
        # apply kekulization
        self.kekulize = kekulize

    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        # get molecule
        mol = data[self.molecule_df]

        if self.kekulize:
            Chem.Kekulize(mol)

        # replace molecule
        data[self.molecule_df] = mol

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.molecule_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.molecule_df]
    
    def args_repr(self) -> str:
        return (
            f'kekulize={self.kekulize}'
        )
        

class DFAtomsData(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            result_nodes_df: str,
            atom_types_df: str
        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df

        # datafield where to put the resulting node features
        self.result_nodes_df = result_nodes_df

        # datafield from which to get atom types
        self.atom_types_df = atom_types_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        # check if atom types dictionary exists
        if self.atom_types_df not in data:
            data[self.atom_types_df] = dict()

        atom_types = data[self.atom_types_df]

        # get molecule
        mol = data[self.molecule_df]

        # collect atom types as indices
        type_idx = []

        for atom in mol.GetAtoms():
            
            # check if current atom is considered in
            # the overall dataset atom types
            atom_symb = atom.GetSymbol()
            if atom_symb not in atom_types:
                atom_types[atom_symb] = len(atom_types)
            
            # append atom type as an index
            type_idx.append(atom_types[atom_symb])

        # put resulting list of atoms into the resulting
        # datafield
        data[self.result_nodes_df] = torch.tensor(type_idx)

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.molecule_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.result_nodes_df, self.atom_types_df]
    

class DFBondsData(DFBaseTransform):

    def __init__(
            self,
            molecule_df: str,
            result_edge_index_df: str,
            result_edge_attr_df: str,
            bond_types_df: str
        ):

        # rdkit molecule datafield
        self.molecule_df = molecule_df

        # datafield where to put the resulting edge index
        # and edge attributes
        self.result_edge_index_df = result_edge_index_df
        self.result_edge_attr_df = result_edge_attr_df

        # datafield from which to get bond types
        self.bond_types_df = bond_types_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        # check if bond types dictionary exists
        if self.bond_types_df not in data:
            data[self.bond_types_df] = dict()

        bond_types = data[self.bond_types_df]

        # get molecule
        mol = data[self.molecule_df]

        # collect 
        row, edge_type = [], []

        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_type = str(bond.GetBondType())

            # check if current bond is considered in
            # the overall dataset bond types
            if bond_type not in bond_types:
                bond_types[bond_type] = len(bond_types)

            row.append([start, end])
            edge_type.append(bond_types[bond_type])

        # tranform to tensor
        if len(row) == 0: # if no edges, then set special case
            edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            edge_index = torch.tensor(row, dtype=torch.long).permute(1, 0)
        edge_attr = torch.tensor(edge_type, dtype=torch.long)

        # put resulting list of atoms into the resulting
        # datafield
        data[self.result_edge_index_df] = edge_index
        data[self.result_edge_attr_df] = edge_attr

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.molecule_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.result_edge_index_df, self.result_edge_attr_df]


class DFGraphToMol(DFBaseTransform):
    
        def __init__(
                self,
                graph_df: str,
                result_mol_df: str,
                atom_decoder_df: str,
                bond_decoder_df: str,
                relaxed: bool=False
            ):
    
            # graph datafield
            self.graph_df = graph_df
    
            # datafield where to put the resulting molecule
            self.result_mol_df = result_mol_df
    
            # datafield from which to get atom types
            self.atom_decoder_df = atom_decoder_df
            self.bond_decoder_df = bond_decoder_df
    
            # relaxed conversion
            self.relaxed = relaxed
    
    
        def __call__(
                self,
                data: Dict,
            ) -> Dict:
    
            # get graph
            graph = data[self.graph_df]
    
            # get decoder
            atom_decoder = data[self.atom_decoder_df]
            bond_decoder = data[self.bond_decoder_df]
    
            # convert graph to molecule
            converter = GraphToMoleculeConverter(
                atom_decoder =  atom_decoder,
                bond_decoder =  bond_decoder,
                relaxed =       self.relaxed
            )
            mol = converter(graph)
    
            # put resulting molecule into the resulting
            # datafield
            data[self.result_mol_df] = mol
    
            return data
        
        @property
        def input_df_list(self) -> List[str]:
            return [self.graph_df]
        @property
        def output_df_list(self) -> List[str]:
            return [self.result_mol_df]
        
        def args_repr(self) -> str:
            return (
                f'relaxed={self.relaxed}'
            )

class DFMolToSmiles(DFBaseTransform):

    def __init__(
            self,
            mol_df: str,
            smiles_df: str,
            sanitize_smiles: bool=True
        ):

        self.mol_df = mol_df
        self.smiles_df = smiles_df
        self.sanitize_smiles = sanitize_smiles


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        smiles = mol2smiles(data[self.mol_df], sanitize=self.sanitize_smiles)
        data[self.smiles_df] = smiles

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.mol_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.smiles_df]
    
    def args_repr(self) -> str:
        return (
            f'sanitize_smiles={self.sanitize_smiles}'
        )
    

class DFSmilesToMol(DFBaseTransform):
    
        def __init__(
                self,
                smiles_df: str,
                mol_df: str,
                sanitize: bool=True
            ):
    
            self.smiles_df = smiles_df
            self.mol_df = mol_df
            self.sanitize = sanitize
    
    
        def __call__(
                self,
                data: Dict,
            ) -> Dict:
    
            smiles = data[self.smiles_df]
            mol = Chem.MolFromSmiles(str(smiles), sanitize=self.sanitize)

            data[self.mol_df] = mol
    
            return data
        
        @property
        def input_df_list(self) -> List[str]:
            return [self.smiles_df]
        @property
        def output_df_list(self) -> List[str]:
            return [self.mol_df]
        
        def args_repr(self) -> str:
            return (
                f'sanitize={self.sanitize}'
            )
    



class DFToGraph(DFBaseTransform):

    def __init__(
            self,
            result_df: str,
            nodes_df: str,
            edge_index_df: str,
            edge_attr_df: str,
            targets_df: str,
            *others_df
        ):

        # datafield where to put the resulting graph
        self.result_df = result_df

        # main graph components
        self.nodes_df = nodes_df
        self.edge_index_df = edge_index_df
        self.edge_attr_df = edge_attr_df
        self.targets_df = targets_df

        # other datafields
        self.others_df = others_df


    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        y = data[self.targets_df]
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float)

        graph = SparseGraph(
            x=			data[self.nodes_df],
            edge_index=	data[self.edge_index_df],
            edge_attr=	data[self.edge_attr_df],
            y=			y,
            **{df: data[df] for df in self.others_df}
        )

        data[self.result_df] = graph

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [
            self.nodes_df,
            self.edge_index_df,
            self.edge_attr_df,
            self.targets_df
            ] + list(self.others_df)
    @property
    def output_df_list(self) -> List[str]:
        return [self.result_df]
    

class DFRemoveGraphsInvalidMol(DFBaseTransform):
    def __init__(
            self,
            graph_list_df: str,
            atom_decoder_df: str,
            bond_decoder_df: str=None,
            others_df: List[str]=None
        ):

        self.graph_list_df = graph_list_df
        self.atom_decoder_df = atom_decoder_df
        self.bond_decoder_df = bond_decoder_df
        self.others_df = others_df if others_df is not None else []

    
    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        converter = GraphToMoleculeConverter(
            atom_decoder =  data[self.atom_decoder_df],
            bond_decoder =  data[self.bond_decoder_df] if self.bond_decoder_df is not None else None,
            relaxed =       False
        )

        graph_list = data[self.graph_list_df]

        invalid = 0
        total = len(graph_list)

        pbar = tqdm(enumerate(graph_list), total=total)

        for i, graph in pbar:
            mol = converter(graph)
            smiles = mol2smiles(mol)

            if smiles is None or smiles == '':
                # remove graph
                for df in [self.graph_list_df] + self.others_df:
                    del data[df][i]
                invalid += 1

            pbar.set_postfix({'invalid': f'{invalid}/{total}'})

        return data
    

class DFComputeMolecularProperties(DFBaseTransform):
    def __init__(
            self,
            mol_df: str,
            graph_df: str,
            properties: List[str],
        ):

        self.mol_df = mol_df
        self.graph_df = graph_df
        self.properties = properties

    
    def __call__(
            self,
            data: Dict,
        ) -> Dict:

        mol = data[self.mol_df]
        prop_values = []

        for prop in self.properties:
            func = PROPS_NAME_TO_FUNC[prop]
            prop_values.append(func(mol))

        prop_values = torch.tensor(prop_values, dtype=torch.float)

        if hasattr(data[self.graph_df], 'y') and data[self.graph_df].y is not None:
            y = torch.cat([data[self.graph_df].y, prop_values], dim=-1)
        else:
            y = prop_values

        data[self.graph_df].y = y

        return data
    
    @property
    def input_df_list(self) -> List[str]:
        return [self.mol_df, self.graph_df]
    @property
    def output_df_list(self) -> List[str]:
        return [self.graph_df]
    
    def args_repr(self) -> str:
        return (
            f'properties={self.properties}'
        )
    

class CondMoleculeHasValidSMILES(Condition):
    def __init__(
            self,
            obj_to_check_df: str
        ):
        
        self.obj_to_check_df = obj_to_check_df


    def __call__(self, data: Dict) -> bool:
        
        # check if current molecule has a valid SMILES
        mol = data[self.obj_to_check_df]
        smiles = Chem.MolToSmiles(mol, canonical=True)

        return smiles is not None and smiles != ''
        
class CondValidSMILES(Condition):
    def __init__(
            self,
            obj_to_check_df: str
        ):
        
        self.obj_to_check_df = obj_to_check_df


    def __call__(self, data: Dict) -> bool:
        smiles = data[self.obj_to_check_df]

        return smiles is not None and smiles != ''