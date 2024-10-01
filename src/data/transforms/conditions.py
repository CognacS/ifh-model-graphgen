from typing import Dict, List
from abc import ABC

class Condition(ABC):

    def __call__(self, data: Dict) -> bool:
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f'Condition on {self.__class__.__name__}()'


class ManyConditions(Condition):

    def __init__(
            self,
            conditions: List[Condition]
        ):

        self.conditions = [
            cond for cond in conditions if cond is not None
        ]

    def __call__(self, data: Dict) -> bool:
        
        final_res = True

        for cond in self.conditions:
            final_res = final_res and cond(data)

            if not final_res:
                break

        return final_res
    
    def __repr__(self) -> str:
        args = [f'{cond}' for cond in self.conditions]
        args_list = ',\n'.join(args)
        return f'Conditions=[\n{args_list}\n]'


class CondInList(Condition):

    def __init__(
            self,
            obj_to_check_df: str,
            check_list_df: str,
            check = 'in'
        ):

        self.obj_to_check_df = obj_to_check_df
        self.check_list_df = check_list_df
        self.check = check
        

    def __call__(self, data: Dict) -> bool:
        
        # check that the data index is not to be skipped
        check_list = data[self.check_list_df]
        elem_to_check = data[self.obj_to_check_df]

        flag = (elem_to_check in check_list) ^ (self.check == 'not in')

        return flag
    

class CondNotNone(Condition):
    def __init__(
            self,
            obj_to_check_df: str
        ):
        
        self.obj_to_check_df = obj_to_check_df


    def __call__(self, data: Dict) -> bool:
        
        # check that the element is not none
        elem_to_check = data[self.obj_to_check_df]

        return elem_to_check is not None
    

class CondNotEqual(Condition):
    def __init__(
            self,
            obj_to_check_df: str,
            value
        ):
        
        self.obj_to_check_df = obj_to_check_df
        self.value = value if isinstance(value, list) else [value]


    def __call__(self, data: Dict) -> bool:
        
        # check that the element is not none
        elem_to_check = data[self.obj_to_check_df]

        return all([elem_to_check != v for v in self.value])