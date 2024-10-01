from typing import List



def prepare_skip_list(filepath: str) -> List[int]:
    with open(filepath, 'r') as f:
        skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

    return skip