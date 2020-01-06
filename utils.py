
import itertools

import numpy as np

def get_LXO(n_reagents, X=1):
    # n_reactions - int: number of reactions
    # X - int: number to leave out for leave-X-out experiments
    
    all_indexes = np.arange(n_reagents)
    combos = itertools.combinations(all_indexes, X)
    remove_indexes = [list(c) for c in combos] 
    remove_arrs = np.empty((len(remove_indexes), n_reagents))
    for i, to_remove in enumerate(remove_indexes):
        remove_arr = np.ones(n_reagents)
        remove_arr[to_remove] = 0
        remove_arrs[i, :] = remove_arr
    return remove_arrs