import copy
import csv
import datetime
import time
import itertools
import math 
import os
import random

import cobra
import reframed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
from tqdm import tqdm, trange

CDM_RXN_IDS = ["ac_exch",
              "ala_exch",
              "arg_exch",
              "asp_exch",
              "asn_exch",
              "cys_exch",
              "glu_exch",
              "gln_exch",
              "gly_exch",
              "his_exch",
              "ile_exch",
              "leu_exch",
              "lys_exch",
              "met_exch",
              "phe_exch",
              "pro_exch",
              "ser_exch",
              "thr_exch",
              "trp_exch",
              "tyr_exch",
              "val_exch",
              "ade_exch",
              "gua_exch",
              "ura_exch",
              "4abz_exch",
              "btn_exch",
              "fol_exch",
              "ncam_exch",
              "NADP_exch",
              "pnto_exch",
              "pydx_pydam_exch",
              "ribflv_exch",
              "thm_exch",
              "vitB12_exch",
              "FeNO3_exch",
              "MgSO4_exch",
              "MnSO4_exch",
              "CaCl2_exch",
              "NaBic_exch",
              "KPi_exch",
              "NaPi_exch"]


def load_cobra(model_path):
    """Load a CobraPy model.
    
    model_path: str
        Path to model.
    """ 
    model = cobra.io.read_sbml_model(model_path)
    return model
    

def random_reactions(num_to_remove=5):
    # num_to_remove - int: number of reactions to remove (set to 0)
    
    remove_indexes = np.random.choice(len(CDM_RXN_IDS), num_to_remove, replace=False)
    remove_arr = np.ones(len(CDM_RXN_IDS))
    remove_arr[remove_indexes] = 0
    return reactions_to_knockout(remove_arr, CDM_RXN_IDS), remove_arr

def get_LXO(n_reactions, X=1):
    # n_reactions - int: number of reactions
    # X - int: number to leave out for leave-X-out experiments
    
    all_indexes = np.arange(n_reactions)
    combos = itertools.combinations(all_indexes, X)
    remove_indexes = [list(c) for c in combos] 
    remove_arrs = list()
    for to_remove in remove_indexes:
        remove_arr = np.ones(n_reactions)
        remove_arr[to_remove] = 0
        remove_arrs.append(remove_arr)
    # print(remove_arrs)
    return remove_arrs

def knockout_and_simulate(model, num_to_remove, return_boolean=False):
    min_growth = 0.50 * model.slim_optimize()
    reactions, remove_arr = random_reactions(num_to_remove)
    grow, objective_value = reaction_knockout(model, reactions, min_growth)
    if return_boolean:
        return int(grow), remove_arr
    return objective_value, remove_arr
        

def reactions_to_knockout(remove_arr, reactions):
    # remove_arr - np.array[int]: binary array (0 = remove, 1 = keep)
    # reactions - [str]: list of reactions
    
    ones = np.where(remove_arr == 1)[0]
    reactions = np.delete(reactions, ones)
    return reactions
    
def reaction_knockout_cobra(model, reactions, growth_cutoff, dummy=None,
                            use_media=False, use_names=True):
    # model - cobrapy.Model: model with reactions to knockout
    # reactions - [str]: list of reactions to knockout
    # growth_cutoff - float: grow/no grow cutoff
    
    # model = copy.deepcopy(model)
    if dummy:
        model.add_reactions([dummy_rxn])

    if use_media:
        with model:
            medium = model.medium
            for reaction in reactions:
                medium[reaction] = 0.0
            model.medium = medium
            objective_value = model.slim_optimize()
    else:
        reaction_bounds = dict()
        for r in reactions:
            if use_names:
                reaction_bounds[r] = model.reactions.get_by_id(r).bounds
                model.reactions.get_by_id(r).bounds = (0,0)
            else:
                reaction_bounds[r] = r.bounds
                r.bounds = (0,0)
        objective_value = model.slim_optimize()
        for r, bounds in reaction_bounds.items():
            if use_names:
                model.reactions.get_by_id(r).bounds = bounds
            else:
                r.bounds = bounds
                
    
    grow = False if objective_value < growth_cutoff else True
    
    if dummy:
        model.reactions.get_by_id(dummy.name).remove_from_model()
    
    return objective_value, grow


def print_compartments():
    model = load_cobra("models/iSMUv01_CDM_LOO_v2.xml")
    print(model.compartments)
    
    print("############## BOUNDARY ##############")
    for rxn in model.boundary:
        print(rxn, rxn.bounds)
    print("\n\n")
    print("############## EXCHANGE ##############")
    for rxn in model.exchanges:
        print(rxn, rxn.bounds)
    print("\n\n")
    print("############## SINKS ##############")
    for rxn in model.sinks:
        print(rxn, rxn.bounds)
    print("\n\n")
    print("############## DEMANDS ##############")
    for rxn in model.demands:
        print(rxn, rxn.bounds)


def reaction_knockout(model, reactions, min_growth):
    """Knock outs reactions in a given model.
    
    Inputs
    ------
    model: cobrapy.Model
        Model to use when knocking out reactions.
    reactions: list[cobrapy.Reaction]
        Reactions in `model` to be knocked out.
    min_growth: float
        Threshold value for determining growth.
    
    Returns
    -------
    grow: boolean
        Returns `True` if the model's objective value after knocking out 
        reactions is larger that the `min_growth` threshold, and `False`
        otherwise.
    objective_value: float
        The objective value of `model` after knocking out the reactions.
        
    """
    
    with model:
        for rxn in reactions:
            rxn.knock_out()
        objective_value = model.slim_optimize()

    grow = True if objective_value > min_growth else False
    return grow, objective_value


def get_number_knocked_out(model):
    """Return number of reactions knocked out in a given model
    """
    return sum([1 if rxn.bounds == (0,0) else 0 for rxn in model.reactions])


def knockout_walk(model, valid_reactions, growth_threshold=0.50):
    """Performs a 'knockout walk' in which a random number of reactions are
    removed one at a time. Each cycle of the walk, a list of 'candidate
    reactions' is built, i.e. all of the reactions in `valid_reactions` that 
    are predicted to still lead to growth in `model` when removed. A single 
    random reaction is chosen from the candidates and is knocked out. The 
    cycle repeats.
    
        
    Inputs
    ------
    model: cobrapy.Model
        Model to use when knocking out reactions.
    valid_reactions: list[cobrapy.Reaction]
        Reactions in `model` to be knocked out.
    growth_threshold: float, default: 0.50
        Threshold percentage for determining growth.
    
    Returns
    -------
    valid_reactions: list[cobrapy.Reaction]
        A list of reactions which is a subset of `valid_reactions` and can be 
        knocked out in the future. Reactions that have been knocked out in this
        walk are the ones that have been removed.
    valid_reactions: list[cobrapy.Reaction]
        A list of reactions that were knocked out.
    """
    
    max_objective = model.slim_optimize()
    growth_cutoff = growth_threshold * max_objective
    
    num_knockouts = sp.poisson.rvs(5)
    print("Number of KOs:", num_knockouts)
    removed_reactions = list()
    for _ in range(num_knockouts):
        candidate_reactions = list()
        for rxn in valid_reactions:
            does_grow, _ = reaction_knockout(model, [rxn], growth_cutoff)
            if does_grow:
                candidate_reactions.append(rxn)
        
        reaction_to_remove = random.choice(candidate_reactions)
        model.reactions.get_by_id(reaction_to_remove.id).knock_out()
        # reaction_to_remove.knock_out()
        valid_reactions.remove(reaction_to_remove)
        removed_reactions.append(reaction_to_remove)
        print(f"\tREMOVED: {reaction_to_remove.id}, # KO'd: {get_number_knocked_out(model)}, # Valid RXNs: {len(valid_reactions)}")
    
    return valid_reactions, removed_reactions


def get_non_media_reactions(model):
    """Return a list of all reactions in `model` that are not the media 
    (CDM) reactions
    """
    
    with model:
        all_reactions = set(model.reactions)
        CDM_reactions = set(
            [model.reactions.get_by_id(id_) for id_ in CDM_RXN_IDS])
        non_media_reactions = all_reactions.difference(CDM_reactions)
    return non_media_reactions
    
    
def make_minimal_media_models(model_path, max_n=10):
    """Generate and save derivations of a CobraPy model (at location 
    `model_path`) and their respective minimal medias. 
    
    Each cycle, a 'knockout walk' is performed and a minimal media is calculated. 
    The cycle repeats, knocking out more reactions and saving a new derivation 
    of current model with its minimal media, only when the length of the 
    minimal media increases (as to not have many model derivations of the same
    length minimal medias). When critical reactions are eventually removed, the 
    current model derivation cannot grow. When this happens, then the model is 
    reset to its original state and the cycle continues until `max_n` is 
    reached.
    
    Inputs
    ------
    model_path: str
        Path to cobrapy.Model to use.
    max_n: int, default: 10
        The number of new model derivation to generate and save.
    
    Outputs
    -------
    CobraPy model derivations, along with a CSV listing their minimal media
    components and the reactions removed. They are stored in the parent folder
    of the parent model in a new directory structure:
        > `model_path` parent folder
            > '[model name]_[timestamp]'
                > length of minimal media
                    > '[model name]_[length of minimal media].xml'
                    > '[model name]_[length of minimal media]_info.csv'
                    
    Returns
    -------
    enclosing_folder: str
        The newly generated folder where all files are saved in.
    """
    
    model = load_cobra(model_path)
    max_objective = model.slim_optimize()
    growth_threshold = 0.90 * max_objective
    print("Growth threshold:", growth_threshold)
    
    parent_folder = "/".join(model_path.split("/")[:-1])
    timestamp = datetime.datetime.now().isoformat(sep='T', 
                                                  timespec='milliseconds')
    model_name = (model_path.split("/")[-1]).split(".")[0]
    enclosing_folder = os.path.join(parent_folder, f"{model_name}_{timestamp}")
    
    
    valid_reactions = get_non_media_reactions(model)
    removed_reactions = list()
    max_length_media = 0
    for _ in range(max_n):
        try:
            valid_reactions, removed = knockout_walk(model, valid_reactions)
            removed_reactions += removed
            minimal_medium = cobra.medium.minimal_medium(
                model, growth_threshold, minimize_components=True)
            current_length_media = len(minimal_medium)
        except Exception as e:
            print("\Model Failed: {}".format(str(e)))
            print("Resetting model...")
            max_length_media = 0
            model = load_cobra(model_path)
            valid_reactions = get_non_media_reactions(model)
            removed_reactions = list()
        else:
            if current_length_media > max_length_media:
                print("\n###########  Found New Minimum!  ############")
                max_length_media = current_length_media
                # reactions.append((current_length_media, None))
                folder = os.path.join(enclosing_folder, str(max_length_media))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                new_model_path = (
                    os.path.join(folder, 
                        f"{model_name}_{max_length_media}.xml"))
                cobra.io.write_sbml_model(model, new_model_path)
            
                new_csv_path = (
                    os.path.join(folder, 
                        f"{model_name}_{max_length_media}_info.csv"))
                with open(new_csv_path, 'w') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow(['Minimal Media', 'Removed Reactions'])
                    reaction_ids = [r.id for r in removed_reactions]
                    rows = itertools.zip_longest(
                        list(minimal_medium.index), reaction_ids)
                    writer.writerows(rows)
            print("Minimal media length:", len(minimal_medium))
            print("Media:", list(minimal_medium.index))
            
    return enclosing_folder

if __name__ == "__main__":
    # model = load_cobra("models/iSMUv01_CDM_LOO_v2.xml")
    # for rxn in model.reactions:
    #     if "CDM_exch" in rxn.id:
    #         print(rxn.id)            
    #         rxn.id = rxn.id[:-8] + "media_exch"
    #     print(rxn.id)
            
    # cobra.io.write_sbml_model(
    #     model, "models/iSMUv01_CDM_LOO_v2.xml")

    # bench()
    filepath = make_minimal_media_models("models/iSMUv01_CDM.xml", max_n=1)
    print("DONE!")