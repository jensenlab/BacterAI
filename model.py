import pprint
import copy
import time
import itertools
import math 
import random

import cobra
import reframed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

PP = pprint.PrettyPrinter(indent=4)
# KO_RXN_IDS = ["ac_CDM_exch",
#               "ala_CDM_exch",
#               "arg_CDM_exch",
#               "asp_CDM_exch",
#               "asn_CDM_exch",
#               "cys_CDM_exch",
#               "glu_CDM_exch",
#               "gln_CDM_exch",
#               "gly_CDM_exch",
#               "his_CDM_exch",
#               "ile_CDM_exch",
#               "leu_CDM_exch",
#               "lys_CDM_exch",
#               "met_CDM_exch",
#               "phe_CDM_exch",
#               "pro_CDM_exch",
#               "ser_CDM_exch",
#               "thr_CDM_exch",
#               "trp_CDM_exch",
#               "tyr_CDM_exch",
#               "val_CDM_exch",
#               "ade_CDM_exch",
#               "gua_CDM_exch",
#               "ura_CDM_exch",
#               "4abz_CDM_exch",
#               "btn_CDM_exch",
#               "fol_CDM_exch",
#               "ncam_CDM_exch",
#               "NADP_CDM_exch",
#               "pnto_CDM_exch",
#               "pydx_pydam_CDM_exch",
#               "ribflv_CDM_exch",
#               "thm_CDM_exch",
#               "vitB12_CDM_exch",
#               "FeNO3_CDM_exch",
#               "MgSO4_CDM_exch",
#               "MnSO4_CDM_exch",
#               "CaCl2_CDM_exch",
#               "NaBic_CDM_exch",
#               "KPi_CDM_exch",
#               "NaPi_CDM_exch"]

KO_RXN_IDS = ["ac_media_exch",
              "ala_media_exch",
              "arg_media_exch",
              "asp_media_exch",
              "asn_media_exch",
              "cys_media_exch",
              "glu_media_exch",
              "gln_media_exch",
              "gly_media_exch",
              "his_media_exch",
              "ile_media_exch",
              "leu_media_exch",
              "lys_media_exch",
              "met_media_exch",
              "phe_media_exch",
              "pro_media_exch",
              "ser_media_exch",
              "thr_media_exch",
              "trp_media_exch",
              "tyr_media_exch",
              "val_media_exch",
              "ade_media_exch",
              "gua_media_exch",
              "ura_media_exch",
              "4abz_media_exch",
              "btn_media_exch",
              "fol_media_exch",
              "ncam_media_exch",
              "NADP_media_exch",
              "pnto_media_exch",
              "pydx_pydam_media_exch",
              "ribflv_media_exch",
              "thm_media_exch",
              "vitB12_media_exch",
              "FeNO3_media_exch",
              "MgSO4_media_exch",
              "MnSO4_media_exch",
              "CaCl2_media_exch",
              "NaBic_media_exch",
              "KPi_media_exch",
              "NaPi_media_exch"]

KO_RXN_IDS_RF = ["R_ac_CDM_exch",
              "R_ala_CDM_exch",
              "R_arg_CDM_exch",
              "R_asp_CDM_exch",
              "R_asn_CDM_exch",
              "R_cys_CDM_exch",
              "R_glu_CDM_exch",
              "R_gln_CDM_exch",
              "R_gly_CDM_exch",
              "R_his_CDM_exch",
              "R_ile_CDM_exch",
              "R_leu_CDM_exch",
              "R_lys_CDM_exch",
              "R_met_CDM_exch",
              "R_phe_CDM_exch",
              "R_pro_CDM_exch",
              "R_ser_CDM_exch",
              "R_thr_CDM_exch",
              "R_trp_CDM_exch",
              "R_tyr_CDM_exch",
              "R_val_CDM_exch",
              "R_ade_CDM_exch",
              "R_gua_CDM_exch",
              "R_ura_CDM_exch",
              "R_4abz_CDM_exch",
              "R_btn_CDM_exch",
              "R_fol_CDM_exch",
              "R_ncam_CDM_exch",
              "R_NADP_CDM_exch",
              "R_pnto_CDM_exch",
              "R_pydx_pydam_CDM_exch",
              "R_ribflv_CDM_exch",
              "R_thm_CDM_exch",
              "R_vitB12_CDM_exch",
              "R_FeNO3_CDM_exch",
              "R_MgSO4_CDM_exch",
              "R_MnSO4_CDM_exch",
              "R_CaCl2_CDM_exch",
              "R_NaBic_CDM_exch",
              "R_KPi_CDM_exch",
              "R_NaPi_CDM_exch"]

def load_cobra(model_path):
    # model_path - str: path to model
    model = cobra.io.read_sbml_model(model_path)
    return model

def load_reframed(model_path):
    # model_path - str: path to model
    
    model = reframed.load_cbmodel(model_path, 
                                  flavor="fbc2", 
                                  use_infinity=False, 
                                  load_gprs=False,
                                  reversibility_check=False,
                                  external_compartment=False,
                                  load_metadata=False)
    return model
    

def random_reactions(num_to_remove=5):
    # num_to_remove - int: number of reactions to remove (set to 0)
    
    remove_indexes = np.random.choice(len(KO_RXN_IDS), num_to_remove, replace=False)
    remove_arr = np.ones(len(KO_RXN_IDS))
    remove_arr[remove_indexes] = 0
    return reactions_to_knockout(remove_arr, KO_RXN_IDS), reactions_to_knockout(remove_arr, KO_RXN_IDS_RF)

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



def reactions_to_knockout(remove_arr, reactions):
    # remove_arr - np.array[int]: binary array (0 = remove, 1 = keep)
    # reactions - [str]: list of reactions
    
    ones = np.where(remove_arr == 1)[0]
    reactions = np.delete(reactions, ones)
    return reactions
    
def reaction_knockout_cobra(model, reactions, growth_cutoff, dummy=None,
                            use_media=False):
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
            reaction_bounds[r] = model.reactions.get_by_id(r).bounds
            model.reactions.get_by_id(r).bounds = (0,0)
        objective_value = model.slim_optimize()
        for r, bounds in reaction_bounds.items():
            model.reactions.get_by_id(r).bounds = bounds
    
    grow = False if objective_value < growth_cutoff else True
    
    if dummy:
        model.reactions.get_by_id(dummy.name).remove_from_model()
    
    return objective_value, grow

def reaction_knockout_reframed(model, reactions, growth_cutoff):
    reaction_bounds = dict()
    # model = copy.deepcopy(model)

    # solution = reframed.reaction_knockout(model, reactions)
    # print(solution)
    for r in reactions:
        reaction_bounds[r] = (model.reactions[r].lb,
                             model.reactions[r].ub)
        model.reactions[r].lb = 0
        model.reactions[r].ub = 0
    objective_value = reframed.FBA(model).fobj
    for r, bounds in reaction_bounds.items():
        model.reactions[r].lb = bounds[0]
        model.reactions[r].ub = bounds[1]
    grow = False if objective_value < growth_cutoff else True
    return objective_value, grow
    
def timed_run(model_cobra, model_reframed, c_reactions, r_reactions, 
              growth_cutoff, dummy=None):
    # COBRA BENCHMARK
    # Default
    t_start = time.time()
    cobra_solution, _ = reaction_knockout_cobra(model_cobra, c_reactions, 
                                             growth_cutoff)
    t1 = time.time()
    
    # Adding dummy to force model rebuild
    cobra_solution, _ = reaction_knockout_cobra(model_cobra, c_reactions, 
                                             growth_cutoff, dummy=dummy)
    t2 = time.time()
    
    # Using media instead of exch reaction knockout
    cobra_solution = reaction_knockout_cobra(model_cobra, c_reactions, 
                                             growth_cutoff, 
                                             use_media=True)
    t3 = time.time()

    # REFRAMED BENCHMARK
    reframed_solution, _ = reaction_knockout_reframed(model_reframed, r_reactions, 
                                                   growth_cutoff)
    t4 = time.time()
    
    cobra_time_1 = t1 - t_start
    cobra_time_2 = t2 - t1
    cobra_time_3 = t3 - t2
    reframed_time = t4 - t3
    return cobra_time_1, cobra_time_2, cobra_time_3, reframed_time, cobra_solution, reframed_solution


def bench():
    t_start = time.time()
    modelcb = load_cobra("models/iSMUv01_CDM_LOO.xml")
    t1 = time.time()
    modelrf = load_reframed("models/iSMUv01_CDM_LOO.xml")
    t2 = time.time()
    print("\nLoad Times")
    print(f"cobra: {t1-t_start}, reframed: {t2-t_start}")
    
    dummy_rxn = cobra.Reaction('DUMMY')
    dummy_rxn.name = 'DUMMY'
    dummy = cobra.Metabolite(
        'dummy',
        formula='H30',
        name='water',
        compartment='e')
    
    dummy_rxn.add_metabolites({
        dummy: -1.0,
        dummy: 1.0,
    })
    
    dummy_rxn.gene_reaction_rule = '( DUMMY_GENE )'
    
    max_objective = modelcb.slim_optimize()
    growth_cutoff = 0.07 * max_objective
    
    
    
    # cobra_time1 = 0.0
    # cobra_time2 = 0.0
    # cobra_time3 = 0.0
    # reframed_time = 0.0
    # n = 100
    # for i in trange(len(KO_RXN_IDS)):
    #     n2 = n//len(KO_RXN_IDS)
    #     for j in range(n2):
    #         c_reactions, r_reactions = random_reactions(num_to_remove=j)
    #         c1, c2, c3, r, cs, rs = timed_run(modelcb, modelrf, 
    #                                  c_reactions, r_reactions,
    #                                  growth_cutoff, dummy_rxn)
    #         cobra_time1 += c1
    #         cobra_time2 += c2
    #         cobra_time3 += c3
    #         reframed_time += r
    #         # print(round(cs,6), round(rs,6))

    
    # print(f"\nTotal Time for {n} Runs")
    # print(f"cobra default: {cobra_time1}")
    # print(f"cobra dummy: {cobra_time2}")
    # print(f"cobra media: {cobra_time3}")
    # print(f"reframed: {reframed_time}")
    

    cobra_time = 0.0
    reframed_time = 0.0
    plot_points_c = list()
    plot_points_r = list()
    n = 2
    no_growth_reactions = list()
    data_all = list()
    data_no_growth = list()
    for i in range(1, n):
        runs = get_LXO(len(KO_RXN_IDS), X=i)
        nested_no_growth = list()
        for j in tqdm(range(len(runs)), desc=f"L{i}Os", 
                      unit=" experiments", dynamic_ncols=True):
            knockouts = reactions_to_knockout(runs[j], KO_RXN_IDS)
            objective_value, grow = (
                reaction_knockout_cobra(modelcb, knockouts, growth_cutoff))
            if not grow:
                nested_no_growth.append((knockouts, objective_value))
                data_no_growth.append([knockouts, f"L{i}O", objective_value, grow])
            data_all.append([knockouts, f"L{i}O", objective_value, grow])
        no_growth_reactions.append(nested_no_growth)

    results_all = pd.DataFrame(data_all, 
                               columns=["Reactions", "Experiment", 
                                        "Objective Value", "Growth"])
    results_no_growth = pd.DataFrame(data_no_growth, 
                                     columns=["Reactions", "Experiment", 
                                              "Objective Value", "Growth"])
    
    results_no_growth.to_csv("data/no_growth.csv", index=False)
    print(f"cobra: {cobra_time}")
    print(f"reframed: {reframed_time}")
    # print(results_all)
    
    
    print("\n\nL1O")    
    print(no_growth_reactions[0])
    print("\n\nL2O")
    print(no_growth_reactions[1])

def set_media(model, media_reactions):
    medium = model.medium
    medium.update({rxn: 1000 for rxn in media_reactions})
    model.medium = medium
    return model
    
def knockout_walk(model, ignore_reactions=None):
    n = 50
    def _poisson(L, K):
        return math.pow(L, K) * math.exp(-L) / math.factorial(K)
    
    num_knockouts = 1
    # while num_knockouts == 0:
    #     L = random.randint(1, n)
    #     K = random.randint(1, n)
    #     num_knockouts = int(_poisson(L, K) * n)
    # print(num_knockouts)

    all_reactions = set(model.reactions)
    CDM_reactions = set([model.reactions.get_by_id(id) 
                         for id in KO_RXN_IDS])
    valid_reactions = all_reactions.difference(CDM_reactions)
    if ignore_reactions:
        valid_reactions = valid_reactions.difference(
            set(ignore_reactions))
    valid_reactions = list(valid_reactions)
    removed_reactions = random.sample(valid_reactions, k=num_knockouts)
    # print(reactions)
    for rxn in removed_reactions:
        print(f"\t{rxn.id}")
        # rxn.remove_from_model(remove_orphans=False)
        rxn.bounds = (0, 0)
    
    all_removed_reactions = removed_reactions + ignore_reactions
    return model, all_removed_reactions

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

def find_minimal_media():
    model = load_cobra("models/iSMUv01_CDM_LOO_v2.xml")
    model_backup = load_cobra("models/iSMUv01_CDM_LOO_v2.xml")
    
    max_objective = model.slim_optimize()
    max_growth = 0.90 * max_objective
    print("Max growth:", max_growth)
    
    removed_reactions = list()
    previous_length_media = 0
    reactions = list()
    for _ in range(1000):
        print(removed_reactions)
        model, removed_reactions = knockout_walk(model, 
                                                 removed_reactions)
        try:
            minimal_medium = cobra.medium.minimal_medium(model, 
                                        max_growth,
                                        minimize_components=True)
            current_length_media = len(minimal_medium)
        except:
            print("Reverting to backup.")
            previous_length_media = 0
            removed_reactions = list()
            model = copy.deepcopy(model_backup)
        else:
            if current_length_media > previous_length_media:
                print("Found New Minimum!")
                previous_length_media = current_length_media
                # reactions = (current_length_media, model.reactions)
                reactions.append((current_length_media, None))
                
            print("Minimal media:", len(minimal_medium))
    
    return reactions


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
    reactions = find_minimal_media()
    print("DONE!")
    print(reactions)