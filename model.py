import pprint
import copy
import time
import itertools

import cobra
import reframed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

PP = pprint.PrettyPrinter(indent=4)
KO_RXN_IDS = ["ac_CDM_exch",
              "ala_CDM_exch",
              "arg_CDM_exch",
              "asp_CDM_exch",
              "asn_CDM_exch",
              "cys_CDM_exch",
              "glu_CDM_exch",
              "gln_CDM_exch",
              "gly_CDM_exch",
              "his_CDM_exch",
              "ile_CDM_exch",
              "leu_CDM_exch",
              "lys_CDM_exch",
              "met_CDM_exch",
              "phe_CDM_exch",
              "pro_CDM_exch",
              "ser_CDM_exch",
              "thr_CDM_exch",
              "trp_CDM_exch",
              "tyr_CDM_exch",
              "val_CDM_exch",
              "ade_CDM_exch",
              "gua_CDM_exch",
              "ura_CDM_exch",
              "4abz_CDM_exch",
              "btn_CDM_exch",
              "fol_CDM_exch",
              "ncam_CDM_exch",
              "NADP_CDM_exch",
              "pnto_CDM_exch",
              "pydx_pydam_CDM_exch",
              "ribflv_CDM_exch",
              "thm_CDM_exch",
              "vitB12_CDM_exch",
              "FeNO3_CDM_exch",
              "MgSO4_CDM_exch",
              "MnSO4_CDM_exch",
              "CaCl2_CDM_exch",
              "NaBic_CDM_exch",
              "KPi_CDM_exch",
              "NaPi_CDM_exch"]

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
    
def reaction_knockout_cobra(model, reactions, growth_cutoff):
    # model - cobrapy.Model: model with reactions to knockout
    # reactions - [str]: list of reactions to knockout
    # growth_cutoff - float: grow/no grow cutoff
    
    reaction_bounds = dict()
    for r in reactions:
        reaction_bounds[r] = model.reactions.get_by_id(r).bounds
        model.reactions.get_by_id(r).bounds = (0,0)
    objective_value = modelcb.slim_optimize()
    for r, bounds in reaction_bounds.items():
        model.reactions.get_by_id(r).bounds = bounds
    grow = False if objective_value < growth_cutoff else True
    return reactions.tolist(), objective_value, grow

def reaction_knockout_reframed(model, reactions):
    reaction_bounds = dict()

    # solution = reframed.reaction_knockout(model, reactions)
    # print(solution)
    for r in reactions:
        reaction_bounds[r] = (model.reactions[r].lb,
                             model.reactions[r].ub)
        model.reactions[r].lb = 0
        model.reactions[r].ub = 0
    solution = reframed.FBA(model).fobj
    for r, bounds in reaction_bounds.items():
        model.reactions[r].lb = bounds[0]
        model.reactions[r].ub = bounds[1]   
    return solution
    
def timed_run(model_cobra, model_reframed, c_reactions, r_reactions):
    # COBRA BENCHMARK
    t_start = time.time()
    cobra_solution = reaction_knockout_cobra(model_cobra, c_reactions, 2)
    t1 = time.time()
    
    # REFRAMED BENCHMARK
    reframed_solution = reaction_knockout_reframed(model_reframed, r_reactions)
    reframed_solution = 0
    t2 = time.time()
    
    cobra_time = t1 - t_start
    reframed_time = t2 - t1
    return cobra_time, reframed_time, cobra_solution, reframed_solution

if __name__ == "__main__":
    t_start = time.time()
    modelcb = load_cobra("models/iSMUv01_CDM_LOO.xml")
    t1 = time.time()
    modelrf = load_reframed("models/iSMUv01_CDM_LOO.xml")
    t2 = time.time()
    print("\nLoad Times")
    print(f"cobra: {t1-t_start}, reframed: {t2-t_start}")
    
    # cobra_time = 0.0
    # reframed_time = 0.0
    # plot_points_c = list()
    # plot_points_r = list()
    # n = 41
    # for i in trange(len(KO_RXN_IDS)):
    #     avg_c = 0
    #     avg_r = 0
    #     n2 = n//len(KO_RXN_IDS)
    #     for _ in range(n2):
    #         c_reactions, r_reactions = random_reactions(num_to_remove=i)
    #         c, r, cs, rs = timed_run(modelcb, modelrf, c_reactions, r_reactions)
    #         cobra_time += c
    #         reframed_time += r
    #         avg_c += c
    #         avg_r += r
    #     plot_points_c.append(avg_c/n2)
    #     plot_points_r.append(avg_r/n2)
    #     # print(round(cs,6), round(rs,6))
    
    # print(f"\nTotal Time for {n} Runs")
    # print(f"cobra: {cobra_time}")
    # print(f"reframed: {reframed_time}")
    
    # plt.plot(range(len(KO_RXN_IDS)), plot_points_c, range(len(KO_RXN_IDS)), plot_points_r)
    # plt.title(f"Time vs. number of reactions removed")
    # # plt.title(f"cobra: {round(cobra_time,3)}, reframed: {round(reframed_time,3)}", fontsize=10)
    # plt.legend(["cobrapy", "reframed"])
    # plt.ylabel("Time (sec)")
    # plt.xlabel("Number of reactions removed")
    # plt.show()
    
    
    max_objective = modelcb.slim_optimize()
    growth_cutoff = 0.07 * max_objective
    
    cobra_time = 0.0
    reframed_time = 0.0
    plot_points_c = list()
    plot_points_r = list()
    n = 3
    no_growth_reactions = list()
    data_all = list()
    data_no_growth = list()
    for i in range(1, n):
        runs = get_LXO(len(KO_RXN_IDS), X=i)
        nested_no_growth = list()
        for j in tqdm(range(len(runs)), desc=f"L{i}Os", 
                      unit=" experiments", dynamic_ncols=True):
            knockouts = reactions_to_knockout(runs[j], KO_RXN_IDS)
            reactions, objective_value, grow = (
                reaction_knockout_cobra(modelcb, knockouts, growth_cutoff))
            if not grow:
                nested_no_growth.append((reactions, objective_value))
                data_no_growth.append([reactions, f"L{i}O", objective_value, grow])
            data_all.append([reactions, f"L{i}O", objective_value, grow])
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
    
    

