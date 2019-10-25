import pprint
import copy
import time
import itertools

import cobra
import reframed
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

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
    model = cobra.io.read_sbml_model(model_path)
    return model

def load_reframed(model_path):
    model = reframed.load_cbmodel(model_path, 
                                  flavor='fbc2', 
                                  use_infinity=False, 
                                  load_gprs=False,
                                  reversibility_check=False,
                                  external_compartment=False,
                                  load_metadata=False)
    return model
    

def random_reagents(num_to_remove=5):
    remove_indexes = np.random.choice(len(KO_RXN_IDS), num_to_remove, replace=False)
    remove_arr = np.ones(len(KO_RXN_IDS))
    remove_arr[remove_indexes] = 0
    # remove_arr = np.random.randint(2, size=len(KO_RXN_IDS))
    return remove_reagents(remove_arr, KO_RXN_IDS), remove_reagents(remove_arr, KO_RXN_IDS_RF)

def get_LXO(x=1):
    all_indexes = np.arange(len(KO_RXN_IDS))
    combos = itertools.combinations(all_indexes, x)
    remove_indexes = [list(c) for c in combos] 
    remove_arrs = list()
    for to_remove in remove_indexes:
        remove_arr = np.ones(len(KO_RXN_IDS))
        remove_arr[to_remove] = 0
        remove_arrs.append(remove_arr)
    # print(remove_arrs)
    return remove_arrs

def remove_reagents(remove_arr, reagents):
    ones = np.where(remove_arr == 1)[0]
    reagents = np.delete(reagents, ones)
    return reagents
    
def timed_run(model_cobra, model_reframed, c_reagents, r_reagents):
    # COBRA BENCH
    t_start = time.time()
    # model_cobra = copy.deepcopy(model_cobra)
    # model_reframed = copy.deepcopy(model_reframed)
    
    c_reagent_bounds = dict()
    for r in c_reagents:
        c_reagent_bounds[r] = model_cobra.reactions.get_by_id(r).bounds
        model_cobra.reactions.get_by_id(r).bounds = (0,0)
    cobra_solution = modelcb.slim_optimize()
    
    for r, bounds in c_reagent_bounds.items():
        model_cobra.reactions.get_by_id(r).bounds = bounds
    t1 = time.time()
    
    #REFRAMED BENCH
    r_reagent_bounds = dict()
    for r in r_reagents:
        r_reagent_bounds[r] = (model_reframed.reactions[r].lb,
                               model_reframed.reactions[r].ub)
        model_reframed.reactions[r].lb = 0
        model_reframed.reactions[r].ub = 0
    reframed_solution = reframed.FBA(modelrf).fobj
    
    for r, bounds in r_reagent_bounds.items():
        model_reframed.reactions[r].lb = bounds[0]
        model_reframed.reactions[r].ub = bounds[1]
    t2 = time.time()
    
    cobra_time = t1 - t_start
    reframed_time = t2 - t1
    return cobra_time, reframed_time, cobra_solution, reframed_solution


if __name__ == "__main__":
    t_start = time.time()
    modelcb = load_cobra('models/iSMUv01_CDM_LOO.xml')
    t1 = time.time()
    modelrf = load_reframed('models/iSMUv01_CDM_LOO.xml')
    t2 = time.time()
    print("\nLoad Times")
    print(f"cobra: {t1-t_start}, reframed: {t2-t_start}")
    
    # cobra_time = 0.0
    # reframed_time = 0.0
    # plot_points_c = list()
    # plot_points_r = list()
    # n = 1000
    # for i in trange(len(KO_RXN_IDS)):
    #     avg_c = 0
    #     avg_r = 0
    #     n2 = n//len(KO_RXN_IDS)
    #     for _ in range(n2):
    #         c_reagents, r_reagents = random_reagents(num_to_remove=i)
    #         c, r, cs, rs = timed_run(modelcb, modelrf, c_reagents, r_reagents)
    #         cobra_time += c
    #         reframed_time += r
    #         avg_c += c
    #         avg_r += r
    #     plot_points_c.append(avg_c/n2)
    #     plot_points_r.append(avg_r/n2)
    #     # print(round(cs,6), round(rs,6))
    
    # print("\nTotal Time for {n} Runs")
    # print(f"cobra: {cobra_time}")
    # print(f"reframed: {reframed_time}")
    
    # plt.plot(range(len(KO_RXN_IDS)), plot_points_c, range(len(KO_RXN_IDS)), plot_points_r)
    # plt.title(f"Time vs. number of reagents removed")
    # # plt.title(f"cobra: {round(cobra_time,3)}, reframed: {round(reframed_time,3)}", fontsize=10)
    # plt.legend(['cobrapy', 'reframed'])
    # plt.ylabel('Time (sec)')
    # plt.xlabel('Number of reagents removed')
    # plt.show()

    
    cobra_time = 0.0
    reframed_time = 0.0
    for i in trange(1, 3):
        runs = get_LXO(i)
        for j in trange(len(runs)):
            c_reagents = remove_reagents(runs[j], KO_RXN_IDS)
            r_reagents = remove_reagents(runs[j], KO_RXN_IDS_RF)
            c, r, cs, rs = timed_run(modelcb, modelrf, c_reagents, r_reagents)
            cobra_time += c
            reframed_time += r

    
    
    print(f"cobra: {cobra_time}")
    print(f"reframed: {reframed_time}")
    

