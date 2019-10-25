from cobra.flux_analysis import (
    single_gene_deletion, single_reaction_deletion, double_gene_deletion,
    double_reaction_deletion)
from cobra.io import read_sbml_model
# List of CDM components exchange reactions

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
                
# Simulate LOOs

def load_cobra(model_path):
    model = read_sbml_model(model_path)
    return model


if __name__ == "__main__":
    model = load_cobra('models/iSMUv01_CDM_LOO.xml')
    solution = model.optimize()
    
    LOO_dataframe = single_reaction_deletion(model, KO_RXN_IDS)
    print(LOO_dataframe)
    LOO_growth_defect = []  # This list will include components that if removed there will be no growth
    LOO_no_growth_defect = [] # This list will include components that if removed growth is not affected

    for i in range(len(LOO_dataframe)):
    
        rxn = list(LOO_dataframe.index[i])[0]
        if LOO_dataframe["growth"][i] < 0.07*solution.objective_value:
            LOO_growth_defect.append(rxn)
        else:
            LOO_no_growth_defect.append(rxn)

    print(LOO_growth_defect)

    # Simulate L2Os

    L2O_dataframe = double_reaction_deletion(model, LOO_no_growth_defect)

    L2O_growth_defect = []
    L2O_no_growth_defect = []

    for i in range(len(L2O_dataframe)):
        reactions = list(L2O_dataframe.index[i])
        if len(reactions) == 2:   
            double_rxn = f"{reactions[0]} + {reactions[0]}"
        else:
            double_rxn = f"{reactions[0]}"            
        if L2O_dataframe["growth"][i] < 0.07*solution.objective_value:
            L2O_growth_defect.append(double_rxn)
        else:
            L2O_no_growth_defect.append(double_rxn)
            
    print(L2O_growth_defect)