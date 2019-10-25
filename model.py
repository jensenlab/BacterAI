import cobra
import reframed
import pprint
import copy

pp = pprint.PrettyPrinter(indent=4)

def load_cobra(model_path):
    model = cobra.io.read_sbml_model(model_path)
    return model

def load_reframed(model_path):
    model = reframed.load_cbmodel(model_path, flavor='fbc2')
    return model

# def change_media_components(model, components):
#     # Components: dict({component : value})
#     model = copy.deepcopy(model)
#     if not isinstance(components, dict):
#         #raise error
#         pass
#     medium = model.medium
#     for c, v in components.items():
#         medium[c] = v
#     model.medium = medium
#     return model
    

if __name__ == "__main__":
    
    modelcb = load_cobra('models/iSMUv01_CDM_LOO.xml')
    # for r in modelcb.reactions:
    #     # print(r, r.objective_coefficient)
    #     print(r.lower_bound, r.upper_bound)
    # print("\n\nREACTIONS\n")
    # pp.pprint(modelcb.reactions)
    
    # print(len(modelcb.metabolites))
    # print(len(modelcb.genes))
    # print("\n\nSUMMARY\n",modelcb.summary())
    
    # print("\n\nMEDIA")
    # pp.pprint(modelcb.medium)
    solution = modelcb.slim_optimize()
    print(solution)
    
    modelrf = load_reframed('models/iSMUv01_CDM_LOO.xml')
    # objective = {r: 0 for r in modelrf.reactions}
    # objective['R_bio00001'] = 1.0
    # print(objective)
    
    
    # print(modelrf.get_objective())
    # solution = reframed.FBA(modelrf)
    
    # print(modelrf.summary())
    
    for reaction in modelrf.reactions.values():
        reaction.lb = -1000.0 if reaction.lb == float("-inf") else reaction.lb
        reaction.ub = 1000.0 if reaction.ub == float("inf") else reaction.ub
        # print(reaction.lb, reaction.ub)
    solution = reframed.FBA(modelrf)
    print(solution)

    
    # print(solution)
    # print(modelrf.reactions.R_R_PG__40__lys_type__41__)
    # print(modelrf.reactions.R_btn_ABC_trans)
    # print(modelrf.reactions.R_fol_ABC_trans)
    # print(modelrf.reactions.R_ribflv_ABC_trans)
    # print(modelrf.reactions.R_thm_ABC_trans)