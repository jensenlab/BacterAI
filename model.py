import cobra
import reframed
import pprint
import copy

pp = pprint.PrettyPrinter(indent=4)

def load_cobra(model_path):
    model = cobra.io.load_matlab_model(model_path)
    return model

def load_reframed(model_path):
    model = reframed.load_cbmodel(model_path)
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
    
    modelcb = load_cobra('models/iSMUv01.mat')
    
    # print("\n\nREACTIONS\n")
    # pp.pprint(modelcb.reactions)
    
    # print(len(modelcb.metabolites))
    # print(len(modelcb.genes))
    # print("\n\nSUMMARY\n",modelcb.summary())
    
    # print("\n\nMEDIA")
    # pp.pprint(modelcb.medium)
    
    
    modelrf = load_reframed('models/iSMUv01.xml')
    solution = reframed.FBA(modelrf)
    print(solution)
    print(modelrf.reactions.R_R_PG__40__lys_type__41__)
    print(modelrf.reactions.R_btn_ABC_trans)
    