import cobra



def open_model(model_path):
    model = cobra.io.load_matlab_model(model_path)
    return model  

if __name__ == "__main__":
    
    model = open_model('models/iSMUv01.mat')
    solution = model.optimize()
    print(solution)
    print(len(model.reactions))
    print(len(model.metabolites))
    print(len(model.genes))