
import os
import pandas as pd
import numpy as np
import itertools

import models
import utils 

experiment_folder = "experiments/07-26-2021_10"
all_data = []
for root, dirs, files in os.walk(experiment_folder):
    for name in files:
        path = os.path.join(root, name)
        if "bad_runs" in path:
            continue

        if "results_all" in name:
            round_name = int(root.split("/")[-1].split("Round")[-1])
            if round_name > 13:
                continue
            results = utils.normalize_ingredient_names(
                pd.read_csv(path, index_col=None)
            )
            if "is_redo" in results.columns:
                results = results[~results["is_redo"]]
            
            all_data.append(results)

all_data = pd.concat(all_data, ignore_index=True).iloc[:, :20].to_numpy().tolist()
all_data = set([tuple(r) for r in all_data])

whole_set = list(itertools.product([0, 1], repeat=20))
whole_set = [r for r in whole_set if tuple(r) not in all_data]
whole_set = np.array(whole_set)
# whole_set = np.random.choice([0,1], (10000, 20))
# print(whole_set)

output = pd.DataFrame(whole_set)

net_folder = "experiments/07-26-2021_10/Round13/nn_models"
m = models.NeuralNetModel.load_trained_models(net_folder)
preds, variances = m.evaluate(whole_set)
# print(preds)
output["pred"] = preds
output["var"] = variances

output.to_csv("predict_whole_space.csv", index=None)
print(output.shape)
print("RESULT over 0.25:\n", (output["pred"] >= 0.25).sum())
