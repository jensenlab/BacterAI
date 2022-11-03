import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from utils import normalize_ingredient_names
from constants import *

# experiment_name = "07-26-2021_10"
# friendly_name = "SGO CH1"

experiment_name = "08-20-2021_12"
friendly_name = "SSA SK36"

experiment_folder = f"../experiments/{experiment_name}"


def process_mapped_data_no_combine(path):
    data = pd.read_csv(path, index_col=None).fillna("")
    if "bad" not in data.columns:
        data["bad"] = False
        print("Added 'bad' column")

    data = normalize_ingredient_names(data)
    plate_control_indexes = data[data["plate_control"]].index
    plate_blank_indexes = data[data["plate_blank"]].index

    data["delta_od"] = data["final_od"] - data["initial_od"]

    leave_out_cols = [c for c in data.columns if "leave_out" in c]

    plate_controls = data.loc[plate_control_indexes, :].drop(columns=leave_out_cols)
    plate_control_means = (
        plate_controls.groupby("parent_plate").mean().to_dict()["delta_od"]
    )
    data["fitness"] = data["delta_od"] / data["parent_plate"].replace(
        plate_control_means
    )

    data = data.drop(data[(data["plate_control"] | data["plate_blank"])].index)
    data = data.drop(
        columns=[
            "plate_control",
            "plate_blank",
            "parent_well",
            "parent_well_index",
            "replicate",
            "solution_id_hex",
        ]
    )

    data_grouped = data.groupby(
        by=leave_out_cols + ["environment", "strain", "parent_plate"],
        as_index=False,
    )

    data = data_grouped.agg({"bad": "median", "delta_od": "count", "fitness": list})
    data = pd.concat(
        [pd.DataFrame(data["fitness"].values.tolist()), data.drop(columns="fitness")],
        axis=1,
    )

    cols = list(AA_SHORT) + list(data.columns)

    data = pd.concat(
        (pd.DataFrame(np.ones((data.shape[0], 20), dtype=int)), data),
        axis=1,
        ignore_index=True,
    )
    data.columns = cols
    for row_idx, row in data.iterrows():
        idxs = pd.unique([i for i in row[leave_out_cols] if i != ""])
        data.loc[row_idx, idxs] = 0

    data = data.drop(columns=leave_out_cols)
    return data


paths = []
for root, dirs, files in os.walk(experiment_folder):
    models = []
    for name in files:
        path = os.path.join(root, name)
        if "bad_runs" in path:
            continue
        if "mapped_data" in name:
            paths.append(path)

paths = sorted(paths, key=lambda x: (len(x), x), reverse=False)

all_results = []
for idx, path in enumerate(paths):
    print(path)
    data = process_mapped_data_no_combine(path)
    results = normalize_ingredient_names(data)
    if "is_redo" in results.columns:
        results = results[~results["is_redo"]]
    all_results.append(results)

all_results = pd.concat(all_results, ignore_index=True)

all_results = all_results.rename(
    columns={
        "var": "growth_pred_var",
        "depth": "num_aa_removed",
        0: "fitness_1",
        1: "fitness_2",
        2: "fitness_3",
    }
)

# all_results["median_fitness"] = all_results[
#     ["fitness_1", "fitness_2", "fitness_3"]
# ].median(axis=1)

print(data.shape)

fig, axs = plt.subplots(
    nrows=1,
    ncols=4,
    sharex=False,
    sharey=False,
    figsize=(14, 4),
)

axs[0].scatter(
    all_results["fitness_1"], all_results["fitness_2"], color="k", s=1, alpha=0.2
)
axs[0].plot([0, 1.2], [0, 1.2], color="r", alpha=0.2)
axs[0].set_title("Replicate 1 vs. 2")
axs[0].set_xlabel("1")
axs[0].set_ylabel("2")

axs[1].scatter(
    all_results["fitness_1"], all_results["fitness_3"], color="k", s=1, alpha=0.2
)
axs[1].plot([0, 1.2], [0, 1.2], color="r", alpha=0.2)
axs[1].set_title("Replicate 1 vs. 3")
axs[1].set_xlabel("1")
axs[1].set_ylabel("3")

axs[2].scatter(
    all_results["fitness_2"], all_results["fitness_3"], color="k", s=1, alpha=0.2
)
axs[2].plot([0, 1.2], [0, 1.2], color="r", alpha=0.2)
axs[2].set_title("Replicate 2 vs. 3")
axs[2].set_xlabel("2")
axs[2].set_ylabel("3")

plt.suptitle(f"{friendly_name} - {experiment_name}")
plt.tight_layout()
plt.savefig(f"replicate_comparison_{experiment_name}.png", dpi=400)


no_grows_12 = len(
    all_results[(all_results["fitness_1"] < 0.25) & (all_results["fitness_2"] < 0.25)]
)
grows_12 = len(
    all_results[(all_results["fitness_1"] >= 0.25) & (all_results["fitness_2"] >= 0.25)]
)
correct_12 = (no_grows_12 + grows_12) / len(all_results)

no_grows_13 = len(
    all_results[(all_results["fitness_1"] < 0.25) & (all_results["fitness_3"] < 0.25)]
)
grows_13 = len(
    all_results[(all_results["fitness_1"] >= 0.25) & (all_results["fitness_3"] >= 0.25)]
)
correct_13 = (no_grows_13 + grows_13) / len(all_results)

no_grows_23 = len(
    all_results[(all_results["fitness_2"] < 0.25) & (all_results["fitness_3"] < 0.25)]
)
grows_23 = len(
    all_results[(all_results["fitness_2"] >= 0.25) & (all_results["fitness_3"] >= 0.25)]
)
correct_23 = (no_grows_23 + grows_23) / len(all_results)


print(f"correct_12: {correct_12*100: .2f}%")
print(f"correct_13: {correct_13*100: .2f}%")
print(f"correct_23: {correct_23*100: .2f}%")
