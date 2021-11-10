import collections
import os
from math import comb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch

import net
import utils


def main(folder):
    max_round_n = 14
    folders = [
        os.path.join(folder, i, "results_all.csv")
        for i in os.listdir(folder)
        if "Round" in i
    ]
    folders = sorted(folders, key=lambda x: (len(x), x))[:max_round_n]

    print(folders)
    output_path = os.path.join(folder, "summary")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    all_round_data = []
    for i, f in enumerate(folders):
        round_data = pd.read_csv(f, index_col=None)
        # round_data.append(df)
        # type_counts = dict(collections.Counter(df["type"].values.tolist()))
        # print(type_counts)

        # round_data = pd.concat(round_data, ignore_index=True)
        # round_data = pd.concat(round_data, ignore_index=True)
        round_data = round_data.drop(
            columns=[
                "var",
                "environment",
                "strain",
                "parent_plate",
                "initial_od",
                "final_od",
                "bad",
                "delta_od",
            ]
        )
        all_round_data.append(round_data)

        # round_output = os.path.join(output_path, f"Round{i+1}")
        # if not os.path.exists(round_output):
        #     os.makedirs(round_output)

        # round_data_grouped = round_data.groupby(by=["type"])
        # threshold = 0.25
        # for group_type, df in round_data_grouped:
        #     results = count(df, threshold)

        #     # print(depth_counts)
        #     grows = df[df["fitness"] >= threshold]
        #     grows = grows.sort_values(by=["depth", "fitness"], ascending=[False, False])

        #     # grows.to_csv(
        #     #     os.path.join(round_output, f"{group_type}_grows.csv"),
        #     #     index=False,
        #     # )
        #     results.to_csv(
        #         os.path.join(round_output, f"summarize_{group_type}_results.csv")
        #     )

        # results_all = count(round_data, threshold)
        # results_all.to_csv(os.path.join(round_output, f"summarize_ALL_results.csv"))

    all_round_data = pd.concat(all_round_data, ignore_index=True)
    all_round_data = all_round_data.iloc[:, :20]
    all_round_data["sum"] = all_round_data.iloc[:, :20].sum(axis=1)
    sum_counts = dict(collections.Counter(all_round_data["sum"].values.tolist()))
    print(all_round_data.shape)
    print(sum_counts)
# def collect_data(folder):
#     files = [f for f in os.listdir(folder) if "mapped_data" in f]
#     dfs = [utils.process_mapped_data(os.path.join(folder, f))[0] for f in files]
#     df = pd.concat(dfs, ignore_index=True)
#     df = df.replace(
#         {"Anaerobic Chamber @ 37 C": "anaerobic", "5% CO2 @ 37 C": "aerobic"}
#     )
#     df = df[df["environment"] == "aerobic"]
#     df = df.drop(
#         columns=["strain", "parent_plate", "initial_od", "final_od", "bad", "delta_od"]
#     )
#     df.to_csv(os.path.join(folder, "SGO CH1 Processed-Aerobic.csv"), index=False)


if __name__ == "__main__":
    # f = "experiments/05-31-2021_7"
    # f = "experiments/05-31-2021_8"
    # f = "experiments/05-31-2021_7 copy"
    f = "experiments/07-26-2021_10"
    # f = "experiments/07-26-2021_11"

    main(f)

    # collect_data("data/SGO_data")
