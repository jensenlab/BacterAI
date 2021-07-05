import collections
import os
from math import comb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch

import net


def plot_model_performance(experiment_folder):

    models_in_rounds = {}
    training_data_in_rounds = {}
    for root, dirs, files in os.walk(experiment_folder):
        models = []
        for name in files:
            path = os.path.join(root, name)
            if "bag_model" in name:
                model = torch.load(path).cuda()
                models.append(model)
                # print(path)
            if "gpr_train_pred" in name:
                round_name = root.split("/")[-1]
                training_data_in_rounds[round_name] = pd.read_csv(path, index_col=None)

        if models:
            round_name = root.split("/")[-2]
            models_in_rounds[round_name] = models

    # print(training_data_in_rounds.values())
    data_path = "L1IO-L2IO-L3O All Rands SMU UA159 Processed-Aerobic.csv"
    data = pd.read_csv(data_path, index_col=None)
    X = data.iloc[:, :20].to_numpy()
    y = data["growth"].to_numpy()

    round_predictions = []
    n_rounds = len(models_in_rounds)
    round_names = sorted(list(models_in_rounds.keys()))
    for name in round_names:
        models = models_in_rounds.get(name, None)
        if models:
            predictions, variances = net.eval_bagged(X, models)
            round_predictions.append((predictions, variances))

    fig, axs = plt.subplots(
        nrows=2, ncols=n_rounds, sharex=False, sharey="row", figsize=(18, 6)
    )

    for i, (name, (pred, var)) in enumerate(zip(round_names, round_predictions)):

        order = np.argsort(y)
        x = np.arange(len(pred))
        mse = mean_squared_error(y, pred)
        axs[0, i].plot(x, pred[order], ".", alpha=0.10, markersize=1)
        axs[0, i].plot(x, y[order], "-")

        axs[0, i].set_xlabel("Experiment")
        axs[0, i].set_title(f"{name} NNs, All Data\nMSE:{mse:.3f}")

        data = training_data_in_rounds[name]
        x = np.arange(len(data))
        # print(data)
        mse = mean_squared_error(data["y_true"], data["y_pred"])
        order = np.argsort(data["y_true"].to_numpy())
        axs[1, i].plot(
            x, data["y_pred"].to_numpy()[order], ".", alpha=0.20, markersize=1
        )
        axs[1, i].plot(x, data["y_true"].to_numpy()[order], "-")
        axs[1, i].set_xlabel("Experiment")
        axs[1, i].set_title(f"{name} NNs, Train Data\nMSE:{mse:.3f}")

    fig.tight_layout()
    fig.savefig("summarize_nn_performance.png", dpi=400)


def count(df, threshold):
    depth_groups = df.groupby(by=["depth"])
    depth_counts = {}
    for jdx, df2 in depth_groups:
        n_total = len(df2)
        n_correct = (df2["fitness"] >= df2["growth_pred"]).sum()
        n_incorrect = (df2["fitness"] < df2["growth_pred"]).sum()

        frontier = df2[df2["frontier_type"] == "FRONTIER"]
        n_frontier = len(frontier)
        n_frontier_grows = (frontier["fitness"] >= threshold).sum()

        beyond = df2[df2["frontier_type"] == "BEYOND"]
        n_beyond = len(beyond)
        n_beyond_no_grows = (beyond["fitness"] < threshold).sum()

        depth_counts[jdx] = {
            "n_total": n_total,
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "n_frontier": n_frontier,
            "n_beyond": n_beyond,
            "proportion_frontier_grow": n_frontier_grows / n_frontier,
            "proportion_beyond_no_grows": n_beyond_no_grows / n_beyond,
            "proportion_explored": n_total / comb(20, jdx),
        }

    results = pd.DataFrame.from_dict(depth_counts, orient="index")
    results.index.name = "depth"
    return results


def main(folder):
    max_round_n = 6
    folders = [
        os.path.join(folder, i, "results_all.csv")
        for i in os.listdir(folder)
        if "Round" in i
    ]
    folders = sorted(folders, key=lambda x: (len(x), x))[:max_round_n]

    print(folders)

    round_data = []
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
        round_data_grouped = round_data.groupby(by=["type"])
        threshold = 0.25
        for idx, df in round_data_grouped:
            print()
            print(f"Group: {idx}")

            results = count(df, threshold)
            print(results)

            # print(depth_counts)
            grows = df[df["fitness"] >= threshold]
            grows = grows.sort_values(by=["depth", "fitness"], ascending=[False, False])
            print(grows)

            grows.to_csv(f"summarize_{idx}_grows_Round{i+1}.csv", index=False)
            results.to_csv(f"summarize_{idx}_results_Round{i+1}.csv")

        results_all = count(round_data, threshold)
        results_all.to_csv(f"summarize_ALL_results_Round{i+1}.csv")


if __name__ == "__main__":
    # f = "experiments/05-31-2021_7"
    # f = "experiments/05-31-2021_8"
    f = "experiments/05-31-2021_7 copy"
    # main(f)
    plot_model_performance(f)
