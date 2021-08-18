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


def plot_model_performance(experiment_folder):

    models_in_rounds = {}
    training_data_in_rounds = {}
    testing_data_in_rounds = {}
    for root, dirs, files in os.walk(experiment_folder):
        models = []
        for name in files:
            path = os.path.join(root, name)
            if "bag_model" in name:
                model = torch.load(path).cuda()
                models.append(model)
                # print(path)
            if "train_pred" in name:
                round_name = root.split("/")[-1]
                training_data_in_rounds[round_name] = pd.read_csv(path, index_col=None)

            if "results_all" in name:
                round_name = root.split("/")[-1]
                testing_data_in_rounds[round_name] = utils.normalize_ingredient_names(
                    pd.read_csv(path, index_col=None)
                )

        if models:
            round_name = root.split("/")[-2]
            models_in_rounds[round_name] = models

    round_names = sorted(list(models_in_rounds.keys()), key=lambda x: (len(x), x))
    # print(training_data_in_rounds.values())
    # data_path = "L1IO-L2IO-L3O All Rands SMU UA159 Processed-Aerobic.csv"
    # data_path = "data/SGO_data/SGO CH1 Processed-Aerobic.csv"

    # data = pd.read_csv(data_path, index_col=None)
    # X = data.iloc[:, :20].to_numpy()
    # y = data["growth"].to_numpy()

    # data[data["growth"] > 1.5].to_csv("Large_preds_SGO.csv")

    # round_predictions = []
    # for name in round_names:
    #     models = models_in_rounds.get(name, None)
    #     if models is not None:
    #         predictions, variances = net.eval_bagged(X, models)
    #         round_predictions.append((predictions, variances))

    n_rounds = len(models_in_rounds)
    fig, axs = plt.subplots(
        nrows=2, ncols=n_rounds, sharex=False, sharey="row", figsize=(30, 6)
    )

    for i, name in enumerate(round_names):
        # for i, (name, (pred, var)) in enumerate(zip(round_names, round_predictions)):
        # order = np.argsort(y)
        # x = np.arange(len(pred))
        # mse = mean_squared_error(y, pred)
        # axs[0, i].plot(x, pred[order], ".", alpha=0.10, markersize=1)
        # axs[0, i].plot(x, y[order], "-")

        # axs[0, i].set_xlabel("Experiment")
        # axs[0, i].set_title(f"{name} NNs, All Data\nMSE:{mse:.3f}")

        test_data = testing_data_in_rounds.get(name, None)
        if test_data is not None:
            x = np.arange(len(test_data))
            # print(data)
            mse = mean_squared_error(test_data["fitness"], test_data["growth_pred"])
            order = np.argsort(test_data["fitness"].to_numpy())
            axs[0, i].plot(
                x,
                test_data["growth_pred"].to_numpy()[order],
                ".",
                alpha=1,
                markersize=1,
            )
            axs[0, i].plot(x, test_data["fitness"].to_numpy()[order], "-")
            axs[0, i].set_xlabel("Experiment")
            axs[0, i].set_title(f"{name} NNs, Test Data\nMSE:{mse:.3f}")

        data = training_data_in_rounds.get(name, None)
        if data is not None:
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
    max_round_n = 12
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

        round_output = os.path.join(output_path, f"Round{i+1}")
        if not os.path.exists(round_output):
            os.makedirs(round_output)

        round_data_grouped = round_data.groupby(by=["type"])
        threshold = 0.25
        for group_type, df in round_data_grouped:
            print()
            print(f"Group: {group_type}")

            results = count(df, threshold)
            print(results)

            # print(depth_counts)
            grows = df[df["fitness"] >= threshold]
            grows = grows.sort_values(by=["depth", "fitness"], ascending=[False, False])
            print(grows)

            # grows.to_csv(
            #     os.path.join(round_output, f"{group_type}_grows.csv"),
            #     index=False,
            # )
            results.to_csv(
                os.path.join(round_output, f"summarize_{group_type}_results.csv")
            )

        results_all = count(round_data, threshold)
        results_all.to_csv(os.path.join(round_output, f"summarize_ALL_results.csv"))


def collect_data(folder):

    files = [f for f in os.listdir(folder) if "mapped_data" in f]
    dfs = [utils.process_mapped_data(os.path.join(folder, f))[0] for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.replace(
        {"Anaerobic Chamber @ 37 C": "anaerobic", "5% CO2 @ 37 C": "aerobic"}
    )
    df = df[df["environment"] == "aerobic"]
    df = df.drop(
        columns=["strain", "parent_plate", "initial_od", "final_od", "bad", "delta_od"]
    )
    df.to_csv(os.path.join(folder, "SGO CH1 Processed-Aerobic.csv"), index=False)


if __name__ == "__main__":
    # f = "experiments/05-31-2021_7"
    # f = "experiments/05-31-2021_8"
    # f = "experiments/05-31-2021_7 copy"
    f = "experiments/07-26-2021_10"
    # f = "experiments/07-26-2021_11"

    # main(f)
    plot_model_performance(f)

    # collect_data("data/SGO_data")
