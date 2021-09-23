import collections
import os
from math import comb

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats.kde import gaussian_kde
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import torch

import net
import utils

COLORS = {
    "ROLLOUT": "violet",
    "ROLLOUT_PROB": "violet",
    "RANDOM": "dodgerblue",
    "GREEDY": "orangered",
    "REDO": "limegreen",
    "GROW": "limegreen",
    "NOGROW": "tomato",
}


def plot_frontier_summary_alt(experiment_folder, all_test_data, all_train_data, skip=1, show_train=True):
    GROUP_WIDTH = 4
    SPACER_WIDTH = 1.5
    N_GROUPS = 21

    def _idx_to_pos(idx):
        x = idx % GROUP_WIDTH
        y = idx // GROUP_WIDTH
        return (x, y)

    threshold = 0.25
    paths = []
    for root, dirs, files in os.walk(experiment_folder):
        models = []
        for name in files:
            path = os.path.join(root, name)
            if "bad_runs" in path:
                continue
            if "results_all" in name:
                paths.append(path)

    paths = sorted(paths, key=lambda x: (len(x), x), reverse=False)
    all_results = []

    for round_idx in range(0, len(paths), skip):
        path = paths[round_idx]
        print(path)
        results = utils.normalize_ingredient_names(pd.read_csv(path, index_col=None))
        results = results.sort_values(by="growth_pred").reset_index(drop=True)
        if "is_redo" in results.columns:
            results = results[~results["is_redo"]]
        all_results.append((round_idx, results))

    height = 20/skip
    width = 11 if show_train else 10

    if show_train:
        fig, axs = plt.subplots(
            nrows=len(all_results),
            ncols=3,
            sharex=False,
            sharey=False,
            figsize=(width, height),
            gridspec_kw={"width_ratios": [5, 1, 1]},
        )
    else:
        fig, axs = plt.subplots(
            nrows=len(all_results),
            ncols=2,
            sharex=False,
            sharey=False,
            figsize=(width, height),
            gridspec_kw={"width_ratios": [6, 1]},
        )

    point_opts = [
        {"markersize": 4, "marker": "."},
        {
            "markerfacecolor": "none",
            "markeredgewidth": 0.75,
            "markersize": 4.5,
            "marker": ".",
        },
    ]
    max_h = 0
    for graph_idx, (round_idx, results) in enumerate(all_results):
        print()
        print(f"{graph_idx=}")
        results = results.reset_index(drop=True)
        cumulative_count = {i: 0 for i in range(0, 21)}
        for kind in ["CORRECT", "INCORRECT"]:
            for t, opts in zip(["FRONTIER", "BEYOND"], point_opts):
                print(kind, t)
                color = "k" if kind == "CORRECT" else "r"
                r = results[results["frontier_type"] == t]

                if (t == "FRONTIER" and kind == "CORRECT") or (
                    t == "BEYOND" and kind == "INCORRECT"
                ):
                    depths = r[r["fitness"] >= threshold]["depth"].to_list()
                elif (t == "FRONTIER" and kind == "INCORRECT") or (
                    t == "BEYOND" and kind == "CORRECT"
                ):
                    depths = r[r["fitness"] < threshold]["depth"].to_list()

                counts = {i: 0 for i in range(0, 21)}
                counts.update(collections.Counter(depths))
                print(counts)
                for group_n, count in counts.items():
                    group_offset = group_n * (GROUP_WIDTH + SPACER_WIDTH)
                    for i in range(count):
                        x, y = _idx_to_pos(i + cumulative_count[group_n])
                        x += group_offset
                        axs[graph_idx, 0].plot(x, y, color=color, **opts)
                    cumulative_count[group_n] += count

        max_h = max(list(cumulative_count.values()) + [max_h])

        # major_ticks = (
        #     np.linspace(, SPACER_WIDTH + (SPACER_WIDTH + GROUP_WIDTH) * 21, 21)
        #     - GROUP_WIDTH / 2
        # )
        # major_ticks = (
        #     np.arange(
        #         SPACER_WIDTH / 2, GROUP_WIDTH * 23 - SPACER_WIDTH / 2, GROUP_WIDTH
        #     )
        #     + np.arange(23) * SPACER_WIDTH
        # )

        major_ticks = (
            np.arange(0, GROUP_WIDTH * 21, GROUP_WIDTH) + np.arange(21) * SPACER_WIDTH
        ) + 1.5

        # axs[graph_idx, 0].set_aspect("auto")
        # axs[graph_idx, 0].set_xlim([-1, 22])
        # axs[graph_idx, 0].set_xbound(lower=-1, upper=22)
        # axs[graph_idx, 0].margins(x=1)
        axs[graph_idx, 0].set_xticks(major_ticks)
        # axs[graph_idx, 0].set_xticklabels([""] + list(range(21)) + [""])
        axs[graph_idx, 0].set_xticklabels(np.arange(0, 21))
        axs[graph_idx, 0].set_yticklabels([])
        axs[graph_idx, 0].set_ylabel(f"Round {round_idx+1}", rotation=45)

        axs[graph_idx, 0].spines["left"].set_visible(False)
        axs[graph_idx, 0].spines["right"].set_visible(False)
        axs[graph_idx, 0].spines["top"].set_visible(False)
        axs[graph_idx, 0].tick_params(axis="y", which="both", length=0)

        if graph_idx != len(all_results) - 1:
            # axs[graph_idx, 0].spines["bottom"].set_visible(False)
            axs[graph_idx, 0].tick_params(axis="x", which="both", length=0)
            axs[graph_idx, 0].axes.get_xaxis().set_visible(False)

        metric_style = dict(
            fontsize=8,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(facecolor="white", alpha=0.5, linewidth=0),
        )
        

        train_data = all_train_data.get(round_idx, None)
        col = 1
        if train_data is not None and show_train and graph_idx + 1 < len(all_results):
            preds, y_true = train_data
            x_axis_points = np.arange(len(preds))

            mse = mean_squared_error(y_true, preds)
            acc = _get_acc(preds, y_true, threshold)
            order = np.argsort(preds)
            axs[graph_idx+1, col].plot(
                x_axis_points, y_true[order], "k.", alpha=0.2, markersize=3, linewidth=0, markeredgewidth=0
            )
            axs[graph_idx+1, col].plot(
                x_axis_points,
                preds[order],
                color="dodgerblue",
            )

            axs[graph_idx+1, col].text(
                0, 1.05, f"Acc: {acc:.3f}\nMSE: {mse:.3f}", **metric_style
            )
        
        if show_train:
            col += 1
            

        test_data = all_test_data.get(round_idx, None)
        if test_data is not None:
            preds, y_true = test_data
            x_axis_points = np.arange(len(preds))
            # print(data)
            mse = mean_squared_error(y_true, preds)
            acc = _get_acc(preds, y_true, threshold)

            order = np.argsort(preds)
            axs[graph_idx, col].plot(
                x_axis_points, y_true[order], "k.", alpha=1, markersize=3, linewidth=0, markeredgewidth=0
            )
            axs[graph_idx, col].plot(x_axis_points, preds[order], color="dodgerblue")
            axs[graph_idx, col].text(
                0, 1.05, f"Acc: {acc:.3f}\nMSE: {mse:.3f}", **metric_style
            )

        if graph_idx == 0 and show_train:
            axs[graph_idx, 1].axis("off")
            
        if graph_idx == 3:
            axs[graph_idx, 1].set_ylabel("Fitness")
        

    for ax in axs[:, 1:].flatten():
        # ax.set_aspect("equal")
        # ax.axes.get_xaxis().set_visible(False)
        ax.set_xticks([]) 
        ax.set_xticklabels([]) 
        ax.set_ybound(-0.15, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([0, 1])


    if show_train:
        axs[-1, 1].set_xlabel(f"Train Set")
        axs[-1, 2].set_xlabel(f"Test Set")
        # axs[-1, 1].axes.get_xaxis().set_visible(True)
        # axs[-1, 2].axes.get_xaxis().set_visible(True)
    else:
        axs[-1, 1].set_xlabel(f"Test Set")
        # axs[-1, 1].axes.get_xaxis().set_visible(True)

    

    fig.text(0.84, 0.07, "Model Performance", ha="center")

    axs[-1, 1].legend(
        handles=[
            Line2D(
                [0],
                [0],
                label="data",
                color="k",
                marker=".",
                markersize=3,
                linewidth=0,
            ),
            Line2D(
                [0],
                [0],
                label="model",
                color="dodgerblue",
                markersize=0,
                linewidth=2,
            ),
        ],
        loc="center",
        bbox_to_anchor=(0.5, -0.5),
        ncol=2,
    )

    h = max_h // GROUP_WIDTH
    print(f"{max_h=}")
    for ax in axs[:, 0]:
        ax.set_ybound(-1, h + 1)

    legend_elements_attrs = [
        dict(color="k", label="Correct - Frontier"),
        dict(
            color="k",
            label="Correct - Beyond",
            markerfacecolor="none",
            markeredgewidth=0.5,
        ),
        dict(
            color="r",
            label="Incorrect - Frontier",
        ),
        dict(
            color="r",
            label="Incorrect - Beyond",
            markerfacecolor="none",
            markeredgewidth=0.5,
        ),
    ]
    legend_elements = [
        Line2D([0], [0], marker="o", markersize=3, linewidth=0, **attrs)
        for attrs in legend_elements_attrs
    ]

    # axs[-2].legend(handles=legend_elements)
    axs[-1, 0].set_xlabel("Number of AAs Removed")
    axs[-1, 0].legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5),
        ncol=4,
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.1, wspace=0, hspace=0.1)

    # plt.suptitle("S. gordonii")
    # plt.suptitle("S. gordonii Frontier Dist")
    plt.tight_layout()
    plt.savefig(f"summarize_hero_fig_SGO_10_full.png", dpi=400)
    plt.savefig(f"summarize_hero_fig_SGO_10_full.svg", dpi=400)
    # plt.savefig(f"summarize_hero_fig_SSA_13.png", dpi=400)


def _get_acc(a, b, threshold):
    a = a.copy()
    b = b.copy()
    a[a >= threshold] = 1
    a[a < threshold] = 0
    b[b >= threshold] = 1
    b[b < threshold] = 0

    acc = (a == b).sum() / a.shape[0]
    return acc


def plot_model_performance(experiment_folder):

    threshold = 0.25
    models_in_rounds = {}
    training_data_in_rounds = {}
    testing_data_in_rounds = {}
    for root, dirs, files in os.walk(experiment_folder):
        models = []
        for name in files:
            path = os.path.join(root, name)
            if "bad_runs" in path:
                continue
            if "bag_model" in name:
                model = torch.load(path).cuda()
                models.append(model)
                # print(path)
            if "train_pred" in name:
                # This round's train_pred.csv has the data from all previous_rounds
                # and is used to train this round's models, therefore, we have to
                # assign the training data to our prev round for these plots

                # Training set is current set and all prev rounds
                # Test set is the #N-1's model's batch preds

                round_name = root.split("/")[-1]
                round_n = int(round_name.split("Round")[-1])
                results = pd.read_csv(path, index_col=None)
                if "is_redo" in results.columns:
                    results = results[~results["is_redo"]]
                training_data_in_rounds[f"Round{round_n-1}"] = results

            if "results_all" in name:
                round_name = root.split("/")[-1]
                results = utils.normalize_ingredient_names(
                    pd.read_csv(path, index_col=None)
                )
                if "is_redo" in results.columns:
                    results = results[~results["is_redo"]]
                testing_data_in_rounds[round_name] = results

        if models:
            round_name = root.split("/")[-2]
            models_in_rounds[round_name] = models

    round_names = sorted(list(models_in_rounds.keys()), key=lambda x: (len(x), x))
    n_rounds = len(models_in_rounds)
    fig, axs = plt.subplots(
        nrows=2, ncols=n_rounds, sharex=False, sharey="row", figsize=(15, 6)
    )

    all_test_data = {}
    all_train_data = {}
    for i, name in enumerate(round_names):
        test_data = testing_data_in_rounds.get(name, None)
        if test_data is not None:
            data_1 = test_data["growth_pred"].to_numpy()
            data_2 = test_data["fitness"].to_numpy()
            all_test_data[i] = (data_1, data_2)
            x_axis_points = np.arange(len(test_data))
            # print(data)
            mse = mean_squared_error(test_data["fitness"], test_data["growth_pred"])
            acc = _get_acc(data_1, data_2, threshold)

            order = np.argsort(data_1)
            axs[0, i].plot(
                x_axis_points,
                data_2[order],
                ".",
                alpha=1,
                markersize=1,
            )
            axs[0, i].plot(x_axis_points, data_1[order], "-")
            axs[0, i].set_xlabel("Experiment")
            axs[0, i].set_title(f"{name} NNs, Test\nMSE:{mse:.3f}\nAcc:{acc:.3f}")

        data = training_data_in_rounds.get(name, None)
        if data is not None:
            models = models_in_rounds.get(name, None)
            preds, variances = net.eval_bagged(data.to_numpy()[:, :20], models)

            data_1 = preds
            data_2 = data["y_true"].to_numpy()
            all_train_data[i] = (data_1, data_2)

            x_axis_points = np.arange(len(data))

            mse = mean_squared_error(data["y_true"], preds)
            acc = _get_acc(data_1, data_2, threshold)

            order = np.argsort(data_1)
            axs[1, i].plot(x_axis_points, data_2[order], ".", alpha=0.20, markersize=1)
            axs[1, i].plot(x_axis_points, data_1[order], "-")
            axs[1, i].set_xlabel("Experiment")
            axs[1, i].set_title(f"{name} NNs, Train\nMSE:{mse:.3f}\nAcc:{acc:.3f}")

    fig.tight_layout()
    fig.savefig("summarize_nn_performance.png", dpi=400)
    return all_test_data, all_train_data


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
            results = count(df, threshold)
            grows = df[df["fitness"] >= threshold]
            grows = grows.sort_values(by=["depth", "fitness"], ascending=[False, False])

            results.to_csv(
                os.path.join(round_output, f"summarize_{group_type}_results.csv")
            )

        results_all = count(round_data, threshold)
        results_all.to_csv(os.path.join(round_output, f"summarize_ALL_results.csv"))


def combined_round_data(experiment_folder):
    paths = []
    for root, dirs, files in os.walk(experiment_folder):
        models = []
        for name in files:
            path = os.path.join(root, name)
            if "bad_runs" in path:
                continue
            if "results_all" in name:
                paths.append(path)

    paths = sorted(paths, key=lambda x: (len(x), x), reverse=False)
    all_results = []
    for idx, path in enumerate(paths):
        print(path)
        results = utils.normalize_ingredient_names(pd.read_csv(path, index_col=None))
        results = results.sort_values(by="growth_pred").reset_index(drop=True)
        if "is_redo" in results.columns:
            results = results[~results["is_redo"]]
        results["round"] = idx + 1
        all_results.append(results)

    all_results = pd.concat(all_results, ignore_index=True)
    return all_results


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
    # folder = "experiments/05-31-2021_7"
    # folder = "experiments/05-31-2021_8"
    # folder = "experiments/05-31-2021_7 copy"
    folder = "experiments/07-26-2021_10"
    # folder = "experiments/07-26-2021_11"
    # folder = "experiments/08-20-2021_13"

    all_test_data, all_train_data = plot_model_performance(folder)
    # plot_ridgeline_policy_summary(folder)
    # plot_ridgeline_frontier_summary(folder)
    plot_frontier_summary_alt(folder, all_test_data, all_train_data, skip=1)
    # plot_frontier_summary_alt(folder, all_test_data, all_train_data, skip=2)
    # plot_frontier_summary_alt(folder, all_test_data, all_train_data, skip=2, show_train=False)
    # plot_frontier_jitter(folder)

    # collect_data("data/SGO_data")
