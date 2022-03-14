import argparse
import collections
import os
from math import comb

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter

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


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['svg.fonttype'] = 'none'

def plot_main_fig(experiment_folder, all_test_data, all_train_data, fig_name, skip=1, show_train=True, max_n=None):
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
    if max_n:
        paths = paths[:max_n]
    all_results = []

    n_plots = len(paths)
    for round_idx in range(0, n_plots, skip):
        path = paths[round_idx]
        print(path)
        results = utils.normalize_ingredient_names(pd.read_csv(path, index_col=None))
        results = results.sort_values(by="growth_pred").reset_index(drop=True)
        if "is_redo" in results.columns:
            results = results[~results["is_redo"]]
        all_results.append((round_idx, results))

    height = 1.5 * n_plots + 1.5
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
        tot = 0
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
                tot += sum(list(counts.values()))
                for group_n, count in counts.items():
                    group_offset = group_n * (GROUP_WIDTH + SPACER_WIDTH)
                    for i in range(count):
                        x, y = _idx_to_pos(i + cumulative_count[group_n])
                        x += group_offset
                        axs[graph_idx, 0].plot(x, y, color=color, **opts)
                    cumulative_count[group_n] += count
        print(f"{tot=}")
        max_h = max(list(cumulative_count.values()) + [max_h])

        major_ticks = (
            np.arange(0, GROUP_WIDTH * 21, GROUP_WIDTH) + np.arange(21) * SPACER_WIDTH
        ) + 1.5

        axs[graph_idx, 0].set_xticks(major_ticks)
        axs[graph_idx, 0].set_xticklabels(np.arange(0, 21))
        axs[graph_idx, 0].set_yticklabels([])
        axs[graph_idx, 0].set_ylabel(f"Day {round_idx+1}", rotation=0, horizontalalignment='left')
        axs[graph_idx, 0].yaxis.set_label_coords(0.0, 0.8)

        axs[graph_idx, 0].spines["left"].set_visible(False)
        axs[graph_idx, 0].spines["right"].set_visible(False)
        axs[graph_idx, 0].spines["top"].set_visible(False)
        axs[graph_idx, 0].tick_params(axis="y", which="both", length=0)

        if graph_idx != len(all_results) - 1:
            # axs[graph_idx, 0].spines["bottom"].set_visible(False)
            axs[graph_idx, 0].tick_params(axis="x", which="both", length=0)
            axs[graph_idx, 0].axes.get_xaxis().set_visible(False)

        metric_style = dict(
            fontsize=10,
            verticalalignment="top",
            # bbox=dict(facecolor="white", alpha=0.5, linewidth=0),
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
                0, 1.05, f"Acc: {acc*100:.1f}%", **metric_style
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
                0, 1.05, f"Acc: {acc*100:.1f}%", **metric_style
            )

        if graph_idx == 0 and show_train:
            axs[graph_idx, 1].axis("off")
            
        # if graph_idx == 3:
            # axs[graph_idx, 1].set_ylabel("Fitness")
        

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

    
    if skip==2:
        # fig.text(0.84, 0.07, "Model Performance", ha="center")
        loc = (0.84, 0.06)
    else:
        # fig.text(0.84, 0.03, "Model Performance", ha="center")
        loc = (0.84, 0.02)

    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                label="Model prediction",
                color="dodgerblue",
                markersize=0,
                linewidth=2,
            ),
            Line2D(
                [0],
                [0],
                label="Experiment",
                color="k",
                marker=".",
                markersize=3,
                linewidth=0,
            ),
            
        ],
        loc="center",
        frameon=False,
        bbox_to_anchor=loc,
        ncol=1,
    )

    h = max_h // GROUP_WIDTH
    print(f"{max_h=}")
    for ax in axs[:, 0]:
        ax.set_ybound(-1, h + 1)

    legend_elements_attrs = [
        dict(color="k", label="Grow (Correct)"),
        dict(
            color="k",
            label="No Grow (Correct)",
            markerfacecolor="none",
            markeredgewidth=0.5,
        ),
        dict(
            color="r",
            label="Grow (Incorrect)",
        ),
        dict(
            color="r",
            label="No Grow (Incorrect)",
            markerfacecolor="none",
            markeredgewidth=0.5,
        ),
    ]
    legend_elements = [
        Line2D([0], [0], marker="o", markersize=3, linewidth=0, **attrs)
        for attrs in legend_elements_attrs
    ]

    # axs[-2].legend(handles=legend_elements)
    axs[-1, 0].set_xlabel("Amino Acids Removed")
    axs[-1, 0].legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5),
        frameon=False,
        ncol=1,
    )
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0.1, wspace=0, hspace=0.1)

    fig_path = os.path.join(experiment_folder, fig_name)
    plt.tight_layout()
    plt.savefig(fig_path + ".png", dpi=400)
    plt.savefig(fig_path + ".svg", dpi=400)


def _get_acc(a, b, threshold):
    a = a.copy()
    b = b.copy()
    a[a >= threshold] = 1
    a[a < threshold] = 0
    b[b >= threshold] = 1
    b[b < threshold] = 0

    acc = (a == b).sum() / a.shape[0]
    return acc


def plot_model_performance(experiment_folder, fig_name, max_n=None):

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
    if max_n:
        round_names = round_names[:max_n]

    n_rounds = len(round_names)
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

    fig_path = os.path.join(
        experiment_folder, f"summarize_nn_performance_{fig_name}.png"
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=400)
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

def make_growth_distribution_hist(bacterai_data, random_data):
    fig, axs = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(10,6)
    )

    width = 0.5
    n_bins = 20

    bins = np.arange(0, 1.01, 1/n_bins)
    rand, rand_bounds = np.histogram(random_data["fitness"], bins)
    bact, bact_bounds = np.histogram(bacterai_data["fitness"], bins)

    rand = rand/len(random_data)
    bact = bact/len(bacterai_data)
    # axs.hist(
    #     random_data["fitness"],
    #     bins=50,
    #     color="r",
    #     alpha=0.5,
    # )
    # axs.hist(
    #     bacterai_data["fitness"],
    #     bins=50,
    #     color="k",
    #     alpha=0.5,
    # )
        
    x = np.arange(n_bins)
    r1 = axs.bar(
        x+width/2, 
        rand,
        width,
        color="dodgerblue",
        # alpha=0.5,
    )
    r2 = axs.bar(
        x+1.5*width, 
        bact,
        width,
        color="k",
        # alpha=0.5,
    )
    axs.set_ylabel("Count Density")
    axs.set_xlabel("Fitness")
    axs.bar_label(r1, padding=2, fmt="%.2f", fontsize=7.5)
    axs.bar_label(r2, padding=2, fmt="%.2f", fontsize=7.5)
    # axs.set_title(f"Round {idx+1}")
    # axs.set_xticks(np.arange(n_bins+1))
    bin_labels = [f"{x:.2f}" for x in np.arange(0, 1.01, 1/n_bins)]
    print(bin_labels)
    axs.set_xticks(np.arange(0, n_bins+1, 1))
    axs.set_xticklabels(bin_labels)
    plt.xticks(rotation='vertical')
    # axs.set_yscale('log')

    # axs.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.legend(["Random", "BacterAI"])
    fig.tight_layout()
    fig.savefig("summarize_simulation_fitness_order_plot_combined.png", dpi=400)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BacterAI Figures")

    parser.add_argument(
        "path",
        type=str,
        help="The path to the experiment folder",
    )

    parser.add_argument(
        "-r",
        "--rounds",
        type=int,
        required=True,
        help="The number of rounds to plot out starting at 1",
    )

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=False,
        help="The name of the figure",
    )

    parser.add_argument(
        "-i",
        "--increment",
        type=int,
        required=False,
        choices=(1, 2),
        default=1,
        help="Plot every or every other",
    )

    args = parser.parse_args()

    name = args.name
    if not name:
        name = args.path.replace(" ", "-").replace("/", "_")

    all_test_data, all_train_data = plot_model_performance(
        args.path, name, max_n=args.rounds
    )

    plot_main_fig(
        args.path,
        all_test_data,
        all_train_data,
        name,
        skip=args.increment,
        max_n=args.rounds,
    )
    # plot_main_fig(folder, all_test_data, all_train_data, fig_name, skip=2)
    # plot_main_fig(folder, all_test_data, all_train_data, fig_name, skip=2, show_train=False)

    # # Second plot
    # data = utils.combined_round_data(folder, max_n=13)
    # print(data)
    # path = "Randoms (1) SGO CH1 17f3 mapped_data.csv"
    # # path = "BacterAI SGO CH1 (10R13) rule verify 12fb mapped_data.csv"
    # rand_data = utils.process_mapped_data(path)[0]
    # print(rand_data)
    # # rand_data = rand_data.sort_values(by="growth_pred").reset_index(drop=True)
    # # if "is_redo" in rand_data.columns:
    # #     rand_data = rand_data[~rand_data["is_redo"]]
    # make_growth_distribution_hist(data, rand_data)