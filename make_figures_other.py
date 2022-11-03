import collections
import os
from math import comb

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
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


def ridgeline_hist_ax(ax, data, overlap=0, fill=True, labels=None, n_points=150):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError("overlap must be in [0 1]")
    xx = np.linspace(0, 20, n_points)
    curves = []
    ys = []
    for i, (data_g, data_ng) in enumerate(data):
        y = 250 * i * (1.0 - overlap)
        ys.append(y)

        bottom = []
        for n in range(20):
            if n in data_g.tolist():
                bottom.append((data_g == n).sum() + y)
            else:
                bottom.append(y)
        ax.hist(
            data_g,
            bins=20,
            range=(0, 20),
            zorder=len(data) - i + 1,
            color="limegreen",
            # edgecolor="black",
            # linewidth=1.2,
            # alpha=0.50,
            bottom=y,
            histtype="stepfilled",
        )
        ax.hist(
            data_ng,
            bins=20,
            range=(0, 20),
            zorder=len(data) - i + 1,
            color="orange",
            # edgecolor="black",
            # linewidth=1.2,
            # alpha=0.50,
            bottom=bottom,
            histtype="stepfilled",
        )

        # ax.hist(
        #     d,
        #     bins=20,
        #     range=(0, 20),
        #     zorder=len(data) - i + 1,
        #     facecolor="None",
        #     edgecolor="black",
        #     linewidth=1.2,
        #     bottom=y,
        #     histtype="stepfilled",
        # )
    if labels:
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)


def ridgeline_ax(ax, data, overlap=0, fill=True, labels=None, n_points=150):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError("overlap must be in [0 1]")
    xx = np.linspace(0, 20, n_points)
    curves = []
    ys = []
    for i, d in enumerate(data):
        alpha = 0.80
        if len(d) <= 1:
            d = [0, 20]
            alpha = 0
        pdf = gaussian_kde(d)
        y = i * (1.0 - overlap)
        ys.append(y)
        curve = pdf(xx)
        if fill:
            ax.fill_between(
                xx,
                np.ones(n_points) * y,
                curve + y,
                zorder=len(data) - i + 1,
                color=fill,
                alpha=alpha,
            )
        if alpha != 0:
            ax.plot(xx, curve + y, c="k", zorder=len(data) - i + 1)
    if labels:
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)


def ridgeline(data, overlap=0, fill=True, labels=None, n_points=150):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError("overlap must be in [0 1]")
    xx = np.linspace(0, 20, n_points)

    color_add = np.linspace(0, 0.5, len(data))
    curves = []
    ys = []

    is_blue = False
    if "dodgerblue" in fill:
        is_blue = True

    for i, d in enumerate(data):

        if len(d) == 0:
            continue
        pdf = gaussian_kde(d)
        y = i * (1.0 - overlap)
        ys.append(y)
        curve = pdf(xx)

        if fill:
            if is_blue:
                fill = [
                    min(0.118 + color_add[i], 1),
                    min(0.565 + color_add[i], 1),
                    min(1 + color_add[i], 1),
                    1,
                ]
            else:
                fill = [
                    min(0.933 + color_add[i], 1),
                    min(0.51 + color_add[i], 1),
                    min(0.933 + color_add[i], 1),
                    1,
                ]

            plt.fill_between(
                xx,
                np.ones(n_points) * y,
                curve + y,
                zorder=len(data) - i + 1,
                color=fill,
                alpha=0.90,
            )
        plt.plot(xx, curve + y, color="k", linewidth=1, zorder=len(data) - i + 1)
    if labels:
        plt.yticks(ys, labels)


def ridgeline_hist(data, overlap=0, fill="black", labels=None, n_points=150):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError("overlap must be in [0 1]")
    for i, d in enumerate(data):
        y = 250 * i * (1.0 - overlap)
        plt.hist(
            d,
            bins=20,
            range=(0, 20),
            zorder=len(data) - i + 1,
            color=fill,
            edgecolor="black",
            linewidth=1.2,
            # alpha=0.50,
            bottom=y,
            histtype="stepfilled",
        )
        # plt.plot(xx, curve + y, c="k", zorder=len(data) - i + 1)
    # if labels:
    #     plt.yticks(ys, labels)


def plot_ridgeline_policy_summary(experiment_folder):
    paths = []
    for root, dirs, files in os.walk(experiment_folder):
        models = []
        for name in files:
            path = os.path.join(root, name)
            if "bad_runs" in path:
                continue
            if "results_all" in name:
                paths.append(path)
    all_results = []
    for path in sorted(paths, key=lambda x: (len(x), x), reverse=True):
        print(path)
        results = utils.normalize_ingredient_names(pd.read_csv(path, index_col=None))
        results = results.sort_values(by="growth_pred").reset_index(drop=True)
        if "is_redo" in results.columns:
            results = results[~results["is_redo"]]
        all_results.append(results)

    ridgeline_data = {"ROLLOUT_PROB": [], "RANDOM": []}
    for ridge_idx, results in enumerate(all_results):
        results = results.reset_index(drop=True)
        count = []
        if "ROLLOUT_PROB" in results["type"].to_list():
            r = results[results["type"] == "ROLLOUT_PROB"]
            count = r["depth"].to_numpy()
        ridgeline_data["ROLLOUT_PROB"].append(count)

        count = []
        if "RANDOM" in results["type"].to_list():
            r = results[results["type"] == "RANDOM"]
            count = r["depth"].to_numpy()
        ridgeline_data["RANDOM"].append(count)

    labels = reversed([f"Round {r}" for r in range(1, len(all_results) + 1)])

    for group_type in ["RANDOM", "ROLLOUT_PROB"]:
        # ridgeline_hist(group_counts, overlap=0.80, fill=COLORS[group_type])
        group_counts = ridgeline_data[group_type]
        print(group_type)
        if group_type != "RANDOM":
            labels = None
        ridgeline(group_counts, overlap=0.85, fill=COLORS[group_type], labels=labels)

    plt.title("S. mutans Removed Dist")
    # plt.title("S. gordonii Removed Dist")
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.tick_params(axis="y", which="both", length=0)
    plt.xticks(range(0, 21))

    legend_elements = [
        Patch(
            facecolor=COLORS["RANDOM"],
            edgecolor="k",
            label="Random Policy",
            alpha=0.80,
        ),
        Patch(
            facecolor=COLORS["ROLLOUT"],
            edgecolor="k",
            label="Rollout Policy",
            alpha=0.80,
        ),
    ]

    # Put a legend below current axis
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
    )
    plt.xlabel("Number of AAs Removed")
    # plt.legend(handles=legend_elements)

    # plt.savefig(f"summarize_ridgeline_hist.png", dpi=400)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    # plt.savefig(f"summarize_ridgeline_SGO.png", dpi=400)
    plt.savefig(f"summarize_ridgeline_SMU.png", dpi=400)


def plot_ridgeline_frontier_summary(experiment_folder):
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
    all_results_frontier = []
    all_results_beyond = []
    for path in sorted(paths, key=lambda x: (len(x), x), reverse=True):
        print(path)
        results = utils.normalize_ingredient_names(pd.read_csv(path, index_col=None))
        results = results.sort_values(by="growth_pred").reset_index(drop=True)
        if "is_redo" in results.columns:
            results = results[~results["is_redo"]]
        all_results_frontier.append(results[results["frontier_type"] == "FRONTIER"])
        all_results_beyond.append(results[results["frontier_type"] == "BEYOND"])

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=False,
        sharey=False,
        figsize=(12, 8),
    )
    titles = ["Frontier", "Beyond"]
    for col_idx, all_results in enumerate([all_results_frontier, all_results_beyond]):
        print(f"{col_idx=}")
        ridgeline_data = {"GROW": [], "NOGROW": []}
        data = []
        for ridge_idx, results in enumerate(all_results):
            results = results.reset_index(drop=True)
            grows = results[results["fitness"] >= threshold]["depth"].to_numpy()
            no_grows = results[results["fitness"] < threshold]["depth"].to_numpy()
            ridgeline_data["GROW"].append(grows)
            ridgeline_data["NOGROW"].append(no_grows)
            data.append((grows, no_grows))
        labels = reversed([f"Round {r}" for r in range(1, len(all_results) + 1)])
        # for group_type in ["GROW", "NOGROW"]:
        #     group_counts = ridgeline_data[group_type]
        #     if group_type != "GROW":
        #         labels = None
        #     ridgeline_hist_ax(
        #         axs[col_idx],
        #         group_counts,
        #         overlap=0.80,
        #         fill=COLORS[group_type],
        #         labels=labels,
        #     )

        ridgeline_hist_ax(
            axs[col_idx],
            data,
            overlap=0.70,
            # fill=COLORS[group_type],
            labels=labels,
        )

        # ridgeline_ax(
        #     axs[col_idx],
        #     group_counts,
        #     overlap=0.80,
        #     fill=COLORS[group_type],
        #     labels=labels,
        # )
        axs[col_idx].set_title(titles[col_idx])
        axs[col_idx].set_xticks(range(0, 21))

    legend_elements = [
        Patch(facecolor="limegreen", edgecolor="k", label="Grow", alpha=1),
        Patch(facecolor="orange", edgecolor="k", label="No grow", alpha=1),
    ]

    for ax in axs:
        ax.tick_params(axis="y", which="both", length=0)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.legend(handles=legend_elements)

    plt.suptitle("S. mutans Frontier Dist")
    # plt.suptitle("S. gordonii Frontier Dist")

    # plt.savefig(f"summarize_ridgeline_fontier_hist_SGO.png", dpi=400)
    plt.savefig(f"summarize_ridgeline_fontier_hist_SMU.png", dpi=400)


def plot_frontier_summary_alt(experiment_folder):
    GROUP_WIDTH = 6
    SPACER_WIDTH = 2
    N_GROUPS = 20

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
    all_results = []
    for path in sorted(paths, key=lambda x: (len(x), x), reverse=False):
        print(path)
        results = utils.normalize_ingredient_names(pd.read_csv(path, index_col=None))
        results = results.sort_values(by="growth_pred").reset_index(drop=True)
        if "is_redo" in results.columns:
            results = results[~results["is_redo"]]
        all_results.append(results)
        # all_results_frontier.append(results[results["frontier_type"] == "FRONTIER"])
        # all_results_beyond.append(results[results["frontier_type"] == "BEYOND"])

    # all_results = [all_results[-1]]
    # print(all_results)
    # all_results_beyond = all_results_beyond[-1]
    # all_results = [r for i, r in enumerate(all_results) if i % 2 != 0]
    fig, axs = plt.subplots(
        nrows=len(all_results),
        ncols=1,
        sharex=True,
        sharey=True,
        figsize=(8, 8),
    )

    # xs = GROUP_WIDTH * N_GROUPS + SPACER_WIDTH * (N_GROUPS - 1)
    types = ["FRONTIER", "BEYOND"]
    point_opts = [
        {"markersize": 1.5, "marker": "."},
        {
            "markerfacecolor": "none",
            "markeredgewidth": 0.5,
            "markersize": 2.5,
            "marker": ".",
        },
    ]
    for graph_idx, results in enumerate(all_results):
        results = results.reset_index(drop=True)
        cumulative_count = {i: 0 for i in range(0, N_GROUPS + 1)}
        print("correct")
        for t, opts in zip(types, point_opts):
            r = results[results["frontier_type"] == t]

            if t == "FRONTIER":
                correct = r[r["fitness"] >= threshold]["depth"].to_list()
            else:
                correct = r[r["fitness"] < threshold]["depth"].to_list()

            correct_counts = {i: 0 for i in range(0, N_GROUPS + 1)}
            correct_counts.update(collections.Counter(correct))

            for group_n, count in correct_counts.items():
                group_offset = (group_n + 1) * GROUP_WIDTH + SPACER_WIDTH * group_n
                for i in range(count):
                    x, y = _idx_to_pos(i + cumulative_count[group_n])
                    x += group_offset
                    axs[graph_idx].plot(
                        x,
                        y,
                        color=(255 / 255, 77 / 255, 64 / 255, 1)
                        if t == "FRONTIER"
                        else (156 / 255, 0, 21 / 255, 1),
                        **opts,
                    )
                cumulative_count[group_n] += count

        print("incorrect")
        for t, opts in zip(types, point_opts):
            r = results[results["frontier_type"] == t]

            if t == "FRONTIER":
                incorrect = results[results["fitness"] < threshold]["depth"].to_list()
            else:
                incorrect = results[results["fitness"] >= threshold]["depth"].to_list()

            incorrect_counts = {i: 0 for i in range(0, N_GROUPS + 1)}
            incorrect_counts.update(collections.Counter(incorrect))
            for group_n, count in incorrect_counts.items():
                group_offset = (group_n + 1) * GROUP_WIDTH + SPACER_WIDTH * group_n
                for i in range(count):
                    x, y = _idx_to_pos(i + cumulative_count[group_n])
                    x += group_offset
                    axs[graph_idx].plot(
                        x,
                        y,
                        color="k" if t == "FRONTIER" else (0.5, 0.5, 0.5, 1),
                        **opts,
                    )
                cumulative_count[group_n] += count
        major_ticks = (
            np.arange(0, GROUP_WIDTH * (N_GROUPS + 1), GROUP_WIDTH)
            + np.arange(N_GROUPS + 1) * SPACER_WIDTH
        ) + 0.5

        axs[graph_idx].set_xticks(major_ticks)
        axs[graph_idx].set_xticklabels(np.arange(0, N_GROUPS + 1))
        axs[graph_idx].set_yticklabels([])
        axs[graph_idx].set_ylabel(f"Round {graph_idx+1}")
        axs[graph_idx].spines["left"].set_visible(False)
        axs[graph_idx].spines["right"].set_visible(False)
        axs[graph_idx].spines["top"].set_visible(False)
        axs[graph_idx].tick_params(axis="y", which="both", length=0)

        if graph_idx != len(all_results) - 1:
            axs[graph_idx].spines["bottom"].set_visible(False)
            axs[graph_idx].tick_params(axis="x", which="both", length=0)

        # axs[graph_idx].set_xticks(minor_ticks, minor=True)
        # axs[graph_idx].set_yticks(major_ticks)
        # axs[graph_idx].set_yticks(minor_ticks, minor=True)
        # axs[graph_idx].set_ylim(-1, N_GROUPS)

        # And a corresponding grid
        # axs[graph_idx].grid(which="both", alpha=0.2)
        # ridgeline_data["GROW"].append(grows)
        # ridgeline_data["NOGROW"].append(no_grows)
        # data.append((grows, no_grows))
    # labels = reversed([f"Round {r}" for r in range(1, len(all_results) + 1)])

    # axs[graph_idx].set_title(titles[col_idx])
    # axs[graph_idx].set_xticks(range(0, 21))

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="r",
            label="Correct - Frontier",
            markersize=3,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="r",
            label="Correct - Beyond",
            markersize=3,
            linewidth=0,
            markerfacecolor="none",
            markeredgewidth=0.5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            label="Incorrect - Frontier",
            markersize=3,
            linewidth=0,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="k",
            label="Incorrect - Beyond",
            markersize=3,
            linewidth=0,
            markerfacecolor="none",
            markeredgewidth=0.5,
        ),
    ]

    # for ax in axs:
    # ax.tick_params(axis="y", which="both", length=0)

    # axs[-2].legend(handles=legend_elements)
    plt.xlabel("Number of AAs Removed")
    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.5),
        ncol=5,
    )
    plt.subplots_adjust(bottom=0.1, hspace=0.001)

    plt.suptitle("S. mutans Frontier Dist")
    # plt.suptitle("S. gordonii Frontier Dist")
    # plt.tight_layout()
    # plt.savefig(f"summarize_ridgeline_fontier_alt_SGO.png", dpi=400)
    plt.savefig(f"summarize_ridgeline_fontier_alt_SMU.png", dpi=400)


def plot_frontier_jitter(experiment_folder):
    threshold = 0.25
    data = utils.combined_round_data(experiment_folder)
    data["grows"] = False
    data.loc[
        data["fitness"] >= threshold,
        "grows",
    ] = True

    data["correct"] = False
    data.loc[
        (data["frontier_type"] == "FRONTIER") & data["grows"],
        "correct",
    ] = True
    data.loc[(data["frontier_type"] == "BEYOND") & ~data["grows"], "correct"] = True

    # print(data)
    # print(data.columns)
    sns.set_style("whitegrid")

    ax = sns.violinplot(
        x="round",
        y="fitness",
        hue="correct",
        data=data,
        inner=None,
        linewidth=0,
        palette=[(0, 0, 0)],
        dodge=True,
        cut=0
        # split=True,
    )
    plt.setp(ax.collections, alpha=0.1)

    # data_1 = data[data["correct"]]
    data_1 = data[data["frontier_type"] == "FRONTIER"]
    styles = dict(
        dodge=True,
        size=2,
        jitter=0.3,
        alpha=0.5,
        edgecolor="none",
    )
    sns.stripplot(
        x="round",
        y="fitness",
        hue="correct",
        marker="o",
        # palette=sns.color_palette("husl")[:2],
        palette=("r", "k"),
        data=data_1,
        **styles,
    )

    # sns.boxplot(
    #     x="round", y="fitness", hue="correct", data=data_1, width=0.6, palette="vlag"
    # )

    # data_2 = data[~data["correct"]]
    data_2 = data[data["frontier_type"] == "BEYOND"]
    sns.stripplot(
        x="round",
        y="fitness",
        hue="correct",
        marker="X",
        palette=("r", "k"),
        data=data_2,
        **styles,
    )
    # sns.boxplot(
    #     x="round", y="fitness", hue="correct", data=data_2, width=0.6, palette="vlag"
    # )
    # plt.show()
    plt.savefig(f"summarize_ridgeline_fontier_jitter_SMU.png", dpi=400)

    # plt.figure()
    # ax = sns.violinplot(
    #     x="round",
    #     y="fitness",
    #     hue="correct",
    #     data=data,
    #     inner="points",
    #     # dodge=True,
    #     # size=1,
    #     # size="depth", sizes=(2, 5)
    # )

    # plt.savefig(f"summarize_ridgeline_fontier_violin_SMU.png", dpi=400)


def plot_model_performance(experiment_folder, n_ingredients=20):
    def _get_acc(a, b):
        a = a.copy()
        b = b.copy()
        a[a >= threshold] = 1
        a[a < threshold] = 0
        b[b >= threshold] = 1
        b[b < threshold] = 0

        acc = (a == b).sum() / a.shape[0]
        return acc

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
                model = torch.load(path, map_location=torch.device(net.DEVICE))
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

    for i, name in enumerate(round_names):
        test_data = testing_data_in_rounds.get(name, None)
        if test_data is not None:
            data_1 = test_data["growth_pred"].to_numpy()
            data_2 = test_data["fitness"].to_numpy()

            x_axis_points = np.arange(len(test_data))
            # print(data)
            mse = mean_squared_error(test_data["fitness"], test_data["growth_pred"])

            acc = _get_acc(data_1, data_2)

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
            preds, variances = net.eval_bagged(
                data.to_numpy()[:, :n_ingredients], models
            )

            data_1 = preds
            data_2 = data["y_true"].to_numpy()

            x_axis_points = np.arange(len(data))

            mse = mean_squared_error(data["y_true"], preds)
            acc = _get_acc(data_1, data_2)
            order = np.argsort(data_1)
            axs[1, i].plot(x_axis_points, data_2[order], ".", alpha=0.20, markersize=1)
            axs[1, i].plot(x_axis_points, data_1[order], "-")
            axs[1, i].set_xlabel("Experiment")
            axs[1, i].set_title(f"{name} NNs, Train\nMSE:{mse:.3f}\nAcc:{acc:.3f}")

    fig.tight_layout()
    fig.savefig("summarize_nn_performance.png", dpi=400)


def count(df, threshold, n_ingredients=20):
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
            "proportion_explored": n_total / comb(n_ingredients, jdx),
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


if __name__ == "__main__":
    folder = "experiments/05-31-2021_7"
    # folder = "experiments/05-31-2021_8"
    # folder = "experiments/05-31-2021_7 copy"
    # folder = "experiments/07-26-2021_10"
    # folder = "experiments/07-26-2021_11"

    plot_model_performance(folder)
    # plot_ridgeline_policy_summary(folder)
    # plot_ridgeline_frontier_summary(folder)
    # plot_frontier_summary_alt(folder)
    # plot_frontier_jitter(folder)

    # collect_data("data/SGO_data")
