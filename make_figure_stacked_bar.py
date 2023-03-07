import argparse
import collections
import os
from math import comb

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter, AutoMinorLocator

import numpy as np
import pandas as pd
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


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"


def plot_main_fig(
    experiment_folder,
    all_test_data,
    all_train_data,
    fig_name,
    n_ingredients,
    skip=1,
    show_train=True,
    max_n=None,
    show_rollout_proportion=False,
):
    N_GROUPS = n_ingredients

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


    bot_legend_pad = 1
    height = 1.3 * (len(all_results) + 1) + bot_legend_pad
    width = 10 if show_train else 8.57

    n_cols = 3 if show_train else 2
    gridspec = {"width_ratios": [5, 1, 1]} if show_train else {"width_ratios": [5, 1]}
    fig, axs = plt.subplots(
        nrows=len(all_results),
        ncols=n_cols,
        sharex=False,
        sharey=False,
        figsize=(width, height),
        gridspec_kw=gridspec,
    )

    max_h = 0
    for graph_idx, (round_idx, results) in enumerate(all_results):
        print()
        results = results.reset_index(drop=True)

        # reverse from # removed to # added
        results["n_media_ingredients"] = n_ingredients - results["depth"]
        
        tot = 0
        all_values = {}
        for kind in ["CORRECT", "INCORRECT"]:
            for t in ["FRONTIER", "BEYOND"]:
                print(kind, t)
                r = results[results["frontier_type"] == t]

                if (t == "FRONTIER" and kind == "CORRECT") or (
                    t == "BEYOND" and kind == "INCORRECT"
                ):
                    # ones that grow
                    n_media_ingredients = r[r["fitness"] >= threshold][
                        "n_media_ingredients"
                    ]

                elif (t == "FRONTIER" and kind == "INCORRECT") or (
                    t == "BEYOND" and kind == "CORRECT"
                ):
                    # ones that don't grow
                    n_media_ingredients = r[r["fitness"] < threshold][
                        "n_media_ingredients"
                    ]

                counts = {i: 0 for i in range(0, N_GROUPS + 1)}
                counts.update(collections.Counter(n_media_ingredients))
                print(counts)

                all_values[f"{kind} {t}"] = list(counts.values())

        bar_width = 0.85
        x_labels = range(0, N_GROUPS + 1)

        axs[graph_idx, 0].bar(
            x_labels,
            all_values["INCORRECT BEYOND"],
            bar_width,
            label="No Grow (Incorrectly Predicted)",
            color="r",
            alpha=0.25,
        )

        axs[graph_idx, 0].bar(
            x_labels,
            all_values["INCORRECT FRONTIER"],
            bar_width,
            bottom=all_values["INCORRECT BEYOND"],
            label="Grow (Incorrectly Predicted)",
            color="r",
            alpha=1.0,
        )

        axs[graph_idx, 0].bar(
            x_labels,
            all_values["CORRECT BEYOND"],
            bar_width,
            bottom=[
                sum(x)
                for x in zip(
                    all_values["INCORRECT FRONTIER"], all_values["INCORRECT BEYOND"]
                )
            ],
            label="No Grow (Correctly Predicted)",
            color="k",
            alpha=0.25,
        )

        axs[graph_idx, 0].bar(
            x_labels,
            all_values["CORRECT FRONTIER"],
            bar_width,
            bottom=[
                sum(x)
                for x in zip(
                    all_values["INCORRECT FRONTIER"],
                    all_values["INCORRECT BEYOND"],
                    all_values["CORRECT BEYOND"],
                )
            ],
            label="Grow (Correctly Predicted)",
            color="k",
            alpha=1.0,
        )

        stacked_heights = [
            sum(x)
            for x in zip(
                all_values["CORRECT FRONTIER"],
                all_values["CORRECT BEYOND"],
                all_values["INCORRECT FRONTIER"],
                all_values["INCORRECT BEYOND"],
            )
        ]
        max_h = max(stacked_heights + [max_h])

        # add minimal media designation
        mm_index = next(
            (i for i, x in enumerate(all_values["CORRECT FRONTIER"]) if x), None
        )
        mm_height = stacked_heights[mm_index]
        axs[graph_idx, 0].annotate(
            "*",
            xy=(mm_index, mm_height),
            xytext=(0, -3),  # xy offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

        if show_rollout_proportion:
            print(f"{stacked_heights=}")
            offset = bar_width / 2
            for media_size in range(n_ingredients + 1):
                m = results[results["n_media_ingredients"] == media_size]
                if not len(m):
                    continue
                rollout_results = m[m["type"] == "ROLLOUT_PROB"]
                proportion = len(rollout_results) / len(m)
                prop_height = stacked_heights[media_size] * proportion
                print(f"{media_size=}: {proportion=:.2f}, {prop_height=}")
                axs[graph_idx, 0].plot(
                    [media_size - offset, media_size - offset],
                    [0, prop_height - 0.75],
                    "b-",
                    markersize=0,
                    linewidth=1,
                )

        major_ticks = np.arange(0, N_GROUPS + 1, 5)
        axs[graph_idx, 0].set_aspect("auto")
        axs[graph_idx, 0].set_xlim(-0.99, N_GROUPS + 0.99)
        axs[graph_idx, 0].set_xticks(major_ticks)
        axs[graph_idx, 0].set_xticklabels([])
        minor_locator = AutoMinorLocator(5)
        axs[graph_idx, 0].xaxis.set_minor_locator(minor_locator)

        axs[graph_idx, 0].set_ylabel(f"Count")
        axs[graph_idx, 0].spines["right"].set_visible(False)
        axs[graph_idx, 0].spines["top"].set_visible(False)

        if graph_idx == len(all_results) - 1:
            axs[graph_idx, 0].set_xticklabels(np.arange(0, N_GROUPS + 1, 5))

        metric_style = dict(
            fontsize=10,
            verticalalignment="center",
        )

        train_data = all_train_data.get(round_idx, None)
        col = 1
        if train_data is not None and show_train and graph_idx + 1 < len(all_results):
            preds, y_true = train_data
            x_axis_points = np.arange(len(preds))

            mse = mean_squared_error(y_true, preds)
            acc = _get_acc(preds, y_true, threshold)
            order = np.argsort(preds)
            axs[graph_idx + 1, col].plot(
                x_axis_points,
                y_true[order],
                "k.",
                alpha=0.2,
                markersize=3,
                linewidth=0,
                markeredgewidth=0,
            )
            axs[graph_idx + 1, col].plot(
                x_axis_points,
                preds[order],
                color="dodgerblue",
            )

            axs[graph_idx + 1, col].text(0, 1.0, f"Acc: {acc*100:.1f}%", **metric_style)

            # grow threshold 25%
            axs[graph_idx + 1, col].plot(
                [0, len(y_true)],
                [0.25, 0.25],
                color="k",
                alpha=0.20,
            )

        if show_train:
            col += 1

        test_data = all_test_data.get(round_idx, None)
        if test_data is not None:
            preds, y_true = test_data
            x_axis_points = np.arange(len(preds))

            mse = mean_squared_error(y_true, preds)
            acc = _get_acc(preds, y_true, threshold)

            order = np.argsort(preds)

            # grow threshold 25%
            axs[graph_idx, col].plot(
                [0, len(y_true)],
                [0.25, 0.25],
                color="k",
                alpha=0.20,
            )

            axs[graph_idx, col].plot(
                x_axis_points,
                y_true[order],
                "k.",
                alpha=1,
                markersize=3,
                linewidth=0,
                markeredgewidth=0,
            )
            axs[graph_idx, col].plot(x_axis_points, preds[order], color="dodgerblue")

            axs[graph_idx, col].text(0, 1.0, f"Acc: {acc*100:.1f}%", **metric_style)

        if graph_idx == 0 and show_train:
            axs[graph_idx, 1].axis("off")

    for ax in axs[:, 1:].flatten():
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_ybound(-0.15, 1.15)
        ax.set_yticks([0, 0.25, 1])
        ax.set_yticklabels([0, 0.25, 1])

    if show_train:
        axs[-1, 1].set_xlabel(f"Train Set")
        axs[-1, 2].set_xlabel(f"Test Set")
    else:
        axs[-1, 1].set_xlabel(f"Test Set")

    for i, ax in enumerate(axs[:, 0]):
        ax.set_ybound(0, max_h)
        ax.text(
            0,
            max_h,
            f"Day {i*skip+1}",
            fontsize=12,
            verticalalignment="top",
            weight="bold",
        )

    fig.legend(
        handles=[
            Line2D([0], [0], label="Experiment", color="k", marker=".", markersize=3, linewidth=0),
            Line2D([0], [0], label="Model prediction", color="dodgerblue", markersize=0, linewidth=2),
            Line2D([0], [0], label="Grow/No Grow Threshold", color="k", markersize=0, linewidth=2, alpha=0.20),
        ],
        loc="upper right",
        frameon=False,
        bbox_to_anchor=(0.95, bot_legend_pad/height),
        ncol=1,
    )
    fig.legend(
        handles=[
            Patch(facecolor="k", label="Grow (Correctly Predicted)"),
            Patch(facecolor="k", label="No Grow (Correctly Predicted)", alpha=0.2),
            Patch(facecolor="r", label="Grow (Incorrectly Predicted)"),
            Patch(facecolor="r", label="No Grow (Incorrectly Predicted)", alpha=0.2),
        ],
        loc="upper left",
        bbox_to_anchor=(0.05, bot_legend_pad/height),
        frameon=False,
        ncol=1,
    )

    x_label = "Ingredients in Media" if n_ingredients > 20 else "Amino Acids in Media"
    axs[-1, 0].set_xlabel(x_label)
    
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.tight_layout(rect=[0, bot_legend_pad/height, 1, 1])

    fig_path = os.path.join(experiment_folder, fig_name)
    for file_ext in (".png", ".svg"):
        plt.savefig(f"{fig_path}{file_ext}", dpi=400)

    return paths


def _get_acc(a, b, threshold):
    a = a.copy()
    b = b.copy()
    a[a >= threshold] = 1
    a[a < threshold] = 0
    b[b >= threshold] = 1
    b[b < threshold] = 0

    acc = (a == b).sum() / a.shape[0]
    return acc


def plot_model_performance(experiment_folder, fig_name, n_ingredients, max_n=None, transfer_learning=False):
    threshold = 0.25
    models_in_rounds = {}
    training_data_in_rounds = {}
    testing_data_in_rounds = {}
    for root, _, files in os.walk(experiment_folder):
        models = []
        for name in files:
            path = os.path.join(root, name)
            if "bad_runs" in path:
                continue
            if "bag_model" in name:
                model = torch.load(path, map_location=torch.device(net.DEVICE))
                models.append(model)
            if "train_pred.csv" in name:
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

            if "results_all.csv" in name:
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
            print()
            print(np.argwhere(np.isnan(data_2)).size, data_2.size)
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

        training_data = training_data_in_rounds.get(name, None)
        if training_data is not None:
            models = models_in_rounds.get(name, None)
            if transfer_learning and i == 0:
                _n_ingredients = 19
            else:
                _n_ingredients = n_ingredients
            preds, _ = net.eval_bagged(
                training_data.to_numpy()[:, :_n_ingredients], models
            )

            data_1 = preds
            data_2 = training_data["y_true"].to_numpy()
            all_train_data[i] = (data_1, data_2)

            x_axis_points = np.arange(len(training_data))

            mse = mean_squared_error(training_data["y_true"], preds)
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

    parser.add_argument(
        "-t",
        "--show_train",
        action="store_true",
        default=False,
        help="Include train plots.",
    )

    parser.add_argument(
        "-rp",
        "--show_rollout_proportion",
        action="store_true",
        default=False,
        help="Show the proportion of 'rollout' guesses alongside each bar.",
    )


    parser.add_argument(
        "-num",
        "--num_ingredients",
        type=int,
        required=False,
        default=20,
        help="The number of experiment ingredients",
    )

    args = parser.parse_args()

    name = args.name
    if not name:
        name = args.path.replace(" ", "-").replace("/", "_")

    is_transfer_learning_scheme = args.num_ingredients > 20
    all_test_data, all_train_data = plot_model_performance(
        args.path, name, n_ingredients=args.num_ingredients, max_n=args.rounds, transfer_learning=is_transfer_learning_scheme
    )

    out_paths = plot_main_fig(
        args.path,
        all_test_data,
        all_train_data,
        name,
        n_ingredients=args.num_ingredients,
        skip=args.increment,
        max_n=args.rounds,
        show_train=args.show_train,
        show_rollout_proportion=args.show_rollout_proportion,
    )

    print(f'Saved to: {out_paths}')
