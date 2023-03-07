import collections
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constants import *


def plot_redos(folder, prev_results, redo_results, ingredients):
    """Plot the rescreen results of the previous round against the previous round's results.

    Parameters
    ----------
    folder : str
        The folder of the current round.
    prev_results : pd.DataFrame
        The previous round's processed results.
    redo_results : pd.DataFrame
        The current round's processed results, includes only the rescreens.
    """

    merged_results = pd.merge(
        prev_results,
        redo_results,
        how="right",
        left_on=ingredients,
        right_on=ingredients,
        sort=True,
        suffixes=["_prev", "_redo"],
    )

    fitness_prev = merged_results["fitness_prev"].to_numpy()
    fitness_redo = merged_results["fitness_redo"].to_numpy()

    if np.isnan(fitness_prev).all():
        print("\n\n====== WARNING: No overlapping data from previous round. ======\n\n")

    order = np.argsort(fitness_prev)
    fig = plt.figure()
    x = np.arange(len(order))
    plt.plot(
        x, fitness_prev[order], "-", c="black", markersize=2, label="Previous Round"
    )
    plt.plot(
        x, fitness_redo[order], ".", c=COLORS["REDO"], markersize=2, label="Rescreen"
    )

    plt.xlabel("Assay N")
    plt.ylabel("Fitness")
    plt.title("Rescreen Fitness Comparison")
    plt.suptitle(f"Experiment: {folder}")
    plt.legend()

    plt.tight_layout()
    save_path = os.path.join(folder, "redo_compare_order_plot.png")
    plt.savefig(save_path, dpi=300)


def plot_results(folder, results, threshold):
    """Plot a summary of the current round's results for both the FRONTIER type and
    BEYOND_FRONTIER type:
        1) Shows a order plot of the  actual fitnesses vs. the model's
        predicted fitnesses.
        2) Shows a histogram of the depth (# ingredients removed) counts for each
        search policy (RANDOM and ROLLOUT)


    Parameters
    ----------
    folder : str
        The folder of the current round.
    results : pd.DataFrame
        The processed results of the round.
    threshold : float
        The no grow/grow threshold.
    """

    results = results.sort_values(by="growth_pred").reset_index(drop=True)
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=False,
        sharey=False,
        figsize=(12, 8),
        gridspec_kw={"width_ratios": [1.25, 2]},
    )

    frontier_grouped = results.groupby(by=["frontier_type"], as_index=False)
    for row_idx, (frontier_type, results) in enumerate(
        reversed(list(frontier_grouped))
    ):
        results = results.reset_index(drop=True)
        sim_type_grouped = results.groupby(by=["type"], as_index=False)
        present_groups = []
        for group_name, data in sim_type_grouped:
            present_groups.append(group_name)
            axs[row_idx, 0].plot(
                data.index,
                data["fitness"],
                ".",
                color=COLORS[group_name],
                markersize=3,
                alpha=0.75,
            )
        axs[row_idx, 0].plot(results.index, results["growth_pred"], "-", color="black")
        axs[row_idx, 0].set_xlabel("Assay N")
        axs[row_idx, 0].set_ylabel("Fitness")
        axs[row_idx, 0].legend(
            [f"{g.title()}" for g in present_groups] + ["Model Prediction"]
        )
        axs[row_idx, 0].set_title(f"Experiment Results - {frontier_type.title()}")

        width = 0.25
        legend_labels = []
        for i, (group_name, data) in enumerate(sim_type_grouped):
            color = COLORS[group_name]
            grows = data[data["fitness"] >= threshold]
            no_grows = data[data["fitness"] < threshold]
            data_g = collections.Counter(list(grows["depth"]))
            data_ng = collections.Counter(list(no_grows["depth"]))
            bottom = [data_g[k] if k in data_g else 0 for k in data_ng.keys()]
            if len(no_grows) > 0:
                legend_labels.append(f"{group_name.title()} - No Grow")
                axs[row_idx, 1].bar(
                    np.array(list(data_ng.keys())) + width * i,
                    data_ng.values(),
                    bottom=bottom,
                    width=width,
                    color=color,
                    edgecolor=color,
                    hatch="////",
                    alpha=0.25,
                    linewidth=0,
                )
            if len(grows) > 0:
                legend_labels.append(f"{group_name.title()} - Grow")
                axs[row_idx, 1].bar(
                    np.array(list(data_g.keys())) + width * i,
                    data_g.values(),
                    width=width,
                    color=color,
                )

        axs[row_idx, 1].set_title(f"Depth - {frontier_type.title()}")
        axs[row_idx, 1].set_xlabel("Depth (n_removed)")
        axs[row_idx, 1].set_ylabel("Count")
        axs[row_idx, 1].set_xticks(np.arange(0, 21) + 2 * width / 2)
        axs[row_idx, 1].set_xticklabels(np.arange(0, 21))
        axs[row_idx, 1].legend(legend_labels)

    plt.suptitle(f"Experiment: {folder}")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "results_graphic.png"), dpi=400)
