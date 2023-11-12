import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["svg.fonttype"] = "none"

def main(data, random_data, experiment_folder, n_bins, bacterai_data=None):
    # Plot the growth front
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    width = 0.5

    bins = np.arange(0, 1.01, 1 / n_bins)
    rand, _ = np.histogram(random_data["fitness"], bins)
    bact, _ = np.histogram(data["fitness"], bins)

    rand = rand / len(random_data)
    bact = bact / len(data)

    x = np.arange(n_bins)
    r1 = axs.bar(
        x + width / 2,
        rand,
        width,
        color="dodgerblue",
    )
    r2 = axs.bar(
        x + 1.5 * width,
        bact,
        width,
        color="k",
    )
    axs.set_ylabel("Count Density")
    axs.set_xlabel("Fitness")
    axs.bar_label(r1, padding=2, fmt="%.2f", fontsize=7.5, rotation=90)
    axs.bar_label(r2, padding=2, fmt="%.2f", fontsize=7.5, rotation=90)

    bin_labels = [f"{x:.2f}" for x in np.arange(0, 1.01, 1 / n_bins)]
    print(bin_labels)
    axs.set_xticks(np.arange(0, n_bins + 1, 1))
    axs.set_xticklabels(bin_labels)
    plt.xticks(rotation=45)
    axs.set_yscale('log')
    for tick in axs.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")

    plt.legend(["Random", "BacterAI"])
    fig.tight_layout()

    fig_path = os.path.join(experiment_folder, "growth_front_hist")
    for file_ext in (".png", ".svg"):
        plt.savefig(f"{fig_path}{file_ext}", dpi=400)

    # Plot the growth distribution
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    width = 0.5

    bins = np.arange(0, 1.01, 1 / n_bins)
    rand, _ = np.histogram(random_data["fitness"], bins)
    bact, _ = np.histogram(data["fitness"], bins)

    rand = rand / len(random_data)
    bact
    # Plot the growth front
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    width = 0.5

    bins = np.arange(0, 1.01, 1 / n_bins)
    rand, _ = np.histogram(random_data["fitness"], bins)
    bact, _ = np.histogram(data["fitness"], bins)

    rand = rand / len(random_data)
    bact = bact / len(data)

    x = np.arange(n_bins)
    r1 = axs.bar(
        x + width / 2,
        rand,
        width,
        color="dodgerblue",
    )
    r2 = axs.bar(
        x + 1.5 * width,
        bact,
        width,
        color="k",
    )
    axs.set_ylabel("Count Density")
    axs.set_xlabel("Fitness")
    axs.bar_label(r1, padding=2, fmt="%.2f", fontsize=7.5, rotation=90)
    axs.bar_label(r2, padding=2, fmt="%.2f", fontsize=7.5, rotation=90)

    bin_labels = [f"{x:.2f}" for x in np.arange(0, 1.01, 1 / n_bins)]
    print(bin_labels)
    axs.set_xticks(np.arange(0, n_bins + 1, 1))
    axs.set_xticklabels(bin_labels)
    plt.xticks(rotation=45)
    axs.set_yscale('log')
    for tick in axs.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")

    plt.legend(["Random", "BacterAI"])
    fig.tight_layout()

    fig_path = os.path.join(experiment_folder, "growth_front_hist")
    for file_ext in (".png", ".svg"):
        plt.savefig(f"{fig_path}{file_ext}", dpi=400)

    # Plot the growth distribution
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    width = 0.5

    bins = np.arange(0, 1.01, 1 / n_bins)
    rand, _ = np.histogram(random_data["fitness"], bins)
    bact, _ = np.histogram(data["fitness"], bins)

    rand
    bacterai_data, random_data, experiment_folder, n_bins
):
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    width = 0.5

    bins = np.arange(0, 1.01, 1 / n_bins)
    rand, _ = np.histogram(random_data["fitness"], bins)
    bact, _ = np.histogram(bacterai_data["fitness"], bins)

    rand = rand / len(random_data)
    bact = bact / len(bacterai_data)

    x = np.arange(n_bins)
    r1 = axs.bar(
        x + width / 2,
        rand,
        width,
        color="dodgerblue",
    )
    r2 = axs.bar(
        x + 1.5 * width,
        bact,
        width,
        color="k",
    )
    axs.set_ylabel("Count Density")
    axs.set_xlabel("Fitness")
    axs.bar_label(r1, padding=2, fmt="%.2f", fontsize=7.5, rotation=90)
    axs.bar_label(r2, padding=2, fmt="%.2f", fontsize=7.5, rotation=90)

    bin_labels = [f"{x:.2f}" for x in np.arange(0, 1.01, 1 / n_bins)]
    print(bin_labels)
    axs.set_xticks(np.arange(0, n_bins + 1, 1))
    axs.set_xticklabels(bin_labels)
    plt.xticks(rotation=45)
    axs.set_yscale('log')
    for tick in axs.xaxis.get_majorticklabels():
        tick.set_horizontalalignment("right")

    plt.legend(["Random", "BacterAI"])
    fig.tight_layout()

    fig_path = os.path.join(experiment_folder, "growth_distribution_hist")
    for file_ext in (".png", ".svg"):
        plt.savefig(f"{fig_path}{file_ext}", dpi=400)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BacterAI Growth Front Figure")

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
        "-num",
        "--num_ingredients",
        type=int,
        required=False,
        default=20,
        help="The number of experiment ingredients",
    )

    args = parser.parse_args()

    # Second plot
    data = utils.combined_round_data(args.path, max_n=args.rounds)

    # originally Randoms (1) SGO CH1 17f3 mapped_data
    random_path = "published_data/SGO random/experiment_data.csv"
    rand_data = pd.read_csv(random_path, index_col=None)
    
    if "is_redo" in rand_data.columns:
        rand_data = rand_data[~rand_data["is_redo"]]

    main(data, rand_data, args.path, args.num_ingredients)
