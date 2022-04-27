import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath("."))
import global_vars
import plot
import utils


def print_info(args, ingredient_names):
    print("\n\n================== Generating Redo Plots ==================")
    print("Ingredients:\n  > ", ingredient_names)
    print("Mapped data path:\n  > ", args.new_mapped_path)
    print("Meta data path:\n  > ", args.new_meta_path)
    print("Previous results path:\n  > ", args.prev_results_path)
    print(
        "Plot output path:\n  > ",
        os.path.join(args.output_folder, "redo_compare_order_plot.png"),
    )
    print()


def main(args):
    if args.ingredients == "aa":
        ingredient_names = global_vars.AA_SHORT
    elif args.ingredients == "non-aa":
        ingredient_names = global_vars.BASE_NAMES
    elif args.ingredients == "all":
        ingredient_names = global_vars.AA_SHORT + global_vars.BASE_NAMES
    else:
        raise Exception(f"Error: invalid ingredients ({args.ingredients}).")

    print_info(args, ingredient_names)

    # Merge results (mapped data) with predictions (batch data)
    data, _, _ = utils.process_mapped_data(args.new_mapped_path, ingredient_names)
    batch_df = utils.normalize_ingredient_names(
        pd.read_csv(args.new_meta_path, index_col=None)
    )
    results = pd.merge(
        batch_df,
        data,
        how="left",
        left_on=ingredient_names,
        right_on=ingredient_names,
        sort=True,
    )

    if "is_redo" not in results.columns:
        raise Exception("Error: 'is_redo' columns is missing.")

    redo_results = results[results["is_redo"] == True]

    prev_results = utils.normalize_ingredient_names(
        pd.read_csv(args.prev_results_path, index_col=None)
    )

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    plot.plot_redos(args.output_folder, prev_results, redo_results, ingredient_names)


if __name__ == "__main__":
    # Read in command arguments
    parser = argparse.ArgumentParser(description="BacterAI Redo Plot Generator")

    parser.add_argument(
        "--new_mapped_path",
        type=str,
        required=True,
        help="The path to the new mapped data ('mapped_data' CSV from DeepPhenotyping)",
    )

    parser.add_argument(
        "--new_meta_path",
        type=str,
        required=True,
        help="The path to the new meta data ('batch_meta' CSV from BacterAI)",
    )

    parser.add_argument(
        "--prev_results_path",
        type=str,
        required=True,
        help="The path to the previous data to compare to ('results_all' CSV from BacterAI processing)",
    )

    parser.add_argument(
        "--ingredients",
        type=str,
        choices=("aa", "non-aa", "all"),
        required=True,
        help="The ingredients of the data.",
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="The folder to save the plot in",
    )

    args = parser.parse_args()

    main(args)
