import argparse
import csv
import datetime
import json
import os
import shutil

import numpy as np
import pandas as pd

from global_vars import *
from models import GPRModel, NeuralNetModel, ModelType
from plot import plot_redos, plot_results
from scripts.size_n_to_m_conversion import fill_new_ingredients
from sim import SimType, SimDirection, perform_simulations
import utils

os.environ["MKL_DEBUG_CPU_TYPE"] = "5"  # Use IntelMKL - does this work?


def export_to_dp_batch(parent_path, batch, ingredient_names, date, nickname=None):
    """ Export the BacterAI batch to a Deep Phenotyping-compatible file. """

    if len(batch) == 0:
        print("Empty Batch: No files generated.")
        return

    n_ingredients = len(ingredient_names)
    batch = batch.rename(
        columns={a: b for a, b in zip(range(n_ingredients), ingredient_names)}
    )
    batch = batch.sort_values(by=["growth_pred", "var"], ascending=[False, True])
    batch.to_csv(os.path.join(parent_path, f"batch_meta_{date}.csv"), index=None)

    # DeepPhenotyping compatible list
    batch = batch.drop(columns=batch.columns[n_ingredients:])
    nickname = f"_{nickname}" if nickname != None else ""

    with open(os.path.join(parent_path, f"batch_dp{nickname}_{date}.csv"), "w") as f:
        writer = csv.writer(f, delimiter=",")
        for _, row_data in batch.iterrows():
            row_data = row_data[row_data == 0]
            removed_ingredients = list(row_data.index.to_numpy())
            writer.writerow(removed_ingredients)


def make_batch(
    model,
    media,
    new_round_n,
    batch_size,
    sim_types,
    rollout_trajectories,
    threshold,
    timeout=60 * 15,
    unique=True,
    direction=SimDirection.DOWN,
    go_beyond_frontier=True,
    used_experiments=None,
    redo_experiments=None,
):
    """ Make a new BacterAI batch; the main function that calls the simulation loops. """

    n_types = len(sim_types)
    n_exps = batch_size // n_types
    batch_set = used_experiments
    sub_batches = []
    all_metrics = {}
    for idx, sim_type in enumerate(sim_types):
        if idx == n_types - 1:
            n_exps = batch_size - sum([len(x) for x in sub_batches])
        print(idx, sim_type, batch_size, n_exps, sum([len(x) for x in sub_batches]))
        batch, batch_set, metrics = perform_simulations(
            model,
            media,
            n_exps,
            threshold,
            sim_type,
            direction,
            new_round_n,
            unique=unique,
            timeout=timeout,
            batch_set=batch_set,
            n_rollout_trajectories=rollout_trajectories,
            go_beyond_frontier=go_beyond_frontier,
        )
        sub_batches.append(batch)
        all_metrics[sim_type.name] = metrics

    batch = pd.concat([redo_experiments] + sub_batches, ignore_index=True)
    return batch, batch_set, all_metrics


def process_results(
    folder,
    prev_folder,
    new_folder,
    new_round_n,
    ingredient_names,
    threshold,
    n_redos=0,
    redo_threshold=[0, 1],
    redo_prev_round=False,
    plot_only=False,
    transfer_padding_needed=False,
):
    """Process the results of the previous round, generate plots, and
    return batch information to be used when generating the new round's
    batch.

    Parameters
    ----------
    folder: str
        The folder path of the current round.
    prev_folder: str
        The folder path of the previous round.
    new_folder: str
        The folder path of the new round.
    new_round_n: int
        The number of the new round.
    ingredient_names: int
        The ingredients used.
    threshold: float
        The grow/no grow threshold used to determine when to terminate
        a rollout simulation.
    n_redos: int, optional
        The number of the experiments to redo from previous round.
    redo_threshold: [float, float], optional
        The lower and upper thresholds used to determine which experiments to use when
        sampling for redos. Lower threshold is inclusive, upper is exclusive.
    redo_prev_round: bool, optional
        Run the previous round's batch again,
        by default False
    plot_only: bool, optional
        Only save plot, don't save/export any other files.
    transfer_padding_needed: bool, optional
        Whether or not to pad datasets with ones for transfer learning (data
        dir method)

    Returns
    -------
    (np.ndarray, np.ndarray, set(tuple), pd.DataFrame)
        The new training set inputs, the  new training set labels, the
        experiments used in all previous experimments, the set of
        experiments to redo.
    """

    folder_contents = os.listdir(folder)
    n_ingredients = len(ingredient_names)

    # Get paths for the necessary files in the current folder (new round # - 1)
    mapped_path = None
    batch_path = None
    dataset_path = None
    for i in folder_contents:
        if "mapped_data" in i:
            mapped_path = os.path.join(folder, i)
        elif "batch_meta" in i and "results" not in i:
            batch_path = os.path.join(folder, i)
        elif "train_pred" in i and "orig" not in i:
            dataset_path = os.path.join(folder, i)

    new_dataset_path = os.path.join(new_folder, "train_pred.csv")

    # Merge results (mapped data) with predictions (batch data)
    data, _, _ = utils.process_mapped_data(mapped_path, ingredient_names)
    batch_df = utils.normalize_ingredient_names(pd.read_csv(batch_path, index_col=None))
    results = pd.merge(
        batch_df,
        data,
        how="left",
        left_on=ingredient_names,
        right_on=ingredient_names,
        sort=True,
    )

    if transfer_padding_needed:
        # Expand dimensions of non-AA data by the length of AA data and pad with ones
        batch_df = fill_new_ingredients(
            batch_df,
            original_size=len(BASE_NAMES),
            fill_column_names=AA_SHORT,
            fill_on_right=False,
        )
        results = fill_new_ingredients(
            results,
            original_size=len(BASE_NAMES),
            fill_column_names=AA_SHORT,
            fill_on_right=False,
        )
        n_ingredients = len(AA_SHORT) + len(BASE_NAMES)

    # Plot the rescreen experiments if available
    if "is_redo" in results.columns:
        redo_results = results[results["is_redo"] == True]
        results = results[results["is_redo"] != True]

        if prev_folder != None:
            prev_result_path = os.path.join(prev_folder, "results_all.csv")
            prev_results = utils.normalize_ingredient_names(
                pd.read_csv(prev_result_path, index_col=None)
            )
            plot_redos(folder, prev_results, redo_results, ingredient_names)

    # Process results
    results.iloc[:, :n_ingredients] = results.iloc[:, :n_ingredients].astype(int)
    results["depth"] = n_ingredients - results.iloc[:, :n_ingredients].sum(axis=1)
    results = results.sort_values(["depth", "fitness"], ascending=False)
    if "frontier_type" not in results.columns:
        results["frontier_type"] = "FRONTIER"
        print("Added 'frontier_type' column")
    if not plot_only:
        results.to_csv(os.path.join(folder, "results_all.csv"), index=None)

    # Generate results figure for current round
    plot_results(folder, results, threshold)
    if plot_only:
        quit()

    # Keep experiments where all replicate wells are not marked "bad"
    results_bad = results.loc[results["bad"] == 1, :].drop(columns="bad")
    results = results.loc[results["bad"] == 0, :].drop(columns="bad")

    # Assemble new training data set, either from scratch or appending to previous Round's set
    cols = list(results.columns[:n_ingredients]) + ["fitness", "growth_pred", "var"]
    cols_new = list(range(n_ingredients)) + ["y_true", "y_pred", "y_pred_var"]
    if dataset_path == None:
        new_dataset = pd.DataFrame(
            results.loc[:, cols].to_numpy(),
            columns=cols_new,
        )
    else:
        dataset = utils.normalize_ingredient_names(
            pd.read_csv(dataset_path, index_col=None)
        )

        data_batch = results.loc[:, cols]
        data_batch.iloc[:, :n_ingredients] = data_batch.iloc[:, :n_ingredients].astype(
            int
        )
        dataset.columns = data_batch.columns = cols_new
        new_dataset = pd.concat([dataset, data_batch], ignore_index=True)

    # Used experiments are the new dataset (old dataset plus "good" experiments from current round)
    used_experiments = set(map(tuple, new_dataset.to_numpy()[:, :n_ingredients]))
    new_dataset.iloc[:, :n_ingredients] = new_dataset.iloc[:, :n_ingredients].astype(
        int
    )
    new_dataset.to_csv(new_dataset_path, index=None)
    X_train = new_dataset.iloc[:, :n_ingredients].to_numpy()
    y_train = new_dataset.loc[:, "y_true"].to_numpy()

    # Assemble redo experiments, starting with bad experiments
    results_grow_only = results[results["fitness"] >= threshold]

    # Obtain the experiments from the current round to rescreen in the new round
    if redo_prev_round:
        redo_experiments = batch_df.loc[batch_df["type"] != "REDO", :]
        if "is_redo" in redo_experiments.columns:
            redo_experiments = redo_experiments[redo_experiments["is_redo"] == False]
        redo_experiments["is_redo"] = True
        redo_experiments["round"] = new_round_n - 1
        redo_experiments.columns = list(range(n_ingredients)) + list(
            redo_experiments.columns[n_ingredients:]
        )
        print(f"Redoing {len(redo_experiments)} experiments from previous round.")
    elif n_redos > 0:
        redo_experiments = results[results["type"] != "REDO"]
        if "is_redo" in redo_experiments.columns:
            redo_experiments = redo_experiments[redo_experiments["is_redo"] == False]

        if isinstance(redo_threshold, list) or isinstance(redo_threshold, tuple):
            if len(redo_threshold) != 2:
                raise Exception("Length of redo_threshold must be 2.")

            thresholded_indexes = redo_experiments[
                (redo_experiments["fitness"] >= redo_threshold[0])
                & (redo_experiments["fitness"] < redo_threshold[1])
            ].index

            n_needed = max(n_redos - len(thresholded_indexes), 0)
            if n_needed > 0:
                # Need more to fill out redos
                print(f"Picking {n_needed} more random experiments to redo.")
                unused_indexes = redo_experiments.index.difference(thresholded_indexes)
                additional_indexes = np.random.choice(
                    unused_indexes,
                    size=min(n_needed, len(unused_indexes)),
                    replace=False,
                )
                thresholded_indexes = thresholded_indexes.union(additional_indexes)

            redo_experiments = redo_experiments.loc[thresholded_indexes, :]

        redo_experiments = redo_experiments.sample(
            min(n_redos, len(redo_experiments)), replace=False
        )

        redo_experiments = redo_experiments.loc[:, batch_df.columns]
        redo_experiments["is_redo"] = True
        redo_experiments["round"] = new_round_n - 1
        redo_experiments.columns = list(range(n_ingredients)) + list(
            redo_experiments.columns[n_ingredients:]
        )
        print(f"Redoing {len(redo_experiments)} experiments from previous round.")

    # Save and output successful results
    results_grow_only.to_csv(os.path.join(folder, "results_grow_only.csv"), index=False)

    # Print some metrics/results
    top_10 = results_grow_only.iloc[:10, :]
    print("Media Results (Top 10):")
    for idx, (_, row) in enumerate(top_10.iterrows()):
        print(f"{idx+1:2}. Depth: {row['depth']:2}, Fitness: {row['fitness']:.3f}")
        for l in row[:n_ingredients][row[:n_ingredients] == 1].index:
            print(f"\t{l}")

    print(f"Total unique experiments: {len(used_experiments)}")
    print(
        f"Total redo experiments chosen: {len(redo_experiments)} ({len(results_bad)} 'bad' repeats)"
    )

    return X_train, y_train, used_experiments, redo_experiments


def main(args):
    # Process command line args
    NEW_ROUND_N = args.round

    # Load the configuration file
    with open(args.path) as f:
        config = json.load(f)

    GROW_THRESHOLD = config["grow_threshold"]
    EXPT_FOLDER = config["experiment_path"]
    NICKNAME = f"{config['nickname']}R{NEW_ROUND_N}"
    BATCH_SIZE = config["batch_size"]
    TIMEOUT_MIN = config["timeout_min"]
    N_ROLLOUTS = config["n_rollouts"]
    MODEL_TYPE = ModelType(config["model_type"])
    DIRECTION = SimDirection(config["direction"])
    SIMULATION_TYPE = [SimType(x) for x in config["simulation_types"]]
    BEYOND_FRONTIER = config["beyond_frontier"]
    USE_UNIQUE = config["use_unique"]
    TRANSFER_MODEL_FOLDER = config.get("transfer_model_folder", None)
    N_REDOS = config.get("redo_size", None)
    REDO_THRESHOLD = config.get("redo_threshold", None)
    AAS_ONLY = config.get("aas_only", True)
    TRANSFER_DATA_DIR = config.get("transfer_model_dir", None)

    if AAS_ONLY:
        INGREDIENTS = AA_SHORT
        TEMPEST_INGREDIENTS = AA_NAMES_TEMPEST
    elif not AAS_ONLY and NEW_ROUND_N > 2:
        INGREDIENTS = AA_SHORT + BASE_NAMES
        TEMPEST_INGREDIENTS = AA_NAMES_TEMPEST + BASE_NAMES_TEMPEST
    elif not AAS_ONLY and NEW_ROUND_N <= 2:
        INGREDIENTS = BASE_NAMES
        TEMPEST_INGREDIENTS = BASE_NAMES_TEMPEST

    n_ingredients = len(INGREDIENTS)

    tl_transition_round = False
    if TRANSFER_DATA_DIR is not None and not AAS_ONLY:
        print(
            f"Using transfer learning (data dir method) from '{TRANSFER_DATA_DIR}' directory."
        )
        tl_transition_round = NEW_ROUND_N == 2

    if TRANSFER_MODEL_FOLDER is not None:
        print(
            f"Using transfer learning (model method) from '{TRANSFER_MODEL_FOLDER}' model."
        )
        transfer_model = NeuralNetModel.load_trained_models(TRANSFER_MODEL_FOLDER)

    date = datetime.datetime.now().isoformat().replace(":", ".")
    prev_round_folder = (
        os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N-2}") if NEW_ROUND_N > 2 else None
    )
    current_round_folder = os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N-1}")
    new_round_folder = os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N}")
    if not os.path.exists(new_round_folder):
        os.makedirs(new_round_folder)
    if NEW_ROUND_N > 1:
        # Continue the experiment (for all rounds except the first)
        redo_entire_round = False if N_REDOS is not None else True
        X_train, y_train, used_experiments, redo_experiments = process_results(
            current_round_folder,
            prev_round_folder,
            new_round_folder,
            NEW_ROUND_N,
            INGREDIENTS,
            GROW_THRESHOLD,
            n_redos=N_REDOS,
            redo_threshold=REDO_THRESHOLD,
            redo_prev_round=redo_entire_round,
            plot_only=args.plot_only,
            transfer_padding_needed=tl_transition_round,
        )
    elif TRANSFER_MODEL_FOLDER:
        # Skip any initial random training if using a pre-trained model
        used_experiments = None
        redo_experiments = None
    else:
        # Cold start the AI (only for the 1st round AND if not using pre-trained model):
        #   1) Use random data to train the model.
        #   2) Compute a new batch using the RANDOM policy
        #   3) The random data doesn't get used for training in future rounds

        # Create random binary inputs of shape (1000, n_ingredients) and assign random fitness [0, 1]
        n_examples = 1000
        X_train = np.random.rand(n_examples, n_ingredients)
        X_train[X_train >= 0.5] = 1
        X_train[X_train < 0.5] = 0
        y_train = np.random.rand(n_examples, 1).flatten()

        # Force at least 25% of the fitnesses to 0
        index_choices = set(range(len(y_train)))
        y_train_zeros = np.random.choice(
            list(index_choices), size=int(n_examples * 0.25), replace=False
        )
        y_train[y_train_zeros] = 0

        # Force at least 25% of the fitnesses to 1 (does not overwrite the zeroed out ones)
        index_choices -= set(y_train_zeros)
        y_train_ones = np.random.choice(
            list(index_choices), size=int(n_examples * 0.25), replace=False
        )
        y_train[y_train_ones] = 1

        SIMULATION_TYPE = [SimType.RANDOM]

        data = pd.DataFrame(np.hstack((X_train, y_train.reshape(-1, 1))))
        random_data_filename = (
            f"random_train_kickstart_{'aas' if AAS_ONLY else 'others'}.csv"
        )
        data.to_csv(os.path.join(new_round_folder, random_data_filename), index=False)
        used_experiments = None
        redo_experiments = None

    # When doing transfer learning (data dir method), Round 1 is a special case
    # using the 'new' ingredients only so that we can kickstart that side of the NN,
    # to prevent those weights from collapsing.
    #
    # So, for the second round for 20+19 CDM, we have to combine round 1 data
    # (which has only non-AA ingredient inputs) with the transfer data of the AA-only
    # experiment (file located at TRANSFER_DATA_DIR) in the following way:
    if tl_transition_round:
        INGREDIENTS = AA_SHORT + BASE_NAMES
        TEMPEST_INGREDIENTS = AA_NAMES_TEMPEST + BASE_NAMES_TEMPEST
        n_ingredients = len(INGREDIENTS)

        # Load transfer data
        if "train_pred" not in TRANSFER_DATA_DIR:
            raise Exception("TRANSFER_DATA_DIR must point to a 'train_pred' CSV.")

        transfer_data = utils.normalize_ingredient_names(
            pd.read_csv(TRANSFER_DATA_DIR, index_col=None)
        )
        # Expand dimensions by the length of non-AA data
        transfer_data = fill_new_ingredients(
            transfer_data,
            original_size=len(AA_SHORT),
            fill_column_names=list(
                range(len(AA_SHORT), len(AA_SHORT) + len(BASE_NAMES))
            ),
            fill_on_right=True,
        )

        # Load Round 1 data (already padded)
        round_one_data_dir = os.path.join(new_round_folder, "train_pred.csv")
        round_one_data = utils.normalize_ingredient_names(
            pd.read_csv(round_one_data_dir, index_col=None)
        )

        # Concat data together
        cols_new = list(range(n_ingredients)) + ["y_true", "y_pred", "y_pred_var"]
        transfer_data.columns = round_one_data.columns = cols_new
        combined_data = pd.concat(
            [transfer_data, round_one_data], axis=0, ignore_index=True
        )

        # Backup og_train_pred and replace files:
        round_one_data_backup_dir = os.path.join(
            new_round_folder, "train_pred_orig.csv"
        )
        shutil.copyfile(round_one_data_dir, round_one_data_backup_dir)
        combined_data.to_csv(round_one_data_dir, index=None)

        # Update references to new data
        X_train = combined_data.iloc[:, :n_ingredients].to_numpy()
        y_train = combined_data.loc[:, "y_true"].to_numpy()

    # Train the models on the data from all previous rounds (excluding Round 1)
    if MODEL_TYPE == ModelType.GPR:
        # Train GPR Model
        model = GPRModel()
        model.train(X_train, y_train)
    elif TRANSFER_MODEL_FOLDER and NEW_ROUND_N == 1:
        # Use purely pre-trained NN model for 1st round
        models_folder = os.path.join(new_round_folder, f"nn_models")
        if os.path.exists(models_folder):
            raise (
                Exception(
                    f"File exists: '{models_folder}'. Cannot copy pre-trained models to here unless you remove it first."
                )
            )
        shutil.copytree(TRANSFER_MODEL_FOLDER, models_folder)
        model = transfer_model
    else:
        # Train pre-trained NN model using new data
        models_folder = os.path.join(new_round_folder, f"nn_models")
        model = NeuralNetModel(models_folder)

        transfer_models = transfer_model.models if TRANSFER_MODEL_FOLDER else []
        model.train(
            X_train,
            y_train,
            n_ingredients=n_ingredients,
            n_bags=config["n_bags"],
            bag_proportion=1.0,
            epochs=50,
            batch_size=360,
            lr=0.001,
            transfer_models=transfer_models,
        )

    if DIRECTION == SimDirection.DOWN:
        starting_media = np.ones(n_ingredients)
        direction = SimDirection.DOWN
        batch_size = BATCH_SIZE
    elif DIRECTION == SimDirection.UP:
        starting_media = np.zeros(n_ingredients)
        direction = SimDirection.UP
        batch_size = BATCH_SIZE
    elif DIRECTION == SimDirection.BOTH:
        starting_media = np.ones(n_ingredients)
        direction = SimDirection.DOWN
        batch_size = BATCH_SIZE // 2

    # Create the batches
    all_metrics = {}
    batch, batch_used, metrics = make_batch(
        model,
        starting_media,
        new_round_n=NEW_ROUND_N,
        batch_size=batch_size,
        sim_types=SIMULATION_TYPE,
        rollout_trajectories=N_ROLLOUTS,
        threshold=GROW_THRESHOLD,
        timeout=60 * TIMEOUT_MIN,
        unique=USE_UNIQUE,
        direction=direction,
        go_beyond_frontier=BEYOND_FRONTIER,
        used_experiments=used_experiments,
        redo_experiments=redo_experiments,
    )
    all_metrics[direction.name] = metrics

    ###### UP DIRECTION (used only when direction is BOTH) #####
    if DIRECTION == SimDirection.BOTH:
        direction = SimDirection.UP
        starting_media = np.zeros(n_ingredients)
        batch2, _, metrics = make_batch(
            model,
            starting_media,
            new_round_n=NEW_ROUND_N,
            batch_size=batch_size,
            sim_types=SIMULATION_TYPE,
            rollout_trajectories=N_ROLLOUTS,
            threshold=GROW_THRESHOLD,
            timeout=60 * TIMEOUT_MIN,
            unique=USE_UNIQUE,
            direction=direction,
            go_beyond_frontier=BEYOND_FRONTIER,
            used_experiments=batch_used,
        )
        batch = pd.concat((batch, batch2), ignore_index=True)
        all_metrics[direction.name] = metrics
    #############################################################

    # Output run metrics
    run_metrics_path = os.path.join(new_round_folder, "run_metrics.json")
    with open(run_metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    model.close()
    export_to_dp_batch(new_round_folder, batch, TEMPEST_INGREDIENTS, date, NICKNAME)


if __name__ == "__main__":
    # Read in command arguments
    parser = argparse.ArgumentParser(description="BacterAI Experiment Generator")

    parser.add_argument(
        "path",
        type=str,
        help="The path to the configuration file (.json)",
    )

    parser.add_argument(
        "-r",
        "--round",
        type=int,
        required=True,
        help="The new round number",
    )

    parser.add_argument(
        "-p",
        "--plot_only",
        action="store_true",
        help="Only export plots",
    )

    args = parser.parse_args()

    main(args)
