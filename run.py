import argparse
import csv
import json
import os
import shutil

import numpy as np
import pandas as pd

from global_vars import *
from models import GPRModel, NeuralNetModel, ModelType
from plot import plot_redos, plot_results
from sim import SimType, SimDirection, perform_simulations
import utils

os.environ["MKL_DEBUG_CPU_TYPE"] = "5"  # Use IntelMKL - does this work?


def export_to_dp_batch(parent_path, batch, date, nickname=None):
    """ Export the BacterAI batch to a Deep Phenotyping-compatible file. """

    if len(batch) == 0:
        print("Empty Batch: No files generated.")
        return

    batch = batch.rename(columns={a: b for a, b in zip(range(20), AA_NAMES_TEMPEST)})
    batch = batch.sort_values(by=["growth_pred", "var"], ascending=[False, True])
    batch.to_csv(os.path.join(parent_path, f"batch_meta_{date}.csv"), index=None)

    # DeepPhenotyping compatible list
    batch = batch.drop(columns=batch.columns[20:])
    nickname = f"_{nickname}" if nickname != None else ""

    with open(os.path.join(parent_path, f"batch_dp{nickname}_{date}.csv"), "w") as f:
        writer = csv.writer(f, delimiter=",")
        for row_idx, row_data in batch.iterrows():
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
    for idx, sim_type in enumerate(sim_types):
        if idx == n_types - 1:
            n_exps = batch_size - sum([len(x) for x in sub_batches])
        print(idx, sim_type, batch_size, n_exps, sum([len(x) for x in sub_batches]))
        batch, batch_set = perform_simulations(
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

    batch = pd.concat([redo_experiments] + sub_batches, ignore_index=True)
    return batch, batch_set


def process_results(
    folder,
    prev_folder,
    new_folder,
    new_round_n,
    threshold,
    redo_prev_round=False,
    plot_only=False,
):
    """Process the results of the previous round, generate plots, and
    return batch information to be used when generating the new round's
    batch.

    Parameters
    ----------
    folder : str
        The folder path of the current round.
    prev_folder : str
        The folder path of the previous round.
    new_folder : str
        The folder path of the new round.
    new_round_n : int
        The number of the new round.
    threshold : float
        The grow/no grow threshold used to determine when to terminate
        a rollout simulation.
    redo_prev_round : bool, optional
        Run the previous round's batch again,
        by default False
    plot_only : bool, optional
        Only save plot, don't save/export any other files.

    Returns
    -------
    (np.ndarray, np.ndarray, set(tuple), pd.DataFrame)
        The new training set inputs, the  new training set labels, the
        experiments used in all previous experimments, the set of
        experiments to redo.
    """

    folder_contents = os.listdir(folder)

    # Get paths for the necessary files in the current folder (new round # - 1)
    mapped_path = None
    batch_path = None
    dataset_path = None
    for i in folder_contents:
        if "mapped_data" in i:
            mapped_path = os.path.join(folder, i)
        elif "batch_meta" in i and "results" not in i:
            batch_path = os.path.join(folder, i)
        elif "train_pred" in i:
            dataset_path = os.path.join(folder, i)

    new_dataset_path = os.path.join(new_folder, "train_pred.csv")

    # Merge results (mapped data) with predictions (batch data)
    data, plate_controls, plate_blanks = utils.process_mapped_data(mapped_path)
    batch_df = utils.normalize_ingredient_names(pd.read_csv(batch_path, index_col=None))
    results = pd.merge(
        batch_df, data, how="left", left_on=AA_SHORT, right_on=AA_SHORT, sort=True
    )

    # Plot the rescreen experiments if available
    if "is_redo" in results.columns:
        redo_results = results[results["is_redo"] == True]
        results = results[results["is_redo"] != True]

        if prev_folder != None:
            prev_result_path = os.path.join(prev_folder, "results_all.csv")
            prev_results = utils.normalize_ingredient_names(
                pd.read_csv(prev_result_path, index_col=None)
            )
            plot_redos(folder, prev_results, redo_results)

    # Process results
    results.iloc[:, :20] = results.iloc[:, :20].astype(int)
    results["depth"] = 20 - results.iloc[:, :20].sum(axis=1)
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
    cols = list(results.columns[:20]) + ["fitness", "growth_pred", "var"]
    cols_new = list(range(20)) + ["y_true", "y_pred", "y_pred_var"]
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
        data_batch.iloc[:, :20] = data_batch.iloc[:, :20].astype(int)
        dataset.columns = data_batch.columns = cols_new
        new_dataset = pd.concat([dataset, data_batch], ignore_index=True)

    # Used experiments are the new dataset (old dataset plus "good" experiments from current round)
    used_experiments = set(map(tuple, new_dataset.to_numpy()[:, :20]))
    new_dataset.to_csv(new_dataset_path, index=None)
    X_train = new_dataset.iloc[:, :20].to_numpy()
    y_train = new_dataset.loc[:, "y_true"].to_numpy()

    # Assemble redo experiments, starting with bad experiments
    results_grow_only = results[results["fitness"] >= threshold]
    remove_cols = [
        "fitness",
        "environment",
        "strain",
        "parent_plate",
        "initial_od",
        "final_od",
        "delta_od",
        "depth",
    ]

    # Obtain the experiments from the current round to rescreen in the new round
    if redo_prev_round:
        redo_experiments = batch_df[batch_df["type"] != "REDO"]
        if "is_redo" in redo_experiments.columns:
            redo_experiments = redo_experiments[redo_experiments["is_redo"] == False]
        print(f"Redoing {len(redo_experiments)} experiments from previous round.")

        redo_experiments["is_redo"] = True
        redo_experiments["round"] = new_round_n - 1
        redo_experiments.columns = list(range(20)) + list(redo_experiments.columns[20:])

    # Save and output successful results
    results_grow_only.to_csv(os.path.join(folder, "results_grow_only.csv"), index=False)

    # Print some metrics/results
    top_10 = results_grow_only.iloc[:10, :]
    print("Media Results (Top 10):")
    for idx, (_, row) in enumerate(top_10.iterrows()):
        print(f"{idx+1:2}. Depth: {row['depth']:2}, Fitness: {row['fitness']:.3f}")
        for l in row[:20][row[:20] == 1].index:
            print(f"\t{l}")

    print(f"Total unique experiments: {len(used_experiments)}")
    print(
        f"Total redo experiments chosen: {len(redo_experiments)} ({len(results_bad)} 'bad' repeats)"
    )

    return X_train, y_train, used_experiments, redo_experiments


def main(args):
    # Process command line args and load the specified configuration file
    NEW_ROUND_N = args.round

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

    if TRANSFER_MODEL_FOLDER is not None:
        print(f"Using transfer learning from '{TRANSFER_MODEL_FOLDER}' model.")
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
        X_train, y_train, used_experiments, redo_experiments = process_results(
            current_round_folder,
            prev_round_folder,
            new_round_folder,
            NEW_ROUND_N,
            GROW_THRESHOLD,
            redo_prev_round=True,
            plot_only=args.plot_only,
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

        # Create 1000 (arbitrary) binary AA inputs and assign random fitness [0, 1]
        n_examples = 1000
        X_train = np.random.rand(n_examples, 20)
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
        data.to_csv(
            os.path.join(new_round_folder, "random_train_kickstart.csv"), index=False
        )
        used_experiments = None
        redo_experiments = None

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
            n_bags=config["n_bags"],
            bag_proportion=1.0,
            epochs=50,
            batch_size=360,
            lr=0.001,
            transfer_models=transfer_models,
        )

    if DIRECTION == SimDirection.DOWN:
        starting_media = np.ones(20)
        direction = SimDirection.DOWN
        batch_size = BATCH_SIZE
    elif DIRECTION == SimDirection.UP:
        starting_media = np.zeros(20)
        direction = SimDirection.UP
        batch_size = BATCH_SIZE
    elif DIRECTION == SimDirection.BOTH:
        starting_media = np.ones(20)
        direction = SimDirection.DOWN
        batch_size = BATCH_SIZE // 2

    # Create the batches
    batch, batch_used = make_batch(
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

    ###### UP DIRECTION (used only when direction is BOTH) #####
    if DIRECTION == SimDirection.BOTH:
        direction = SimDirection.UP
        starting_media = np.zeros(20)
        batch2, _ = make_batch(
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
    #############################################################

    model.close()
    export_to_dp_batch(new_round_folder, batch, date, NICKNAME)


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
