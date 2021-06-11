import argparse
import collections
import csv
from enum import Enum
import json
import math
import multiprocessing as mp
import os
import pickle
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from global_vars import *
from models import GPRModel, NeuralNetModel, ModelType
import utils
from utils import decoratortimer

os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
# print(robjects.r("version")) # R environment details


class SimType(Enum):
    """The supported simulation types when performing simulations.

    RANDOM:
        Takes a random action given a set of available actions.
    GREEDY:
        Predicts the growth at for the L1Os at a given state. Chooses the best
        performing action.
    ROLLOUT:
        Perform rollout simulations for the L1Os at a given state. Chooses the
        best performing action.
    ROLLOUT_PROB:
        Perform rollout simulations for the L1Os at a given state. Samples a random
        action using the softmax(rollout rewards) as the probability distribution.
    """

    RANDOM = 0
    GREEDY = 1
    ROLLOUT = 2
    ROLLOUT_PROB = 3


class SimDirection(Enum):
    """The supported simulation directions
    Down is working from the top down, Up is working from the bottom up.
    """

    DOWN = 0
    UP = 1
    BOTH = 2

    def action_value(self):
        """The value of the action to be taken.

        Returns
        -------
        int
            0 for DOWN and 1 for UP.
        """
        if self.value == 0:
            return 0
        return 1

    def target_value(self):
        """The value of the available target actions to be removed .

        Returns
        -------
        int
            1 for DOWN and 0 for UP.
        """
        if self.value == 0:
            return 1
        return 0


def get_data(path, train_size=0.1, test_size=0.1):
    data = pd.read_csv(path, index_col=None)
    if "environment" in data.columns:
        data = data.drop(columns="environment")

    X, y = data[data.columns[:-1]].to_numpy(), data["growth"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=1
    )
    return X_train, X_test, y_train, y_test


# @decoratortimer(2)
def rollout_trajectory(model, states, n_trajectories, threshold, sim_direction):
    """Performs a randomized rollout simulation. The random walk looks for all available
    actions at a current state, then chooses a random one. This process is repeated until no
    more actions can be taken, or if the actions results in no growth above the threshold,
    at which point we store the number of steps take (# of ingredients removed before no
    growth). The results are then averaged, to give a reward for each of the tested input
    states.

    Parameters
    ----------
    model : models.Model
        The model used when running the simulation.
    states : np.ndarray
        A 2D array of the states to run the rollouts on.
    n_trajectories : int
        The number of rollouts to perform, which the rewards are averaged over.
    threshold : float
        The grow/no grow threshold used to determine when to terminate
        a rollout simulation.
    sim_direction : SimDirection
        The directions in which the simulations run.

    Returns
    -------
    np.ndarray
        The averaged rewards of the trajectories performed on each of the input states.
    """
    trajectory_states = np.repeat(states, n_trajectories, axis=0)
    rewards = {i: [] for i in range(len(states))}
    states_boundaries = np.arange(0, n_trajectories * len(states) + 1, n_trajectories)

    reward_idx = 0
    step = 0
    # Random walk to remove 'n_trajectories' ingredients
    while trajectory_states.size > 0:
        choices = np.argwhere(trajectory_states == sim_direction.target_value())
        if choices.size == 0:
            for k, v in rewards.items():
                remaining = n_trajectories - len(v)
                if remaining > 0:
                    rewards[k] = v + [step] * remaining
            break

        boundaries = np.r_[
            0,
            np.flatnonzero(choices[1:, 0] > choices[:-1, 0]) + 1,
            choices.shape[0],
        ]

        for i in range(boundaries.shape[0] - 1):
            row = choices[boundaries[i], 0]
            idxes = choices[boundaries[i] : boundaries[i + 1], 1]
            np.random.shuffle(idxes)
            chosen_action = idxes[0]
            trajectory_states[row, chosen_action] = sim_direction.action_value()

        results, _ = model.evaluate(trajectory_states)
        # print(results)
        no_grows = np.argwhere(results < threshold)[:, 0]
        # print("len(no grows)", len(no_grows))
        new_state_boundaries = states_boundaries.copy()
        for result_idx in no_grows:
            for i in range(len(states_boundaries) - 1):
                lb = states_boundaries[i]
                ub = states_boundaries[i + 1]
                if lb <= result_idx < ub:
                    rewards[i].append(step)
                    new_state_boundaries[i + 1 :] -= 1
                    # print(i, (lb, result_idx, ub))
                    # print("new:", new_state_boundaries)
                    break
                # if i == len(states_boundaries) - 2:
                # print("DIDN'T FIND", result_idx)

        # print("len(trajectory_states)", len(trajectory_states))
        # rewards[reward_idx : reward_idx + len(no_grows)] = step
        # reward_idx += len(no_grows)

        states_boundaries = new_state_boundaries
        trajectory_states = np.delete(trajectory_states, no_grows, axis=0)
        step += 1
        # print(trajectory_states)
        # print()
    # print("DONE")
    # for k, v in rewards.items():
    #     print(f"{k}: {v}, {len(v)}")
    rewards = np.array(list(rewards.values()))
    return rewards.mean(axis=1)


@decoratortimer(2)
def perform_simulations(
    model,
    state,
    n,
    threshold,
    sim_type,
    sim_direction,
    unique=False,
    batch_set=None,
    timeout=None,
    n_rollout_trajectories=1,
    go_beyond_frontier=True,
):
    """Performs simulations and generate a batch of experiments to determine the
    'growth frontier' of a bacteria. The simulations determine available actions
    and chooses the next best action to take from the current state. Depending on the
    simulation type, this method differs. If there are no actions that result in a
    predicted growth, the simulation terminates and adds the desired state to the
    batch to test.

    Parameters
    ----------
    model : models.Model
        The model used when running the simulation.
    state : np.ndarray()
        The starting state of the media.
    n : Int
        The number of simulations to perform for this batch.
    threshold : float
        The grow/no grow threshold used to determine when to terminate
        a rollout simulation.
    sim_type : SimType
        The type of simulations to run.
    sim_direction : SimDirection
        The directions in which the simulations run.
    unique : bool, optional
        Take only unique states for the batch, by default False
    batch_set : set(tuple), optional
        The current states already in the batch, by default None
    timeout : int, optional
        The timeout length before forced temination of the simulations
        in seconds, by default None
    n_rollout_trajectories : int, optional
        The number of simulations to perform per state in the rollouts,
        by default 1
    go_beyond_frontier : bool, optional
        Add the state one step beyond the determined 'growth frontier',
        by default True

    Returns
    -------
    pd.DataFrame
        The batch of experiments to perform, where each row is a state
        to test and their associated metadata (simulation parameters, predicted
        growth, etc.)
    """
    state = state.astype(int)
    if batch_set == None:
        batch_set = set()
    batch = []
    batch_frontier_types = []
    terminating_growths = []
    terminating_variances = []

    desc = f"Performing {sim_type.name} Sims ({sim_direction.name})"
    tq = tqdm(total=n, desc=desc)
    not_timed_out = True
    start_time = time.time()
    loops = 1
    n_found_but_exists = 0

    # n_matches_softmax = 0
    # n_softmax = 0
    # n_matches_custom = 0
    # n_custom = 0

    while len(batch) < n and not_timed_out:
        tq.desc = f"{desc} ({loops} loops)"
        current_state = state.copy()

        current_grow_pred = 0
        current_grow_var = 0
        while (current_state == sim_direction.target_value()).sum() > 0:
            choices = np.argwhere(current_state == sim_direction.target_value())[:, 0]
            # print(f"CHOICES: {choices}")
            if choices.size == 0:
                break

            test_states = np.tile(current_state, (choices.size, 1))
            if sim_type == SimType.RANDOM:
                action = np.random.choice(choices, 1, False)  # Random leave-one-out
                test_states[0, action] = sim_direction.action_value()  # Take action
                test_states = test_states[0].reshape((1, -1))  # Reshape to 2D
                choices = [action]

            elif sim_type == SimType.GREEDY:
                # Take all leave-one-out actions
                test_states[
                    np.arange(choices.size), choices
                ] = sim_direction.action_value()

            elif sim_type == SimType.ROLLOUT or sim_type == SimType.ROLLOUT_PROB:
                # Take all leave-one-out actions
                rollout_results = np.zeros(choices.size)
                test_states[
                    np.arange(choices.size), choices
                ] = sim_direction.action_value()

                # Perform rollouts
                rollout_results = rollout_trajectory(
                    model,
                    test_states,
                    n_rollout_trajectories,
                    threshold,
                    sim_direction,
                )

                # print("choices:", choices)
                if sim_type == SimType.ROLLOUT_PROB:
                    # Pick an action idx from a distribution based on softmax of rollout results
                    # Adaptive K, defaults to 100, where softmax() acts as ~max(). Then as
                    # the number of duplicate media increases, K decreases. At K=1, it
                    # is the standard softmax(). At K=0, the choice become random. The
                    # percent of actions remaining is also calculated to contribute a
                    # depth decay effect as well.
                    n_actions_remaining = (
                        current_state == sim_direction.target_value()
                    ).sum()
                    percent_remaining = n_actions_remaining / len(current_state)
                    a = 100
                    k = -a * (1 - np.exp(-np.power(n_found_but_exists / 30, 3))) + a
                    k = (k * 0.50) * (1 + percent_remaining)

                    # Weighted softmax
                    p = utils.softmax(rollout_results, k=k)
                    action_idx = np.random.choice(choices.size, 1, p=p)[0]
                    max_action_idx = np.argsort(rollout_results)[-1]

                    # if match:
                    #     n_matches_custom += 1
                    # n_custom += 1

                    # # pure softmax
                    # p = utils.softmax(rollout_results)
                    # action_idx = np.random.choice(choices.size, 1, p=p)[0]
                    # match = max_action_idx == action_idx
                    # if match:
                    #     n_matches_softmax += 1
                    # n_softmax += 1

                    # match = max_action_idx == action_idx
                    # match_text = "MATCH" if match else "NO MATCH"
                    # print()
                    # print(f"{rollout_results}")
                    # print(f"{match_text}\t{action_idx=}, {max_action_idx=}")
                    # print(f"{n_found_but_exists=}, {k=}")
                else:
                    # Pick highest predicted reward (mean # removed)
                    action_idx = np.argsort(rollout_results)[-1]

                action = choices[action_idx]
                test_states = test_states[action_idx].reshape((1, -1))
                choices = [action]

            results, results_vars = model.evaluate(test_states)
            # Pick highest predicted growth as best action
            best_action_idx = np.argsort(results)[-1]
            best_action = choices[best_action_idx]

            # Keep track of prev state values
            old_state = current_state.copy()
            old_growth_result = current_grow_pred
            old_growth_var = current_grow_var
            # Set new state values
            new_state = current_state.copy()
            new_growth_result = float(results[best_action_idx])
            new_growth_var = float(results_vars[best_action_idx])
            new_state[best_action] = sim_direction.action_value()  # Take best action

            is_down = sim_direction == SimDirection.DOWN
            grows_present = (results >= threshold).sum() > 0

            if (is_down and grows_present) or (not is_down and not grows_present):
                # Keep going if grows are present and DOWN direction, or
                # Keep going if no grows are present and UP direction

                # Update state values
                current_state = new_state
                current_grow_pred = new_growth_result
                current_grow_var = new_growth_var

            elif (is_down and (not grows_present or new_state.sum() == 0)) or (
                not is_down and (grows_present or new_state.sum() == len(new_state))
            ):
                # If going DOWN terminate if:
                #   - no more grows present or removed all ingredients
                #   - Use old state (last known growth predicted), or
                # If going UP terminate if:
                #   - there are grows present or added all ingredients
                #   - Use new state (first known growth predicted)
                if is_down:
                    f_state, b_state = old_state, new_state
                    f_grow_result, b_grow_result = old_growth_result, new_growth_result
                    f_grow_var, b_grow_var = old_growth_var, new_growth_var
                else:
                    f_state, b_state = new_state, old_state
                    f_grow_result, b_grow_result = new_growth_result, old_growth_result
                    f_grow_var, b_grow_var = new_growth_var, old_growth_var

                if go_beyond_frontier:
                    # Add both the "frontier" and "beyond frontier" states
                    states = [f_state, b_state]
                    growth_preds = [f_grow_result, b_grow_result]
                    var_preds = [f_grow_var, b_grow_var]
                    frontier_types = ["FRONTIER", "BEYOND"]
                else:
                    states = [f_state]
                    growth_preds = [f_grow_result]
                    var_preds = [f_grow_var]
                    frontier_types = ["FRONTIER"]

                for st, gr, va, ft in zip(
                    states, growth_preds, var_preds, frontier_types
                ):
                    key = tuple(st)
                    if key not in batch_set or not unique:
                        batch.append(st)
                        terminating_growths.append(gr)
                        terminating_variances.append(va)
                        batch_frontier_types.append(ft)
                        batch_set.add(key)
                        tq.update()
                        print(f"ADDED: {st} - {ft}")
                        if sim_type == SimType.ROLLOUT_PROB:
                            n_found_but_exists -= 1
                            n_found_but_exists = max(n_found_but_exists, 0)
                    else:
                        if sim_type == SimType.ROLLOUT_PROB:
                            n_found_but_exists += 1
                        print(f"EXISTS: {st} - {ft}")

                    if len(batch) >= n:
                        break
                break

        if timeout is not None:
            not_timed_out = (time.time() - start_time) <= timeout
        loops += 1

    tq.close()
    if batch:
        batch = pd.DataFrame(np.vstack(batch))
        batch["type"] = sim_type.name
        batch["direction"] = sim_direction.name
        batch["frontier_type"] = batch_frontier_types
        batch["growth_pred"] = terminating_growths
        batch["var"] = terminating_variances
    else:
        batch = pd.DataFrame()

    # print("MATCH RESULTS")
    # print(f"Softmax: {n_matches_softmax/n_softmax:.4f}")
    # print(f"Weighted Softmax: {n_matches_custom/n_custom:.4f}")

    return batch, batch_set


def export_to_dp_batch(parent_path, batch, date, nickname=None):
    if len(batch) == 0:
        print("Empty Batch: No files generated.")
        return

    batch = batch.rename(columns={a: b for a, b in zip(range(20), AA_NAMES)})
    batch = batch.sort_values(by=["growth_pred", "var"], ascending=[False, True])
    batch.to_csv(os.path.join(parent_path, f"batch_gpr_meta_{date}.csv"), index=None)

    # DeepPhenotyping compatible list
    batch = batch.drop(columns=batch.columns[20:])
    # batch = batch.sort_values(by=AA_NAMES, ascending=False)
    nickname = f"_{nickname}" if nickname != None else ""

    with open(
        os.path.join(parent_path, f"batch_gpr_dp{nickname}_{date}.csv"), "w"
    ) as f:
        writer = csv.writer(f, delimiter=",")
        for row_idx, row_data in batch.iterrows():
            row_data = row_data[row_data == 0]
            removed_ingredients = list(row_data.index.to_numpy())
            writer.writerow(removed_ingredients)


def make_batch(
    model,
    media,
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
            unique=unique,
            timeout=timeout,
            batch_set=batch_set,
            n_rollout_trajectories=rollout_trajectories,
            go_beyond_frontier=go_beyond_frontier,
        )
        sub_batches.append(batch)

    batch = pd.concat([redo_experiments] + sub_batches, ignore_index=True)
    return batch, batch_set


# def plot_and_export_data(
#     model, X_trainR, X_testR, X_train, X_test, y_train, y_test, date
# ):
#     y_train_pred, y_train_var = sample_GP(model, X_trainR)
#     y_test_pred, y_test_var = sample_GP(model, X_testR)

#     train_data = pd.DataFrame(
#         np.hstack(
#             (
#                 X_train,
#                 y_train.reshape((-1, 1)),
#                 y_train_pred.reshape((-1, 1)),
#                 y_train_var.reshape((-1, 1)),
#             )
#         )
#     )
#     train_data.columns = list(range(20)) + ["y_true", "y_pred", "y_pred_var"]
#     test_data = pd.DataFrame(
#         np.hstack(
#             (
#                 X_test,
#                 y_test.reshape((-1, 1)),
#                 y_test_pred.reshape((-1, 1)),
#                 y_test_var.reshape((-1, 1)),
#             )
#         )
#     )
#     test_data.columns = list(range(20)) + ["y_true", "y_pred", "y_pred_var"]
#     train_data.to_csv(f"gpr_train_pred_{date}.csv", index=None)
#     test_data.to_csv(f"gpr_test_pred_{date}.csv", index=None)

#     fig, axs = plt.subplots(
#         nrows=2, ncols=1, sharex=False, sharey=False, figsize=(6, 10)
#     )

#     test_mse = mean_squared_error(y_test_pred, y_test)
#     sort_order = np.argsort(y_test)
#     print("Test data MSE:", test_mse)
#     print(y_test_pred.shape)
#     print(y_test.shape)
#     axs[0].scatter(range(len(y_test_pred)), y_test_pred[sort_order], s=3, alpha=0.50)
#     axs[0].plot(range(len(y_test)), y_test[sort_order], "-r")
#     axs[0].set_title(f"Test Set Predictions (MSE={round(test_mse, 3)})")
#     axs[0].set_xlabel("True")
#     axs[0].set_ylabel("Prediction")

#     train_mse = mean_squared_error(y_train_pred, y_train)
#     sort_order = np.argsort(y_train)
#     print("Train data MSE:", train_mse)
#     print(y_train_pred.shape)
#     print(y_train.shape)
#     axs[1].scatter(range(len(y_train_pred)), y_train_pred[sort_order], s=3, alpha=0.50)
#     axs[1].plot(range(len(y_train)), y_train[sort_order], "-r")
#     axs[1].set_title(f"Train Set Predictions (MSE={round(train_mse, 3)})")
#     axs[1].set_xlabel("True")
#     axs[1].set_ylabel("Prediction")

#     plt.tight_layout()
#     plt.savefig(f"result_gpr.png")


def process_results(prev_folder, new_folder, threshold, n_redos=0, plot_only=False):
    """Process the results of the previous round, generate plots, and
    return batch information to be used when generating the new round's
    batch.

    Parameters
    ----------
    prev_folder : str
        The folder path of the previous round.
    new_folder : str
        The folder path of the new round.
    threshold : float
        The grow/no grow threshold used to determine when to terminate
        a rollout simulation.
    n_redos : int, optional
        The number of experiments from the previous batch to rescreen,
        by default 0
    plot_only : bool, optional
        Only save plot, don't save/export any other files.

    Returns
    -------
    (np.ndarray, np.ndarray, set(tuple), pd.DataFrame)
        The new training set inputs, the  new training set labels, the
        experiments used in all previous experimments, the set of
        experiments to redo.
    """

    prev_folder_contents = os.listdir(prev_folder)
    new_folder_contents = os.listdir(prev_folder)

    mapped_path = None
    batch_path = None
    dataset_path = None
    for i in prev_folder_contents:
        if "mapped_data" in i:
            mapped_path = os.path.join(prev_folder, i)
        elif "batch_gpr_meta" in i and "results" not in i:
            batch_path = os.path.join(prev_folder, i)
        elif "gpr_train_pred" in i:
            dataset_path = os.path.join(prev_folder, i)

    new_dataset_path = os.path.join(new_folder, "gpr_train_pred.csv")

    # Merge results (mapped data) with predictions (batch data)
    data, plate_controls, plate_blanks = utils.process_mapped_data(mapped_path)
    batch_df = pd.read_csv(batch_path, index_col=None)
    results = pd.merge(
        batch_df, data, how="left", left_on=AA_NAMES, right_on=AA_NAMES, sort=True
    )
    results.iloc[:, :20] = results.iloc[:, :20].astype(int)
    results["depth"] = 20 - results.iloc[:, :20].sum(axis=1)
    results = results.sort_values(["depth", "fitness"], ascending=False)
    if "frontier_type" not in results.columns:
        results["frontier_type"] = "FRONTIER"
        print("Added 'frontier_type' column")
    if not plot_only:
        results.to_csv(os.path.join(prev_folder, "results_all.csv"), index=None)

    # Generate results figure
    plot_results(prev_folder, results, threshold)
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
        dataset = pd.read_csv(dataset_path, index_col=None)
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
    redo_experiments = results_bad.drop(columns=remove_cols)
    if n_redos > 0:
        redos_chosen = results_grow_only.iloc[:n_redos, :].drop(columns=remove_cols)
        redo_experiments = pd.concat(
            (redo_experiments, redos_chosen), ignore_index=True
        )
    redo_experiments["type"] = "REDO"
    redo_experiments.columns = list(range(20)) + list(redo_experiments.columns[20:])

    # Save and output successful results
    results_grow_only.to_csv(
        os.path.join(prev_folder, "results_grow_only.csv"), index=False
    )

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


def plot_results(prev_folder, results, threshold):
    results = results.sort_values(by="growth_pred").reset_index(drop=True)
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=False,
        sharey=False,
        figsize=(12, 8),
        gridspec_kw={"width_ratios": [1.25, 2]},
    )

    colors = {
        "ROLLOUT": "violet",
        "ROLLOUT_PROB": "violet",
        "RANDOM": "dodgerblue",
        "GREEDY": "orangered",
        "REDO": "limegreen",
    }
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
                color=colors[group_name],
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
            color = colors[group_name]
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

    plt.suptitle(f"Experiment: {prev_folder}")
    plt.tight_layout()
    plt.savefig(os.path.join(prev_folder, "results_graphic.png"), dpi=400)


def main(args):
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
    N_REDOS = int(BATCH_SIZE * 0.10)

    date = datetime.datetime.now().isoformat().replace(":", ".")
    prev_round_folder = os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N-1}")
    new_round_folder = os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N}")
    if not os.path.exists(new_round_folder):
        os.makedirs(new_round_folder)
    if NEW_ROUND_N > 1:
        X_train, y_train, used_experiments, redo_experiments = process_results(
            prev_round_folder, new_round_folder, GROW_THRESHOLD, N_REDOS, args.plot_only
        )
        BATCH_SIZE -= N_REDOS

    else:
        # data = pd.read_csv(
        #     "experiments/04-30-21/up/Round1/random_train_kickstart.csv", index_col=None
        # )
        # X_train, y_train = data.iloc[:, :20].to_numpy(), data.iloc[:, -1].to_numpy()
        if MODEL_TYPE == ModelType.NEURAL_NET:
            n_examples = BATCH_SIZE
        else:
            n_examples = 1000
        X_train = np.random.rand(n_examples, 20)
        X_train[X_train >= 0.5] = 1
        X_train[X_train < 0.5] = 0
        y_train = np.random.rand(n_examples, 1)
        # y_train[y_train >= 0.5] = 1
        # y_train[y_train < 0.5] = 0

        print("First Round, making random training set:")
        print(X_train)
        print(y_train)
        data = np.hstack((X_train, y_train))
        data = pd.DataFrame(data, columns=list(range(20)) + ["y_true"])
        data.to_csv(
            os.path.join(new_round_folder, "random_train_kickstart.csv"), index=False
        )
        used_experiments = None
        redo_experiments = None

        if MODEL_TYPE == ModelType.NEURAL_NET:
            data = data.rename(columns={"y_true": "growth_pred"})
            data["var"] = 0
            export_to_dp_batch(new_round_folder, data, date, NICKNAME)
            return

    if MODEL_TYPE == ModelType.GPR:
        model = GPRModel()
        model.train(X_train, y_train)
    else:
        models_folder = os.path.join(new_round_folder, f"nn_models")
        model = NeuralNetModel(models_folder)

        model.train(
            X_train,
            y_train,
            # n_bags=1,
            n_bags=config["n_bags"],
            bag_proportion=1.0,
            epochs=50,
            batch_size=360,
            lr=0.001,
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

    batch, batch_used = make_batch(
        model,
        starting_media,
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

    #################### UP DIRECTION ####################
    if DIRECTION == SimDirection.BOTH:
        # batch_size = BATCH_SIZE - len(batch)
        direction = SimDirection.UP
        starting_media = np.zeros(20)
        batch2, batch_used = make_batch(
            model,
            starting_media,
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
    #######################################################

    model.close()
    export_to_dp_batch(new_round_folder, batch, date, NICKNAME)


if __name__ == "__main__":
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
