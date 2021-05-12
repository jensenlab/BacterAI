import collections
import csv
from enum import Enum
import math
import multiprocessing as mp
import os
import pickle
import time
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri as rpyn
from rpy2.robjects.packages import STAP
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from global_vars import *
import net
import utils

os.environ["MKL_DEBUG_CPU_TYPE"] = "5"
# print(robjects.r("version")) # R environment details


class SimType(Enum):
    RANDOM = 0
    GREEDY = 1
    ROLLOUT = 2


class SimDirection(Enum):
    DOWN = 0
    UP = 1

    def change_value(self):
        if self.value == 0:
            return 0
        return 1

    def target_value(self):
        if self.value == 0:
            return 1
        return 0


class ModelType(Enum):
    GPR = 0
    NEURAL_NET = 1


class Model(object):
    def __init__(self, model, model_type):
        self.model_type = model_type

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def train(self, X_train, y_train):
        pass

    def evaluate(self, X):
        pass

    def get_type(self):
        return self.model_type


class GPRModel(Model):
    def __init__(self):
        self.activate_R()
        super().__init__(self, ModelType.GPR)

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def train(self, X_train, y_train):
        self.model = self.gpr_lib.train_new_GP(X_train, y_train)

    def evaluate(self, X, clip=True, n=1):
        if self.model is None:
            raise Exception("GPR model needs to be trained before evaluating.")

        resultR = self.gpr_lib.sample_GP(self.model, X, n)
        result = np.array(resultR)
        if clip:
            result = np.clip(result, 0, 1)
        samples, variances = result[:, 0], result[:, 1]
        return samples, variances

    def activate_R(self):
        with open("gpr_lib.R", "r") as f:
            s = f.read()
            self.gpr_lib = STAP(s, "gpr_lib")
            robjects.r("Sys.setenv(MKL_DEBUG_CPU_TYPE = '5')")
        rpyn.activate()

    def close(self):
        # Clean up R's GPR model object
        self.gpr_lib.delete_GP(self.model)


class NeuralNetModel(Model):
    def __init__(self, model_path):
        self.model_path = model_path
        super().__init__(self, ModelType.NEURAL_NET)

    def __enter__(self):
        return self

    def __exit__(self):
        self.close()

    def train(
        self,
        X_train,
        y_train,
        n_bags=25,
        bag_proportion=1.0,
        epochs=50,
        batch_size=360,
        lr=0.001,
    ):
        net.train_bagged(
            X_train,
            y_train,
            self.model_path,
            **kwargs,
        )

    def evaluate(self, X, model_path_folder, clip=True):
        model_names = [f for f in os.listdir(model_path_folder) if "bag_model" in f]
        if len(model_names) == 0:
            raise Exception("Neural net model needs to be trained before evaluating.")

        preds = np.zeros((len(X), len(model_names)))
        for i, name in enumerate(model_names):
            model = torch.load(os.path.join(model_path_folder, name))
            y_pred = model.evaluate(X)
            preds[:, i] = y_pred

        predictions = np.mean(preds, axis=1)
        variances = np.var(preds, axis=1, ddof=0)
        return predictions, variances


def decoratortimer(decimal):
    def decoratorfunction(f):
        def wrap(*args, **kwargs):
            time1 = time.monotonic()
            result = f(*args, **kwargs)
            time2 = time.monotonic()
            print(
                "{:s} function took {:.{}f} ms".format(
                    f.__name__, ((time2 - time1) * 1000.0), decimal
                )
            )
            return result

        return wrap

    return decoratorfunction


def get_data(path, train_size=0.1, test_size=0.1):
    data = pd.read_csv(path, index_col=None)
    if "environment" in data.columns:
        data = data.drop(columns="environment")

    X, y = data[data.columns[:-1]].values, data["growth"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=1
    )
    return X_train, X_test, y_train, y_test


# @decoratortimer(2)
def rollout_trajectory(model, state, n_trajectories, threshold, sim_direction):
    trajectory_states = np.tile(state, (n_trajectories, 1))
    rewards = np.zeros((n_trajectories,))

    reward_idx = 0
    step = 0
    # Random walk to remove 'n_trajectories' ingredients
    while trajectory_states.size > 0:
        choices = np.argwhere(trajectory_states == sim_direction.target_value())
        if choices.size == 0:
            break

        s0 = np.r_[
            0,
            np.flatnonzero(choices[1:, 0] > choices[:-1, 0]) + 1,
            choices.shape[0],
        ]

        for i in range(s0.shape[0] - 1):
            row = choices[s0[i], 0]
            idxes = choices[s0[i] : s0[i + 1], 1]
            np.random.shuffle(idxes)
            chosen_action = idxes[0]
            trajectory_states[row, chosen_action] = sim_direction.change_value()

        results_grow_only, _ = model.evaluate(trajectory_states)

        idx_dels = list()
        for i, r in enumerate(results_grow_only):
            if r <= threshold:
                idx_dels.append(i)
                rewards[reward_idx] = step
                reward_idx += 1

        trajectory_states = np.delete(trajectory_states, idx_dels, axis=0)
        step += 1

    return rewards.mean()


# @decoratortimer(2)
# def rollout_simulations(
#     model,
#     state,
#     n,
#     threshold,
#     sim_direction,
#     n_trajectories=1,
#     depth=1,
#     unique=False,
#     batch_set=None,
#     timeout=None,
# ):
#     state = state.astype(int)
#     if batch_set == None:
#         batch_set = set()
#     batch = []
#     terminating_growth = []
#     variances = []

#     desc = "Performing Rollout Sims"
#     t = tqdm(total=n, desc=desc)
#     not_timed_out = True
#     start_time = time.time()
#     loops = 1
#     while len(batch) < n and not_timed_out:
#         t.desc = f"{desc} ({loops} its)"
#         trajectory_state = state.copy()

#         prev_result = 0
#         prev_var = 0
#         while (trajectory_state == sim_direction.target_value()).sum() > 0:
#             choices = np.argwhere(trajectory_state == sim_direction.target_value())[
#                 :, 0
#             ]
#             # print(f"CHOICES: {choices}")
#             if choices.size == 0:
#                 continue

#             rollout_states = np.tile(trajectory_state, (choices.size, 1))
#             results = np.zeros(choices.size)
#             for i in range(choices.size):
#                 rollout_states[i, choices[i]] = sim_direction.change_value()
#                 mean_reward = rollout_trajectory(
#                     model, rollout_states[i], n_trajectories, threshold, sim_direction
#                 )
#                 results[i] = mean_reward

#             best_action = np.argsort(results)[-1]  # Pick highest predicted growth

#             new_state = rollout_states[best_action].reshape((1, -1))
#             result, result_var = sample_GP(model, new_state)

#             # Take best action if there are still actions available (> 0 grows),
#             # or if the direction is up, set new state.
#             is_down = sim_direction == SimDirection.DOWN
#             if result[0] >= threshold or not is_down:
#                 trajectory_state = rollout_states[best_action]
#                 prev_result = float(result[0])
#                 prev_var = float(result_var[0])
#             # Then terminate by setting last successful media if direction is down,
#             # or set the current (first) successful media if direction is up.
#             if (is_down and result[0] < threshold) or (
#                 not is_down and result[0] >= threshold
#             ):
#                 key = tuple(trajectory_state)
#                 if key not in batch_set or not unique:
#                     batch.append(trajectory_state)
#                     terminating_growth.append(prev_result)
#                     variances.append(prev_var)
#                     batch_set.add(key)
#                     t.update()
#                 break

#         if timeout is not None:
#             not_timed_out = (time.time() - start_time) <= timeout
#         loops += 1

#     t.close()
#     if batch:
#         batch = pd.DataFrame(np.vstack(batch))
#         batch["type"] = "ROLLOUT"
#         batch["direction"] = sim_direction.name
#         batch["growth_pred"] = terminating_growth
#         batch["var"] = variances
#     else:
#         batch = pd.DataFrame()
#     return batch, batch_set


@decoratortimer(2)
def perform_simulations(
    model,
    state,
    n,
    threshold,
    sim_type,
    sim_direction,
    depth=1,
    unique=False,
    batch_set=None,
    timeout=None,
    n_rollout_trajectories=1,
):
    state = state.astype(int)
    if batch_set == None:
        batch_set = set()
    batch = []
    terminating_growth = []
    variances = []

    if sim_type == SimType.RANDOM:
        desc = "Performing Random Sims"
    elif sim_type == SimType.GREEDY:
        desc = "Performing Greedy Sims"
    elif sim_type == SimType.ROLLOUT:
        desc = "Performing Rollout Sims"

    t = tqdm(total=n, desc=desc)
    not_timed_out = True
    start_time = time.time()
    loops = 1
    while len(batch) < n and not_timed_out:
        t.desc = f"{desc} ({loops} its)"
        trajectory_state = state.copy()

        prev_result = 0
        prev_var = 0
        while (trajectory_state == sim_direction.target_value()).sum() > 0:
            choices = np.argwhere(trajectory_state == sim_direction.target_value())[
                :, 0
            ]
            # print(f"CHOICES: {choices}")
            if choices.size == 0:
                continue

            if sim_type == SimType.RANDOM:
                test_state = trajectory_state.copy()
                # chosen_action = np.random.choice(choices, 1, False)
                np.random.shuffle(choices)
                chosen_action = choices[0]
                test_state[chosen_action] = sim_direction.change_value()
                test_states = test_state.reshape((1, -1))

            elif sim_type == SimType.GREEDY:
                test_states = np.tile(trajectory_state, (choices.size, 1))
                for i in range(choices.size):
                    test_states[i, choices[i]] = sim_direction.change_value()

            elif sim_type == SimType.ROLLOUT:
                rollout_states = np.tile(trajectory_state, (choices.size, 1))
                results = np.zeros(choices.size)
                for i in range(choices.size):
                    rollout_states[i, choices[i]] = sim_direction.change_value()
                    mean_reward = rollout_trajectory(
                        model,
                        rollout_states[i],
                        n_rollout_trajectories,
                        threshold,
                        sim_direction,
                    )
                    results[i] = mean_reward

                best_action = np.argsort(results)[
                    -1
                ]  # Pick highest predicted reward (mean # removed)
                test_states = rollout_states[best_action].reshape((1, -1))
                chosen_action = choices[best_action]

            results, results_vars = model.evaluate(test_states)

            # Take best action if there are still actions available (> 0 grows),
            # or if the direction is up, set new state.
            is_down = sim_direction == SimDirection.DOWN
            if (results >= threshold).sum() > 0 or not is_down:
                best_action = np.argsort(results)[-1]  # Pick highest predicted growth
                if sim_type == SimType.GREEDY:
                    chosen_action = choices[best_action]

                trajectory_state[chosen_action] = sim_direction.change_value()
                prev_result = float(results[best_action])
                prev_var = float(results_vars[best_action])

            # Then terminate by setting last successful media if direction is down,
            # or set the current (first) successful media if direction is up.
            if (is_down and (results >= threshold).sum() == 0) or (
                not is_down and (results >= threshold).sum() > 0
            ):
                key = tuple(trajectory_state)
                if key not in batch_set or not unique:
                    batch.append(trajectory_state)
                    terminating_growth.append(prev_result)
                    variances.append(prev_var)
                    batch_set.add(key)
                    t.update()
                break

        if timeout is not None:
            not_timed_out = (time.time() - start_time) <= timeout
        loops += 1

    t.close()
    if batch:
        batch = pd.DataFrame(np.vstack(batch))
        batch["type"] = sim_type.name
        batch["direction"] = sim_direction.name
        batch["growth_pred"] = terminating_growth
        batch["var"] = variances
    else:
        batch = pd.DataFrame()
    return batch, batch_set


def export_to_dp_batch(parent_path, batch, date):
    if len(batch) == 0:
        print("Empty Batch: No files generated.")
        return

    batch = batch.rename(columns={a: b for a, b in zip(range(20), AA_NAMES)})
    batch = batch.sort_values(by="var", ascending=True)
    batch.to_csv(os.path.join(parent_path, f"batch_gpr_meta_{date}.csv"), index=None)

    # DeepPhenotyping compatible list
    batch = batch.drop(columns=batch.columns[20:])
    batch = batch.sort_values(by=AA_NAMES, ascending=False)
    with open(os.path.join(parent_path, f"batch_gpr_dp_{date}.csv"), "w") as f:
        writer = csv.writer(f, delimiter=",")
        for row_idx, row_data in batch.iterrows():
            row_data = row_data[row_data == 0]
            removed_ingredients = list(row_data.index.values)
            writer.writerow(removed_ingredients)


def make_batch(
    model,
    media,
    batch_size,
    rollout_trajectories,
    threshold,
    timeout=60 * 15,
    unique=True,
    direction=SimDirection.DOWN,
    used_experiments=None,
    redo_experiments=None,
):
    n_rollout = batch_size // 2
    # n_greedy = batch_size // 3
    batch_set = used_experiments
    rollout_batch, batch_set = perform_simulations(
        model,
        media,
        n_rollout,
        threshold,
        SimType.ROLLOUT,
        direction,
        unique=unique,
        timeout=timeout,
        batch_set=batch_set,
        n_rollout_trajectories=rollout_trajectories,
    )

    # greedy_batch, batch_set = perform_simulations(
    #     model,
    #     media,
    #     n_greedy,
    #     threshold,
    #     SimType.GREEDY,
    #     direction,
    #     unique=unique,
    #     batch_set=batch_set,
    #     timeout=timeout,
    # )

    n_random = batch_size - len(rollout_batch)  # - len(greedy_batch)
    rand_batch, batch_set = perform_simulations(
        model,
        media,
        n_random,
        threshold,
        SimType.RANDOM,
        direction,
        unique=unique,
        batch_set=batch_set,
    )

    batch = pd.concat([redo_experiments, rollout_batch, rand_batch])
    # batch = pd.concat([redo_experiments, rollout_batch, greedy_batch, rand_batch])
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


def process_results(prev_folder, new_folder, threshold, n_redos=0):
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
    results.to_csv(os.path.join(prev_folder, "results_all.csv"), index=None)

    # Keep experiments where all replicate wells are not marked "bad"
    results_bad = results.loc[results["bad"] != 0, :].drop(columns="bad")
    results = results.loc[results["bad"] == 0, :].drop(columns="bad")

    # Assemble new training data set, either from scratch or appending to previous Round's set
    cols = list(results.columns[:20]) + ["fitness", "growth_pred", "var"]
    cols_new = list(range(20)) + ["y_true", "y_pred", "y_pred_var"]
    if dataset_path == None:
        new_dataset = pd.DataFrame(
            results.loc[:, cols].values,
            columns=cols_new,
        )
    else:
        dataset = pd.read_csv(dataset_path, index_col=None)
        data_batch = results.loc[:, cols]
        data_batch.iloc[:, :20] = data_batch.iloc[:, :20].astype(int)
        dataset.columns = data_batch.columns = cols_new
        new_dataset = pd.concat([dataset, data_batch], ignore_index=True)

    # Used experiments are the new dataset (old dataset plus "good" experiments from current round)
    used_experiments = set(map(tuple, new_dataset.values[:, :20]))
    new_dataset.to_csv(new_dataset_path, index=None)
    X_train = new_dataset.values[:, :20]
    y_train = new_dataset.loc[:, "y_true"].values

    # Assemble redo experiments, starting with bad experiments
    remove_cols = [
        "growth_pred",
        "environment",
        "strain",
        "parent_plate",
        "initial_od",
        "final_od",
        "delta_od",
        "depth",
    ]
    redo_experiments = results_bad.drop(columns=remove_cols).rename(
        {"fitness": "growth_pred"}
    )
    redo_experiments["var"] = 0
    if n_redos > 0:
        redos_chosen = results.iloc[:n_redos, :].drop(columns=remove_cols)
        redo_experiments = pd.concat((redo_experiments, redos_chosen))
    redo_experiments["type"] = "REDO"

    # Save and output successful results
    results_grow_only = results[results["fitness"] >= threshold]
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

    # Generate results figure
    plot_results(prev_folder, results, threshold)

    return X_train, y_train, used_experiments, redo_experiments


def plot_results(prev_folder, results, threshold):
    results = results.sort_values(by="growth_pred").reset_index(drop=True)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=False,
        sharey=False,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.25, 2]},
    )
    greedy = results[results["type"] == "GREEDY"]
    rollout = results[results["type"] == "ROLLOUT"]
    random = results[results["type"] == "RANDOM"]
    colors = ["orange", "magenta", "blue"]
    for data, color in zip([greedy, rollout, random], colors):
        axs[0].plot(
            data.index, data["fitness"], ".", color=color, markersize=3, alpha=0.75
        )
    axs[0].plot(results.index, results["growth_pred"], "-", color="black")
    # axs[0].fill_between(
    #     results.index,
    #     results["growth_pred"] - results["var"],
    #     results["growth_pred"] + results["var"],
    #     facecolor="black",
    #     alpha=0.25,
    # )
    axs[0].set_xlabel("Assay N")
    axs[0].set_ylabel("Growth")
    axs[0].legend(
        [
            "Fitness - Greedy",
            "Fitness - Rollout",
            "Fitness - Random",
            "Model Prediction",
        ]
    )
    axs[0].set_title("Experiment Results")

    width = 0.25
    for i, (data, color) in enumerate(zip((greedy, rollout, random), colors)):
        data_g = collections.Counter(list(data[data["fitness"] >= threshold]["depth"]))
        data_ng = collections.Counter(list(data[data["fitness"] < threshold]["depth"]))
        bottom = [data_g[k] if k in data_g else 0 for k in data_ng.keys()]

        axs[1].bar(
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
        axs[1].bar(
            np.array(list(data_g.keys())) + width * i,
            data_g.values(),
            width=width,
            color=color,
        )

    axs[1].set_title("Depth")
    axs[1].set_xlabel("Depth (n_removed)")
    axs[1].set_ylabel("Count")
    axs[1].set_xticks(np.arange(0, 21) + 2 * width / 2)
    axs[1].set_xticklabels(np.arange(0, 21))
    axs[1].legend(
        [
            "Greedy - No Grow",
            "Greedy - Grow",
            "Rollout - No Grow",
            "Rollout - Grow",
            "Random - No Grow",
            "Random - Grow",
        ],
    )
    plt.suptitle(f"Experiment: {prev_folder}")
    plt.tight_layout()
    plt.savefig(os.path.join(prev_folder, "results_graphic.png"), dpi=400)


def check_uniqueness(batch):
    batch = batch.iloc[:, :20].values
    unique = np.unique(batch, axis=0)
    diff = len(batch) - len(unique)
    assert diff == 0, f"Found {diff} non-unique experiments in batch"


def main():
    GROW_THRESHOLD = 0.25
    NEW_ROUND_N = 9

    EXPT_FOLDER = "experiments/04-05-2021"
    # EXPT_FOLDER = "experiments/04-07-2021"
    TIMEOUT_MIN = 10
    BATCH_SIZE = 448
    N_REDOS = int(BATCH_SIZE * 0.10)

    # EXPT_FOLDER = "experiments/04-30-2021/both"
    # TIMEOUT_MIN = 30
    # BATCH_SIZE = 224
    # N_REDOS = int(BATCH_SIZE * 0.10) * 2

    USE_UNIQUE = True
    DIRECTION = SimDirection.DOWN

    prev_round_folder = os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N-1}")
    new_round_folder = os.path.join(EXPT_FOLDER, f"Round{NEW_ROUND_N}")
    if not os.path.exists(new_round_folder):
        os.makedirs(new_round_folder)
    if NEW_ROUND_N > 1:
        X_train, y_train, used_experiments, redo_experiments = process_results(
            prev_round_folder, new_round_folder, GROW_THRESHOLD, N_REDOS
        )
        BATCH_SIZE -= N_REDOS
    else:
        # data = pd.read_csv(
        #     "experiments/04-30-21/up/Round1/random_train_kickstart.csv", index_col=None
        # )
        # X_train, y_train = data.iloc[:, :20].values, data.iloc[:, -1].values
        n_examples = 1000
        X_train = np.random.rand(n_examples, 20)
        X_train[X_train >= 0.5] = 1
        X_train[X_train < 0.5] = 0
        y_train = np.random.rand(n_examples, 1)

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

    X_train = X_train[:100, :]
    y_train = y_train[:100]

    X_trainR = robjects.r.matrix(X_train, nrow=X_train.shape[0], ncol=X_train.shape[1])
    y_trainR = robjects.r.matrix(y_train, nrow=y_train.shape[0], ncol=1)

    model = train_new_GP(X_trainR, y_trainR)

    date = datetime.datetime.now().isoformat().replace(":", ".")
    if DIRECTION == SimDirection.DOWN:
        starting_media = np.ones(20)
    else:
        starting_media = np.zeros(20)
    batch, batch_used = make_batch(
        model,
        starting_media,
        batch_size=BATCH_SIZE,
        rollout_trajectories=25,
        threshold=GROW_THRESHOLD,
        timeout=60 * TIMEOUT_MIN,
        unique=USE_UNIQUE,
        direction=DIRECTION,
        used_experiments=used_experiments,
        redo_experiments=redo_experiments,
    )

    # #################### UP DIRECTION ####################
    # DIRECTION = SimDirection.UP
    # starting_media = np.zeros(20)
    # batch2, batch_used = make_batch(
    #     model,
    #     starting_media,
    #     batch_size=BATCH_SIZE,
    #     rollout_trajectories=25,
    #     threshold=GROW_THRESHOLD,
    #     timeout=60 * TIMEOUT_MIN,
    #     unique=USE_UNIQUE,
    #     direction=DIRECTION,
    #     used_experiments=batch_used,
    # )
    # batch = pd.concat((batch, batch2), ignore_index=True)
    # #######################################################

    export_to_dp_batch(new_round_folder, batch, date)
    delete_GP(model)  # clean up model


if __name__ == "__main__":
    main()

    # test = pd.read_csv("gpr_test_pred_2021-04-05T14:14:36.096451.csv", index_col=None)
    # train = pd.read_csv("gpr_train_pred_2021-04-05T14:14:36.096451.csv", index_col=None)
    # X_train, y_train = train.values[:, :-1], train.values[:, -1]
    # X_test, y_test = test.values[:, :-1], test.values[:, -1]

    # for train_size in [0.01, 0.05, 0.1, 0.25, 0.5]:
    #     print(f"TRAINING {train_size}")
    #     X_train, X_test, y_train, y_test = get_data(
    #         "L1IO-L2IO-L3O All Rands SMU UA159 Processed-Aerobic.csv",
    #         train_size=train_size,
    #         test_size=0.5,
    #     )

    #     X_trainR = robjects.r.matrix(
    #         X_train, nrow=X_train.shape[0], ncol=X_train.shape[1]
    #     )
    #     X_testR = robjects.r.matrix(X_test, nrow=X_test.shape[0], ncol=X_test.shape[1])
    #     y_trainR = robjects.r.matrix(y_train, nrow=y_train.shape[0], ncol=1)
    #     y_testR = robjects.r.matrix(y_test, nrow=y_test.shape[0], ncol=1)

    #     model = train_new_GP(X_trainR, y_trainR)

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
    #     train_data.columns = list(range(20)) + [
    #         "y_true",
    #         "y_pred_gpr",
    #         "y_pred_var_gpr",
    #     ]
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
    #     test_data.columns = list(range(20)) + ["y_true", "y_pred_gpr", "y_pred_var_gpr"]
    #     train_data.to_csv(f"GPRvNN_train_pred_{train_size:.2f}.csv", index=None)
    #     test_data.to_csv(f"GPRvNN_test_pred_{train_size:.2f}.csv", index=None)
    #     delete_GP(model)  # clean up model

    # X_trainR = robjects.r.matrix(X_train, nrow=X_train.shape[0], ncol=X_train.shape[1])
    # y_trainR = robjects.r.matrix(y_train, nrow=y_train.shape[0], ncol=1)

    # X_testR = robjects.r.matrix(X_test, nrow=X_test.shape[0], ncol=X_test.shape[1])
    # y_testR = robjects.r.matrix(y_test, nrow=y_test.shape[0], ncol=1)

    # model = train_new_GP(X_trainR, y_trainR)

    # date = datetime.datetime.now().isoformat().replace(":", ".")
    # plot_and_export_data(model, X_trainR, X_testR, X_train, X_test, y_train, y_test, date)
    # delete_GP(model)  # clean up model
