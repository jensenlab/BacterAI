import collections
import csv
from enum import Enum
import math
import os
import pickle
import time
import datetime

import cupy as cp
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
print(robjects.r("version"))

with open("gpr_lib.R", "r") as f:
    s = f.read()
    gpr_lib = STAP(s, "gpr_lib")

rpyn.activate()


class SimType(Enum):
    RANDOM = 1
    GREEDY = 2
    ROLLOUT = 3


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
    print(data)
    if "environment" in data.columns:
        data = data.drop(columns="environment")

    X, y = data[data.columns[:-1]].values, data["growth"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size, random_state=1
    )
    return X_train, X_test, y_train, y_test


@decoratortimer(2)
def train_new_GP(X_train, y_train):
    model = gpr_lib.train_new_GP(X_train, y_train)
    return model


# @decoratortimer(2)
def predict_GP(model, X):
    resultsR = gpr_lib.predict_GP(model, X)
    results = np.array(resultsR)
    return results


# @decoratortimer(2)
def sample_GP(model, X, n=1, clip=True):
    resultR = gpr_lib.sample_GP(model, X, n)
    result = np.array(resultR)
    if clip:
        result = np.clip(result, 0, 1)
    samples, var = result[:, 0], result[:, 1]
    return samples, var


# @decoratortimer(2)
def delete_GP(model):
    gpr_lib.delete_GP(model)


# @decoratortimer(2)
def rollout_trajectory(model, state, n_trajectories, threshold):
    state = cp.asarray(state)
    trajectory_states = cp.tile(state, (n_trajectories, 1))
    rewards = cp.zeros((n_trajectories,))

    reward_idx = 0
    step = 0
    # Random walk to remove 'n_trajectories' ingredients
    while trajectory_states.size > 0:
        choices = cp.argwhere(trajectory_states == 1)
        if choices.size == 0:
            break

        s0 = cp.r_[
            0,
            cp.flatnonzero(choices[1:, 0] > choices[:-1, 0]) + 1,
            choices.shape[0],
        ]

        for i in range(s0.shape[0] - 1):
            row = choices[s0[i], 0]
            idxes = choices[s0[i] : s0[i + 1], 1]
            cp.random.shuffle(idxes)
            chosen_action = idxes[0]
            trajectory_states[row, chosen_action] = 0

        trajectory_states = cp.asnumpy(trajectory_states)
        grow_results, _ = sample_GP(model, trajectory_states)

        idx_dels = list()
        for i, r in enumerate(grow_results):
            if r <= threshold:
                idx_dels.append(i)
                rewards[reward_idx] = step
                reward_idx += 1
                continue

        trajectory_states = np.delete(trajectory_states, idx_dels, axis=0)
        trajectory_states = cp.asarray(trajectory_states)
        step += 1

    return rewards.mean()


@decoratortimer(2)
def rollout_simulations(
    model,
    state,
    n,
    threshold,
    n_trajectories=1,
    depth=1,
    unique=False,
    batch_set=None,
    timeout=None,
):
    state = cp.asarray(state)
    if batch_set == None:
        batch_set = set()
    batch = cp.zeros((n, state.size), dtype=int)
    terminating_growth = cp.zeros(n, dtype=cp.float64)
    variances = cp.zeros(n, dtype=cp.float64)

    desc = "Performing Rollout Sims"
    t = tqdm(total=n, desc=desc)
    batch_n = 0
    not_timed_out = True
    start_time = time.time()
    loops = 1
    while batch_n < n and not_timed_out:
        t.desc = f"{desc} ({loops} its)"
        trajectory_state = state.copy()

        prev_result = 0
        prev_var = 0
        while trajectory_state.sum() > 0:
            choices = cp.argwhere(trajectory_state == 1)[:, 0]
            # print(f"CHOICES: {choices}")
            if choices.size == 0:
                continue

            rollout_states = cp.tile(trajectory_state, (choices.size, 1))
            results = cp.zeros(choices.size)
            for i in range(choices.size):
                rollout_states[i, choices[i]] = 0
                mean_removed = rollout_trajectory(
                    model, rollout_states[i], n_trajectories, threshold
                )
                results[i] = mean_removed

            best_action = cp.argsort(results)[-1]  # Pick highest predicted growth

            new_state = cp.asnumpy(rollout_states[best_action].reshape((1, -1)))
            result, result_var = sample_GP(model, new_state)
            result = cp.asarray(result)
            result_var = cp.asarray(result_var)

            # print(
            #     f"Best actions, remove idx: {best_action}, grow result: {round(result[0], 4)}"
            # )
            # print(results)
            if result[0] < threshold:
                # print("NO GROWS")
                # Add previous media state before it failed to grow
                key = trajectory_state.tobytes()
                if key not in batch_set or not unique:
                    batch[batch_n] = trajectory_state
                    batch_set.add(key)
                    terminating_growth[batch_n] = prev_result
                    variances[batch_n] = prev_var
                    batch_n += 1
                    t.update()
                break
            else:
                trajectory_state = rollout_states[best_action]
                prev_result = float(result[0])
                prev_var = float(result_var[0])

                # print(f"Best actions, remove idx: {chosen_action}")
                # print(f"New State: {trajectory_state}")
        if timeout is not None:
            not_timed_out = (time.time() - start_time) <= timeout
        loops += 1

    t.close()
    batch = pd.DataFrame(batch)
    batch["type"] = "ROLLOUT"
    batch["growth_pred"] = terminating_growth
    batch["var"] = variances
    return batch, batch_set


@decoratortimer(2)
def perform_simulations(
    model,
    state,
    n,
    threshold,
    sim_type=SimType.RANDOM,
    depth=1,
    unique=False,
    batch_set=None,
    timeout=None,
):
    if batch_set == None:
        batch_set = set()
    batch = cp.zeros((n, state.size), dtype=int)
    terminating_growth = cp.zeros(n, dtype=cp.float64)
    variances = cp.zeros(n, dtype=cp.float64)

    if sim_type == SimType.RANDOM:
        desc = "Performing Random Sims"
    elif sim_type == SimType.GREEDY:
        desc = "Performing Greedy Sims"

    t = tqdm(total=n, desc=desc)
    batch_n = 0
    not_timed_out = True
    start_time = time.time()
    loops = 1
    while batch_n < n and not_timed_out:
        t.desc = f"{desc} ({loops} its)"
        trajectory_state = state.copy()

        prev_result = 0
        prev_var = 0
        while trajectory_state.sum() > 0:
            choices = cp.argwhere(trajectory_state == 1)[:, 0]
            # print(f"CHOICES: {choices}")
            if choices.size == 0:
                continue

            if sim_type == SimType.RANDOM:
                test_state = trajectory_state.copy()
                # chosen_action = cp.random.choice(choices, 1, False)
                cp.random.shuffle(choices)
                chosen_action = choices[0]
                test_state[chosen_action] = 0
                test_states = test_state.reshape((1, -1))

            elif sim_type == SimType.GREEDY:
                test_states = cp.tile(trajectory_state, (choices.size, 1))
                for i in range(choices.size):
                    test_states[i, choices[i]] = 0

            # print(test_states)
            test_states = cp.asnumpy(test_states)
            results, results_vars = sample_GP(model, test_states)
            results = cp.asarray(results)
            results_vars = cp.asarray(results_vars)
            # print(results)
            if (results >= threshold).sum() == 0:
                # Add previous media state before it failed to grow
                key = trajectory_state.tobytes()
                if key not in batch_set or not unique:
                    batch[batch_n] = trajectory_state
                    batch_set.add(key)
                    terminating_growth[batch_n] = prev_result
                    variances[batch_n] = prev_var
                    batch_n += 1
                    t.update()
                break
            else:
                best_action = cp.argsort(results)[-1]  # Pick highest predicted growth
                if sim_type == SimType.GREEDY:
                    chosen_action = choices[best_action]

                trajectory_state[chosen_action] = 0
                prev_result = float(results[best_action])
                prev_var = float(results_vars[best_action])

                # print(f"Best actions, remove idx: {chosen_action}")
                # print(f"New State: {trajectory_state}")
        if timeout is not None:
            not_timed_out = (time.time() - start_time) <= timeout
        loops += 1

    t.close()
    batch = pd.DataFrame(batch)
    batch["type"] = sim_type.name
    batch["growth_pred"] = terminating_growth
    batch["var"] = variances
    return batch, batch_set


def export_to_dp_batch(parent_path, batch, date):
    batch.columns = AA_NAMES + list(batch.columns[-3:])
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
    unique=False,
    used_experiments=None,
):
    n_greedy = batch_size // 3
    n_rollout = batch_size // 3

    rollout_batch, batch_set = rollout_simulations(
        model,
        media,
        n_rollout,
        threshold,
        unique=True,
        n_trajectories=rollout_trajectories,
        timeout=60 * 120,
        batch_set=used_experiments,
    )

    greedy_batch, batch_set = perform_simulations(
        model,
        media,
        n_greedy,
        threshold,
        SimType.GREEDY,
        unique=unique,
        batch_set=batch_set,
        timeout=60 * 120,
    )

    n_random = batch_size - greedy_batch.shape[0] - rollout_batch.shape[0]
    rand_batch, _ = perform_simulations(
        model,
        media,
        n_random,
        threshold,
        SimType.RANDOM,
        unique=unique,
        batch_set=batch_set,
    )
    batch = pd.concat([rollout_batch, greedy_batch, rand_batch])
    return batch


def plot_and_export_data(
    model, X_trainR, X_testR, X_train, X_test, y_train, y_test, date
):
    y_train_pred, y_train_var = sample_GP(model, X_trainR)
    y_test_pred, y_test_var = sample_GP(model, X_testR)

    train_data = pd.DataFrame(
        np.hstack(
            (
                X_train,
                y_train.reshape((-1, 1)),
                y_train_pred.reshape((-1, 1)),
                y_train_var.reshape((-1, 1)),
            )
        )
    )
    train_data.columns = list(range(20)) + ["y_true", "y_pred", "y_pred_var"]
    test_data = pd.DataFrame(
        np.hstack(
            (
                X_test,
                y_test.reshape((-1, 1)),
                y_test_pred.reshape((-1, 1)),
                y_test_var.reshape((-1, 1)),
            )
        )
    )
    test_data.columns = list(range(20)) + ["y_true", "y_pred", "y_pred_var"]
    train_data.to_csv(f"gpr_train_pred_{date}.csv", index=None)
    test_data.to_csv(f"gpr_test_pred_{date}.csv", index=None)

    fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex=False, sharey=False, figsize=(6, 10)
    )

    test_mse = mean_squared_error(y_test_pred, y_test)
    sort_order = np.argsort(y_test)
    print("Test data MSE:", test_mse)
    print(y_test_pred.shape)
    print(y_test.shape)
    axs[0].scatter(range(len(y_test_pred)), y_test_pred[sort_order], s=3, alpha=0.50)
    axs[0].plot(range(len(y_test)), y_test[sort_order], "-r")
    axs[0].set_title(f"Test Set Predictions (MSE={round(test_mse, 3)})")
    axs[0].set_xlabel("True")
    axs[0].set_ylabel("Prediction")

    train_mse = mean_squared_error(y_train_pred, y_train)
    sort_order = np.argsort(y_train)
    print("Train data MSE:", train_mse)
    print(y_train_pred.shape)
    print(y_train.shape)
    axs[1].scatter(range(len(y_train_pred)), y_train_pred[sort_order], s=3, alpha=0.50)
    axs[1].plot(range(len(y_train)), y_train[sort_order], "-r")
    axs[1].set_title(f"Train Set Predictions (MSE={round(train_mse, 3)})")
    axs[1].set_xlabel("True")
    axs[1].set_ylabel("Prediction")

    plt.tight_layout()
    plt.savefig(f"result_gpr.png")


def process_results(mapped_path, batch_path, dataset_path, new_dataset_path):
    data, plate_controls, plate_blanks = utils.process_mapped_data(mapped_path)
    batch_df = pd.read_csv(batch_path, index_col=None)
    batch_w_results = pd.merge(
        batch_df, data, how="left", left_on=AA_NAMES, right_on=AA_NAMES, sort=True
    )
    print("batch_w_results:", batch_w_results.shape)
    batch_w_results.iloc[:, :20] = batch_w_results.iloc[:, :20].astype(int)
    out_base = batch_path.split(".csv")[0]
    out_path = out_base + "_results.csv"
    batch_w_results["depth"] = 20 - batch_w_results.iloc[:, :20].sum(axis=1)
    batch_w_results.to_csv(out_path, index=None)

    cols = list(batch_w_results.columns[:20]) + ["fitness", "growth_pred", "var"]
    cols_new = list(range(20)) + ["y_true", "y_pred", "y_pred_var"]
    if dataset_path == None:
        new_dataset = pd.DataFrame(
            batch_w_results.loc[:, cols].values,
            columns=cols_new,
        )
    else:
        dataset = pd.read_csv(dataset_path, index_col=None)
        data_batch = batch_w_results.loc[:, cols]
        data_batch.iloc[:, :20] = data_batch.iloc[:, :20].astype(int)
        dataset.columns = data_batch.columns = cols_new
        new_dataset = pd.concat([dataset, data_batch], ignore_index=True)

    new_dataset.to_csv(new_dataset_path, index=None)
    X_train = new_dataset.values[:, :20]
    y_train = new_dataset.loc[:, "y_true"].values
    used_experiments = set([i.tobytes() for i in batch_df.values[:, :20]])

    order = np.argsort(batch_w_results["growth_pred"].values)
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=False,
        sharey=False,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1.25, 2]},
    )
    x = np.arange(len(batch_w_results))
    batch_w_results = batch_w_results.loc[order, :]
    greedy = batch_w_results[batch_w_results["type"] == "GREEDY"]
    rollout = batch_w_results[batch_w_results["type"] == "ROLLOUT"]
    random = batch_w_results[batch_w_results["type"] == "RANDOM"]
    colors = ["orange", "magenta", "blue"]
    for data, color in zip([greedy, rollout, random], colors):
        axs[0].plot(
            data.index, data["fitness"], ".", color=color, markersize=3, alpha=0.75
        )
    axs[0].plot(x, batch_w_results["growth_pred"], "-", color="black")
    # axs[0].fill_between(
    #     x,
    #     batch_w_results["growth_pred"] - batch_w_results["var"],
    #     batch_w_results["growth_pred"] + batch_w_results["var"],
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
            "GPR Prediction",
        ]
    )
    axs[0].set_title("GPR Pred Results")
    # for i, row in batch_w_results.iterrows():
    #     axs[0].annotate(f"{row['depth']}", (i, row["fitness"]))

    threshold = 0.25
    counts_greedy_g = collections.Counter(
        list(greedy[greedy["fitness"] >= threshold]["depth"])
    )
    counts_greedy_ng = collections.Counter(
        list(greedy[greedy["fitness"] < threshold]["depth"])
    )
    counts_rollout_g = collections.Counter(
        list(rollout[rollout["fitness"] >= threshold]["depth"])
    )
    counts_rollout_ng = collections.Counter(
        list(rollout[rollout["fitness"] < threshold]["depth"])
    )
    counts_random_g = collections.Counter(
        list(random[random["fitness"] >= threshold]["depth"])
    )
    counts_random_ng = collections.Counter(
        list(random[random["fitness"] < threshold]["depth"])
    )

    width = 0.25
    for i, ((data_g, data_ng), color) in enumerate(
        zip(
            [
                (counts_greedy_g, counts_greedy_ng),
                (counts_rollout_g, counts_rollout_ng),
                (counts_random_g, counts_random_ng),
            ],
            colors,
        )
    ):
        bottom = []
        for key in data_ng.keys():
            if key in data_g:
                bottom.append(data_g[key])
            else:
                bottom.append(0)
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

    plt.tight_layout()
    plt.savefig(out_base + "_results.png", dpi=400)

    grow_results = batch_w_results[batch_w_results["fitness"] >= threshold].sort_values(
        "depth", ascending=False
    )
    grow_results.to_csv(out_base + "_grow_results.csv", index=False)
    top_10 = grow_results.iloc[:10, :]

    ingredients = list(top_10.columns[:20])
    print("Results (Top 10):\n", top_10)

    print("Media:")
    for idx, row in top_10.iterrows():
        leave_ins = row[:20]
        leave_ins = leave_ins[leave_ins == 1]
        leave_ins = list(leave_ins.index)
        print(leave_ins)
    return X_train, y_train, used_experiments


FIRST_BATCH = False
if __name__ == "__main__":
    # test = pd.read_csv("gpr_test_pred_2021-04-05T14:14:36.096451.csv", index_col=None)
    # train = pd.read_csv("gpr_train_pred_2021-04-05T14:14:36.096451.csv", index_col=None)
    # X_train, y_train = train.values[:, :-1], train.values[:, -1]
    # X_test, y_test = test.values[:, :-1], test.values[:, -1]

    expt_folder = "experiments/04-07-2021 test copy/"
    new_round_N = 2
    new_round_folder = os.path.join(expt_folder, f"Round{new_round_N}")
    if not os.path.exists(new_round_folder):
        os.makedirs(new_round_folder)
    if not FIRST_BATCH:
        X_train, y_train, used_experiments = process_results(
            mapped_path=os.path.join(
                expt_folder,
                f"Round{new_round_N-1}",
                "GPR SMU UA159 (2) 461c mapped_data.csv",
            ),
            batch_path=os.path.join(
                expt_folder,
                f"Round{new_round_N-1}",
                "batch_gpr_meta_2021-04-07T13:44:05.346122.csv",
            ),
            dataset_path=None,
            new_dataset_path=os.path.join(
                expt_folder, f"Round{new_round_N}", "gpr_train_pred_2021-04-07.csv"
            ),
        )

    else:
        n_examples = 1000
        X_train = np.random.rand(n_examples, 20)
        X_train[X_train >= 0.5] = 1
        X_train[X_train < 0.5] = 0
        print(X_train)
        X_test = X_train
        y_train = y_test = np.random.rand(n_examples, 1)
        print(y_train)
        used_experiments = None

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
    #     robjects.r.assign("X_train", X_trainR)
    #     robjects.r.assign("X_test", X_testR)
    #     robjects.r.assign("y_train", y_trainR)
    #     robjects.r.assign("y_test", y_testR)

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

    X_trainR = robjects.r.matrix(X_train, nrow=X_train.shape[0], ncol=X_train.shape[1])
    y_trainR = robjects.r.matrix(y_train, nrow=y_train.shape[0], ncol=1)
    robjects.r.assign("X_train", X_trainR)
    robjects.r.assign("y_train", y_trainR)

    # X_testR = robjects.r.matrix(X_test, nrow=X_test.shape[0], ncol=X_test.shape[1])
    # y_testR = robjects.r.matrix(y_test, nrow=y_test.shape[0], ncol=1)
    # robjects.r.assign("X_test", X_testR)
    # robjects.r.assign("y_test", y_testR)

    model = train_new_GP(X_trainR, y_trainR)

    date = datetime.datetime.now().isoformat().replace(":", ".")
    starting_media = np.ones(20)
    batch = make_batch(
        model,
        starting_media,
        batch_size=448,
        rollout_trajectories=25,
        threshold=0.25,
        unique=True,
        used_experiments=used_experiments,
    )
    export_to_dp_batch(new_round_folder, batch, date)
    # plot_and_export_data(model, X_trainR, X_testR, X_train, X_test, y_train, y_test, date)
    delete_GP(model)  # clean up model
