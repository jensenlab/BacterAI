import math
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as rpyn
from rpy2.robjects.packages import STAP
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    WhiteKernel,
    RBF,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


with open("gpr_lib.R", "r") as f:
    s = f.read()
    gpr_lib = STAP(s, "gpr_lib")

rpyn.activate()
# import mcts


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


@decoratortimer(2)
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
def train_gpr(X_train, y_train):
    kernel = 0.1 * RBF([1] * X_train.shape[1]) + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X_train, y_train)
    score = gpr.score(X_train, y_train)
    return gpr, score


@decoratortimer(2)
def test_gpr(gpr, X_test, y_test):
    y_mean, y_std = gpr.predict(X_test, return_std=True)
    y_preds = []
    for mu, stdev in zip(y_mean, y_std):
        y_pred = np.random.normal(mu, stdev, 1)
        y_preds.append(y_pred)
    y_preds = np.array(y_preds)
    mse = mean_squared_error(y_test, y_preds)
    return mse, y_preds


def random_simulations(model, state, n, threshold):

    batch = np.zeros((n, state.size))
    trajectory_states = np.tile(state, (n, 1))
    n_completed = 0
    terminating_growth = np.zeros(n)
    # Random walk to remove 'horizon' ingredients
    loop = 0
    while trajectory_states.size > 0:
        print(f"Loop: {loop}")
        choices = np.argwhere(trajectory_states == 1)
        if choices.size == 0:
            # Add final remaining states if cannot remove anymore
            for i in trajectory_states:
                batch[n_completed] = trajectory_states[i]
                n_completed += 1
            break

        s0 = np.r_[
            0,
            np.flatnonzero(choices[1:, 0] > choices[:-1, 0]) + 1,
            choices.shape[0],
        ]

        new_trajectory_states = trajectory_states.copy()
        for i in range(s0.shape[0] - 1):
            new_trajectory_states[
                i, np.random.choice(choices[s0[i] : s0[i + 1], 1], 1, False)
            ] = 0

        grow_resultsR = gpr_lib.gpr_pred(model, new_trajectory_states)
        grow_results = np.array(grow_resultsR)

        idx_dels = list()
        for i, r in enumerate(grow_results):
            if r <= threshold:
                batch[n_completed] = trajectory_states[i]
                terminating_growth[n_completed] = r
                idx_dels.append(i)
                n_completed += 1
                continue

        trajectory_states = new_trajectory_states
        trajectory_states = np.delete(trajectory_states, idx_dels, axis=0)
        loop += 1
    print(batch)
    print(terminating_growth)
    return batch


def make_batch(model, media, batch_size, exploration, threshold):
    n_random = math.floor(batch_size * exploration)
    n_rollout = math.ceil(batch_size * exploration)
    print(n_random + n_rollout, n_random, n_rollout)

    rand_batch = random_simulations(model, media, n_random, threshold)
    return rand_batch


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(
        "L1IO-L2IO-L3O All Rands SMU UA159 Processed-Aerobic.csv",
        train_size=0.025,
        test_size=0.1
        # "models/SMU_NN_oracle/SMU_training_data_aerobic_L1L2IO_rands.csv"
    )

    X_trainR = ro.r.matrix(X_train, nrow=X_train.shape[0], ncol=X_train.shape[1])
    X_testR = ro.r.matrix(X_test, nrow=X_test.shape[0], ncol=X_test.shape[1])
    y_trainR = ro.r.matrix(y_train, nrow=y_train.shape[0], ncol=1)
    y_testR = ro.r.matrix(y_test, nrow=y_test.shape[0], ncol=1)
    ro.r.assign("X_train", X_trainR)
    ro.r.assign("X_test", X_testR)
    ro.r.assign("y_train", y_trainR)
    ro.r.assign("y_test", y_testR)

    # if not os.path.exists("gpr_model.pkl"):
    #     model = gpr_lib.make_gpr(X_trainR, y_trainR)
    #     pickle.dump(model, open("gpr_model.pkl", "wb"))
    # else:
    #     model = pickle.load(open("gpr_model.pkl", "rb"))
    model = gpr_lib.make_gpr(X_trainR, y_trainR)

    starting_media = np.ones(20)
    print(starting_media)
    random_batch = make_batch(model, starting_media, 500, 0.5, 0.25)

    batch_resultsR = gpr_lib.gpr_pred(model, random_batch)
    batch_results = np.array(batch_resultsR)
    print(batch_results)

    # y_test_predR = gpr_lib.gpr_pred(model, X_testR)
    # y_test_pred = np.array(y_test_predR)

    # y_train_predR = gpr_lib.gpr_pred(model, X_trainR)
    # y_train_pred = np.array(y_train_predR)

    # print(y_test_pred)
    # print(y_train_pred)

    # fig, axs = plt.subplots(
    #     nrows=2, ncols=1, sharex=False, sharey=False, figsize=(6, 10)
    # )

    # test_mse = mean_squared_error(y_test_pred, y_test)
    # sort_order = np.argsort(y_test)
    # print("Test data MSE:", test_mse)
    # print(y_test_pred.shape)
    # print(y_test.shape)
    # axs[0].scatter(range(len(y_test_pred)), y_test_pred[sort_order], s=3, alpha=0.50)
    # axs[0].plot(range(len(y_test)), y_test[sort_order], "-r")
    # axs[1].set_title(f"Test Set Predictions (MSE={round(test_mse, 3)})")
    # axs[0].set_xlabel("True")
    # axs[0].set_ylabel("Prediction")

    # train_mse = mean_squared_error(y_train_pred, y_train)
    # sort_order = np.argsort(y_train)
    # print("Train data MSE:", train_mse)
    # print(y_train_pred.shape)
    # print(y_train.shape)
    # axs[1].scatter(range(len(y_train_pred)), y_train_pred[sort_order], s=3, alpha=0.50)
    # axs[1].plot(range(len(y_train)), y_train[sort_order], "-r")
    # axs[0].set_title(f"Train Set Predictions (MSE={round(train_mse, 3)})")
    # axs[1].set_xlabel("True")
    # axs[1].set_ylabel("Prediction")

    # plt.tight_layout()
    # plt.savefig(f"gpr.png")
    # plt.show()
