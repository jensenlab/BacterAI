import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    WhiteKernel,
    EuclideanDistance,
    RBF,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import mcts


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
def get_data(path):
    data = pd.read_csv(path, index_col=None)
    if "environment" in data.columns:
        data = data.drop(columns="environment")

    X, y = data[data.columns[:-1]].values, data["growth"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.1, test_size=0.1, random_state=1
    )
    return X_train, X_test, y_train, y_test


# class ExponentialDecayEuclideanDistance(
#     StationaryKernelMixin, NormalizedKernelMixin, Kernel
# ):
#     def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
#         self.length_scale = length_scale
#         self.length_scale_bounds = length_scale_bounds

#     @property
#     def hyperparameter_length_scale(self):
#         if self.anisotropic:
#             return Hyperparameter(
#                 "length_scale",
#                 "numeric",
#                 self.length_scale_bounds,
#                 len(self.length_scale),
#             )
#         return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

#     def __call__(self, X, Y=None, eval_gradient=False):
#         exp_decay = RBF(length_scale)(X, Y)
#         e_dist = np.sqrt(np.dot(X, Y))
#         return exp_decay + e_dist


@decoratortimer(2)
def train_gpr(X_train, y_train):
    kernel = RBF()
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


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(
        "data/L1L2IO-Rand-Tempest-SMU/L1IO-L2IO-All Rands SMU UA159 Processed-Aerobic.csv"
        # "models/SMU_NN_oracle/SMU_training_data_aerobic_L1L2IO_rands.csv"
    )

    gpr, score = train_gpr(X_train, y_train)

    print("Score:", score)

    fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex=False, sharey=False, figsize=(6, 12)
    )

    mse, y_preds = test_gpr(gpr, X_train, y_train)
    print("Train data MSE:", mse)
    axs[0].scatter(y_train, y_preds)
    axs[0].set_title(f"Train Set Predictions (MSE={round(mse, 3)})")
    axs[0].set_xlabel("True")
    axs[0].set_ylabel("Prediction")

    mse, y_preds = test_gpr(gpr, X_test, y_test)
    print("Test data MSE:", mse)
    axs[1].scatter(y_test, y_preds)
    axs[1].set_title(f"Test Set Predictions (MSE={round(mse, 3)})")
    axs[1].set_xlabel("True")
    axs[1].set_ylabel("Prediction")

    plt.tight_layout()
    plt.savefig(f"gpr_large.png")
