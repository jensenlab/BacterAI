import time

import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
# import rpy2.rinterface as ri
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri as rpyn

# from rpy2.robjects.packages import importr
# import rpy2.robjects
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    WhiteKernel,
    RBF,
)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from rpy2.robjects.packages import STAP
with open('gpr_lib.R', 'r') as f:
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


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_data(
        "L1IO-L2IO-L3O All Rands SMU UA159 Processed-Aerobic.csv",
        train_size=0.25, test_size=0.1
        # "models/SMU_NN_oracle/SMU_training_data_aerobic_L1L2IO_rands.csv"
    )

    # X_train, X_test, y_train, y_test = X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()

    X_trainR = ro.r.matrix(
        X_train, nrow=X_train.shape[0], ncol=X_train.shape[1])
    X_testR = ro.r.matrix(X_test, nrow=X_test.shape[0], ncol=X_test.shape[1])
    y_trainR = ro.r.matrix(y_train, nrow=y_train.shape[0], ncol=1)
    y_testR = ro.r.matrix(y_test, nrow=y_test.shape[0], ncol=1)
    ro.r.assign("X_train", X_trainR)
    ro.r.assign("X_test", X_testR)
    ro.r.assign("y_train", y_trainR)
    ro.r.assign("y_test", y_testR)

    # r = rpy2.robjects.r
    # gprR = r['source']('gpr_lib.R')
    # print(gprR)
    model = gpr_lib.make_gpr(X_trainR, y_trainR)
    y_predR = gpr_lib.gpr_pred(model, X_testR)

    y_preds = np.array(y_predR)
    print(y_preds)
    # eps = math.sqrt(2.220446e-16)
    # gpi = laGP.newGPsep(X_train, y_train, d=0.1, g=0.1*stats.var(y_train), dK=TRUE)
    # ndim = dim(X)[[2]]
    # tmin = rep(eps, ndim+1)
    # tmax = c(rep(100, ndim), var(y))
    # mle = laGP.mleGPsep(gpi, para="both", tmin=tmin, tmax=tmax, verb=2)

    # gpr, score = train_gpr(X_train, y_train)
    # print("Score:", score)

    fig, axs = plt.subplots(
        nrows=2, ncols=1, sharex=False, sharey=False, figsize=(6, 12)
    )
    sort_order = np.argsort(y_test)
    # mse, y_preds = test_gpr(gpr, X_train, y_train)
    # print("Train data MSE:", mse)
    print(y_test.shape)
    print(y_preds.shape)
    axs[0].scatter(range(len(y_preds)), y_preds[sort_order], s=3, alpha=0.50)
    axs[0].plot(range(len(y_test)), y_test[sort_order], "-r")
    axs[0].set_title(f"Train Set Predictions")  # (MSE={round(mse, 3)})")
    axs[0].set_xlabel("True")
    axs[0].set_ylabel("Prediction")

    # sort_order = np.argsort(y_test)
    # mse, y_preds = test_gpr(gpr, X_test, y_test)
    # print("Test data MSE:", mse)
    # axs[1].scatter(range(len(y_preds)), y_preds[sort_order], s=3, alpha=0.50)
    # axs[1].plot(range(len(y_preds)), y_test[sort_order], "-r")
    # axs[1].set_title(f"Test Set Predictions (MSE={round(mse, 3)})")
    # axs[1].set_xlabel("True")
    # axs[1].set_ylabel("Prediction")

    plt.tight_layout()
    plt.savefig(f"gpr.png")
    plt.show()
