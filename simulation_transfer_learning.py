import argparse
from cmath import exp
import datetime
import json
import os
from re import L, M

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import models
import net
import utils

SEED = 0
NP_RAND_STATE = utils.seed_numpy_state(SEED)


def _get_unique(data, n_ingredients):
    unique = set()
    for row in data.iloc[:, :n_ingredients].to_numpy():
        unique.add(tuple(row.tolist()))

    print(f"N unique rows: {len(unique)}")


def _get_acc(a, b, threshold):
    a = a.copy()
    b = b.copy()
    a[a >= threshold] = 1
    a[a < threshold] = 0
    b[b >= threshold] = 1
    b[b < threshold] = 0
    acc = (a == b).sum() / a.shape[0]
    return acc


def load_all_experiment_data(experiment_path):
    paths = []
    for root, _, files in os.walk(experiment_path):
        for name in files:
            path = os.path.join(root, name)
            if "bad_runs" in path:
                continue
            if "results_all" in name:
                paths.append(path)

    paths = sorted(paths, key=lambda x: (len(x), x))

    print("Loaded Data Files:")
    print(json.dumps(paths, indent=2))

    all_results = []
    for path in paths:
        results = utils.normalize_ingredient_names(pd.read_csv(path, index_col=None))
        results = results[results["type"] != "REDO"]
        all_results.append(results)

    return all_results


def batch_data(data, batch_size, max_batches=8):
    if "environment" in data.columns:
        data = data.drop(columns="environment")

    data.loc[data["fitness"] >= 1, "fitness"] = 1
    indexes = np.array(data.index)
    if SHUFFLE:
        NP_RAND_STATE.shuffle(indexes)

    if len(indexes) < batch_size:
        raise Exception(
            f"You must provide more than {batch_size} (size of batch) data points."
        )

    batch_indexes = []
    for i in range(len(data) // batch_size):
        if i >= max_batches:
            break
        batch_indexes.append(indexes[i * batch_size : (i + 1) * batch_size])

    if len(indexes) % batch_size != 0:
        batch_indexes = batch_indexes[:-1]

    max_batches += 1  # to account for round 1 training set, we'll use first batch
    batch_indexes = batch_indexes[:max_batches]
    batches = [data.iloc[indexes, :] for indexes in batch_indexes]
    return batches


def _plot_data(model, X, y, ax, growth_threshold):
    preds, _ = model.evaluate(X)
    acc = _get_acc(y, preds, growth_threshold)
    mse = mean_squared_error(y, preds)

    order = np.argsort(preds)
    ax.plot(np.arange(len(order)), y[order], "k.", markersize=1, alpha=0.25)
    ax.plot(
        np.arange(len(order)),
        preds[order],
        "-",
        color="dodgerblue",
        linewidth=0.5,
    )
    ax.text(
        0,
        1.05,
        f"Acc: {acc*100:.1f}%\nMSE: {mse:0.3f}",
        fontsize=10,
        verticalalignment="top",
    )

    return acc, mse


def main(
    experiment_folder, transfer_model_folder, growth_threshold=0.25, n_ingredients=20
):
    # Import data and create batches from all data
    experiment_data = load_all_experiment_data(experiment_folder)
    all_data = pd.concat(experiment_data, ignore_index=True)
    all_data = all_data.reset_index(drop=True)
    batches = batch_data(all_data, batch_size=336, max_batches=len(experiment_data))

    # Load transfer model
    transfer_model = models.NeuralNetModel.load_trained_models(transfer_model_folder)

    # Make new folders where models will be stored
    new_models_path = os.path.join(
        experiment_folder, "sim_trained_models", "new_models"
    )
    pre_trained_models_path = os.path.join(
        experiment_folder, "sim_trained_models", "pre_trained_models"
    )

    if not os.path.exists(new_models_path):
        os.makedirs(new_models_path)

    if not os.path.exists(pre_trained_models_path):
        os.makedirs(pre_trained_models_path)

    fig_order_plot, axs_order_plot = plt.subplots(
        nrows=4, ncols=len(batches), sharex=False, sharey="row", figsize=(20, 7)
    )
    fig_metrics, axs_metrics = plt.subplots(
        nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8, 4)
    )
    data_accum = []
    metrics = {
        "new_model_acc": [],
        "pre_trained_model_acc": [],
        "new_model_mse": [],
        "pre_trained_model_mse": [],
    }

    for batch_idx in range(len(batches)):
        if batch_idx == 0:
            rand_kickstart_path = os.path.join(
                experiment_folder, "Round1", "random_train_kickstart.csv"
            )
            data = utils.normalize_ingredient_names(
                pd.read_csv(rand_kickstart_path, index_col=None)
            )
            data.columns = list(range(n_ingredients)) + ["fitness"]

            pre_trained_model = transfer_model
        else:
            batch = batches[batch_idx - 1]

            # Create new model from pre-trained model
            pre_trained_model_path = os.path.join(
                pre_trained_models_path, f"Batch{batch_idx+1}"
            )
            pre_trained_model = models.NeuralNetModel(pre_trained_model_path)

            # Train models on all previous + current days' batch data
            data_accum.append(batch)
            data = pd.concat(data_accum, ignore_index=True)

        # Create new model
        new_model_path = os.path.join(new_models_path, f"Batch{batch_idx+1}")
        new_model = models.NeuralNetModel(new_model_path)

        X_train = data.iloc[:, :n_ingredients].to_numpy()
        y_train = data["fitness"].to_numpy()
        training_scheme = dict(
            n_bags=25,
            bag_proportion=1.0,
            epochs=50,
            batch_size=360,
            lr=0.001,
        )

        new_model.train(X_train, y_train, **training_scheme)

        if batch_idx != 0:
            training_scheme["transfer_models"] = transfer_model.models
            pre_trained_model.train(X_train, y_train, **training_scheme)

        # Evaulate both models' on n+1 batch data
        next_batch = batches[batch_idx]
        next_batch_X = next_batch.iloc[:, :n_ingredients].to_numpy()
        next_batch_y = next_batch["fitness"].to_numpy()

        _plot_data(
            new_model,
            X_train,
            y_train,
            axs_order_plot[0, batch_idx],
            growth_threshold,
        )

        new_model_acc, new_model_mse = _plot_data(
            new_model,
            next_batch_X,
            next_batch_y,
            axs_order_plot[1, batch_idx],
            growth_threshold,
        )

        _plot_data(
            pre_trained_model,
            X_train,
            y_train,
            axs_order_plot[2, batch_idx],
            growth_threshold,
        )

        pre_trained_model_acc, pre_trained_model_mse = _plot_data(
            pre_trained_model,
            next_batch_X,
            next_batch_y,
            axs_order_plot[3, batch_idx],
            growth_threshold,
        )

        # Metrics
        metrics["new_model_acc"].append(new_model_acc)
        metrics["pre_trained_model_acc"].append(pre_trained_model_acc)
        metrics["new_model_mse"].append(new_model_mse)
        metrics["pre_trained_model_mse"].append(pre_trained_model_mse)

        print(f"-- Batch {batch_idx} -- ")
        print(
            f"   Accuracy - New: {new_model_acc:.4f}, Pre-trained: {pre_trained_model_acc:.4f}"
        )
        print(
            f"   MSE      - New: {new_model_mse:.4f}, Pre-trained: {pre_trained_model_mse:.4f}"
        )
        print()

        axs_order_plot[0, 0].set_ylabel("Fitness\nTrain (New)")
        axs_order_plot[1, 0].set_ylabel("Fitness\nTest (New)")
        axs_order_plot[2, 0].set_ylabel("Fitness\nTrain (Pre-Trained)")
        axs_order_plot[3, 0].set_ylabel("Fitness\nTest (Pre-Trained)")
        fig_order_plot.tight_layout()

        out_path = (
            os.path.join(experiment_folder, "order_plot_shuffled.png")
            if SHUFFLE
            else os.path.join(experiment_folder, "order_plot_not_shuffled.png")
        )
        fig_order_plot.savefig(out_path, dpi=400)

    axs_metrics[0].plot(
        np.arange(len(batches)), metrics["new_model_acc"], label="new_model_acc"
    )
    axs_metrics[0].plot(
        np.arange(len(batches)),
        metrics["pre_trained_model_acc"],
        label="pre_trained_model_acc",
    )
    axs_metrics[1].plot(
        np.arange(len(batches)), metrics["new_model_mse"], label="new_model_mse"
    )
    axs_metrics[1].plot(
        np.arange(len(batches)),
        metrics["pre_trained_model_mse"],
        label="pre_trained_model_mse",
    )
    axs_metrics[0].set_ylabel("Accuracy")
    axs_metrics[1].set_ylabel("MSE")
    fig_metrics.legend()
    fig_metrics.tight_layout()

    out_path = (
        os.path.join(experiment_folder, "metrics_plot_shuffled.png")
        if SHUFFLE
        else os.path.join(experiment_folder, "metrics_plot_not_shuffled.png")
    )
    fig_metrics.savefig(out_path, dpi=400)

    _get_unique(data, n_ingredients)


SHUFFLE = False

if __name__ == "__main__":
    # experiment_folder = "experiments/08-20-2021_12_TL_sim"
    # transfer_model_folder = "experiments/08-20-2021_12_TL_sim/transfer_models"
    experiment_folder = "experiments/07-26-2021_10_TL_sim"
    transfer_model_folder = "experiments/07-26-2021_10_TL_sim/transfer_models"

    main(experiment_folder, transfer_model_folder)