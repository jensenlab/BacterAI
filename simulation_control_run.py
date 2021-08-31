import argparse
import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import models
import net
import run
import utils

SEED = 0
NP_RAND_STATE = utils.seed_numpy_state(SEED)


def chunks(a, n):
    """ Yield successive n-sized chunks from a."""

    for i in range(0, len(a), n):
        yield a[i : i + n]


def load_data_and_batch(filepath, batch_size, max_batches=8, exclude_LOs=False):
    data = pd.read_csv(filepath, index_col=None)
    if exclude_LOs:
        print(data.shape)
        data["sum"] = data.iloc[:, :20].sum(axis=1)

        data = data[(data["sum"] > 2) & (data["sum"] < 18)]
        data = data.drop(columns="sum")
        print(data.shape)

    if "environment" in data.columns:
        data = data.drop(columns="environment")

    indexes = np.array(data.index)
    NP_RAND_STATE.shuffle(indexes)

    if len(indexes) < batch_size:
        raise Exception(
            f"You must provide more than {batch_size} (size of batch) data points."
        )

    batch_indexes = list(chunks(indexes, batch_size))

    if len(indexes) % batch_size != 0:
        batch_indexes = batch_indexes[:-1]

    batch_indexes = batch_indexes[:max_batches]
    batches = [data.loc[idx, :] for idx in batch_indexes]
    return batches


def load_experiment_batch_data(experiment_path):
    paths = []
    for root, dirs, files in os.walk(experiment_path):
        models = []
        for name in files:
            path = os.path.join(root, name)
            if "bad_runs" in path:
                continue
            if "results_all" in name:
                paths.append(path)

    paths = sorted(paths, key=lambda x: (len(x), x))
    print(paths)
    all_results = []
    for path in paths:
        results = utils.normalize_ingredient_names(pd.read_csv(path, index_col=None))
        results = results[results["type"] != "REDO"]
        all_results.append(results)

    return all_results


def train_on_batches(batches, bag_quantity, folder):

    trained_models = []
    for i in range(len(batches)):
        X_train = np.vstack([b.iloc[:, :20].to_numpy() for b in batches[: i + 1]])
        y_train = np.hstack([b["growth"].to_numpy() for b in batches[: i + 1]])

        round_name = f"Round{i+1}"
        models_folder = os.path.join(folder, round_name, f"nn_models")
        model = models.NeuralNetModel(models_folder)

        model.train(
            X_train,
            y_train,
            n_bags=20,
            # n_bags=5,
            bag_proportion=1.0,
            epochs=50,
            batch_size=360,
            lr=0.001,
        )
        trained_models.append(model)

    return trained_models


def get_test_train_data(experiment_path, batches):
    nn_paths = []
    for root, dirs, files in os.walk(experiment_path):
        for name in dirs:
            path = os.path.join(root, name)
            if "nn_models" in name:
                nn_paths.append(path)

    nn_paths.sort(key=lambda x: (len(x), x))

    trained_models = []
    for path in nn_paths:
        model = models.NeuralNetModel.load_trained_models(path)
        trained_models.append(model)

    n_rounds = len(batches)

    test_results = []
    for i in range(n_rounds - 1):
        next_batch = batches[i + 1]
        X_test = next_batch.iloc[:, :20].to_numpy()
        y_true = next_batch["growth"].to_numpy()
        preds = trained_models[i].evaluate(X_test)[0]
        test_results.append((y_true, preds))
    test_results.append(())

    train_results = []
    for i in range(n_rounds):
        X_train = np.vstack([b.iloc[:, :20].to_numpy() for b in batches[: i + 1]])
        y_train = np.hstack([b["growth"].to_numpy() for b in batches[: i + 1]])
        preds = trained_models[i].evaluate(X_train)[0]
        train_results.append((y_train, preds))

    results = list(zip(test_results, train_results))
    return results


def test_train_plot(results):
    fig, axs = plt.subplots(
        nrows=2, ncols=len(results), sharex=False, sharey="row", figsize=(30, 6)
    )

    for i, (test, train) in enumerate(results):
        # Plot test data i.e. next batch predictions
        if test:
            data_1 = test[1]
            data_2 = test[0]

            x_axis_points = np.arange(len(data_1))
            # print(data)
            mse = mean_squared_error(test[0], test[1])
            order = np.argsort(data_1)
            axs[0, i].plot(
                x_axis_points,
                data_2[order],
                ".",
                alpha=1,
                markersize=1,
            )
            axs[0, i].plot(x_axis_points, data_1[order], "-")
            axs[0, i].set_xlabel("Experiment")
            axs[0, i].set_title(f"Round {i+1} NNs, Test\nMSE:{mse:.3f}")

        # Plot train data
        data_1 = train[1]
        data_2 = train[0]

        x_axis_points = np.arange(len(data_1))

        mse = mean_squared_error(train[0], train[1])
        order = np.argsort(data_1)
        axs[1, i].plot(x_axis_points, data_2[order], ".", alpha=0.20, markersize=1)
        axs[1, i].plot(x_axis_points, data_1[order], "-")
        axs[1, i].set_xlabel("Experiment")
        axs[1, i].set_title(f"Round {i+1} NNs, Train\nMSE:{mse:.3f}")

    fig.tight_layout()
    fig.savefig("summarize_nn_performance_simulation.png", dpi=400)


def fitness_order_plot(batch_1, batch_2):
    # print(f"{batch_1=}")
    # print(f"{batch_2=}")

    fig, axs = plt.subplots(
        nrows=1, ncols=len(batch_1), sharex=False, sharey="row", figsize=(15, 3)
    )

    for idx, (b1, b2) in enumerate(zip(batch_1, batch_2)):
        b1.sort()
        b1 = np.clip(b1, 0, 1)
        b2.sort()
        b2 = np.clip(b2, 0, 1)

        axs[idx].plot(np.arange(len(b1)), b1, "r.", markersize=1)
        axs[idx].plot(np.arange(len(b2)), b2, "k.", markersize=1)

        axs[idx].set_ylabel("Fitness")
        axs[idx].set_xlabel("Assay")
        axs[idx].set_title(f"Round {idx+1}")

    plt.legend(["Random", "BacterAI"])
    fig.tight_layout()
    fig.savefig("summarize_simulation_fitness_order_plot.png", dpi=400)

    fig, axs = plt.subplots(
        nrows=1, ncols=1, sharex=False, sharey="row", figsize=(4, 4)
    )

    all_b1, all_b2 = np.hstack(batch_1), np.hstack(batch_2)
    all_b1.sort()
    all_b1 = np.clip(all_b1, 0, 1)
    all_b2.sort()
    all_b2 = np.clip(all_b2, 0, 1)

    axs.plot(np.arange(len(all_b1)), all_b1, "r.", markersize=2, alpha=0.025)
    axs.plot(np.arange(len(all_b2)), all_b2, "k.", markersize=2, alpha=0.025)
    # axs.hist(
    #     all_b1,
    #     bins=50,
    #     color="r",
    #     alpha=0.5,
    # )
    # axs.hist(
    #     all_b2,
    #     bins=50,
    #     color="k",
    #     alpha=0.5,
    # )
    axs.set_ylabel("Count")
    # axs.set_xlabel("Fitness")
    # axs.set_title(f"Round {idx+1}")

    plt.legend(["Random", "BacterAI"])
    fig.tight_layout()
    fig.savefig("summarize_simulation_fitness_order_plot_combined_line.png", dpi=400)


def main(args):
    with open(args.path) as f:
        config = json.load(f)

    EXPT_FOLDER = config["experiment_path"]
    DATA_PATH = config["data_path"]
    BATCH_SIZE = config["batch_size"]
    N_BAGS = config["n_bags"]

    batches = load_data_and_batch(DATA_PATH, BATCH_SIZE, exclude_LOs=True)
    # trained_models = train_on_batches(batches, N_BAGS, EXPT_FOLDER)
    results = get_test_train_data(EXPT_FOLDER, batches)
    # test_train_plot(results)

    batch_fitnesses_1 = [d["growth"].to_numpy().flatten() for d in batches]

    print(len(batch_fitnesses_1))
    results = load_experiment_batch_data("experiments/05-31-2021_7")
    batch_fitnesses_2 = [
        d["fitness"].to_numpy().flatten() for d in results[: len(batch_fitnesses_1)]
    ]
    fitness_order_plot(batch_1=batch_fitnesses_1, batch_2=batch_fitnesses_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BacterAI Experiment Generator")

    parser.add_argument(
        "path",
        type=str,
        help="The path to the configuration file (.json)",
    )

    args = parser.parse_args()

    main(args)