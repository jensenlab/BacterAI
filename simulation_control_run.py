import argparse
import datetime
import json
import os

from matplotlib.lines import Line2D
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


def _get_acc(a, b, threshold):
    a = a.copy()
    b = b.copy()
    a[a >= threshold] = 1
    a[a < threshold] = 0
    b[b >= threshold] = 1
    b[b < threshold] = 0
    acc = (a == b).sum() / a.shape[0]
    return acc


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

    max_batches += 1  # to account for round 1 training set, we'll use first batch
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
            n_bags=bag_quantity,
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
    for i in range(1, n_rounds):
        batch = batches[i]
        X_test = batch.iloc[:, :20].to_numpy()
        y_true = batch["growth"].to_numpy()

        # use prev round's trained model to pred this batch
        preds = trained_models[i - 1].evaluate(X_test)[0]
        test_results.append((y_true, preds))

    train_results = []
    for i in range(1, n_rounds):
        X_train = np.vstack([b.iloc[:, :20].to_numpy() for b in batches[: i + 1]])
        y_true = np.hstack([b["growth"].to_numpy() for b in batches[: i + 1]])

        # get batch data on current round and all prev rounds as train data
        preds = trained_models[i].evaluate(X_train)[0]
        train_results.append((y_true, preds))

    return test_results, train_results


def test_train_plot(results, name=None):
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
    if name is not None:
        fig.savefig(f"summarize_nn_performance_simulation_{name}.png", dpi=400)
    else:
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


def plot_summary_with_replicates(n_replicates):
    real_data = load_experiment_batch_data("experiments/05-31-2021_7")
    # real_data = load_experiment_batch_data("experiments/07-26-2021_10")[:8]

    all_test_accs = []
    all_test_mses = []
    all_train_accs = []
    all_train_mses = []
    for r in range(n_replicates):
        test_data = pd.read_csv(
            f"simulation_nn_test_replicate_{r}_SMU.csv", index_col=None
        )
        train_data = pd.read_csv(
            f"simulation_nn_train_replicate_{r}_SMU.csv", index_col=None
        )

        test_accs = []
        test_mses = []
        train_accs = []
        train_mses = []
        for i in range(test_data.shape[1] // 2):
            true = test_data.loc[:, f"test_true_R{i+1}"].to_numpy()
            pred = test_data.loc[:, f"test_pred_R{i+1}"].to_numpy()
            true = true[~np.isnan(true)]
            pred = pred[~np.isnan(pred)]

            acc = _get_acc(true, pred, 0.25)
            mse = mean_squared_error(true, pred)
            test_accs.append(acc)
            test_mses.append(mse)

            true = train_data.loc[:, f"train_true_R{i+1}"].to_numpy()
            pred = train_data.loc[:, f"train_pred_R{i+1}"].to_numpy()
            true = true[~np.isnan(true)]
            pred = pred[~np.isnan(pred)]

            acc = _get_acc(true, pred, 0.25)
            mse = mean_squared_error(true, pred)
            train_accs.append(acc)
            train_mses.append(mse)

        all_test_accs.append(test_accs)
        all_test_mses.append(test_mses)
        all_train_accs.append(train_accs)
        all_train_mses.append(train_mses)

    all_test_accs = np.array(all_test_accs)
    all_test_mses = np.array(all_test_mses)
    all_train_accs = np.array(all_train_accs)
    all_train_mses = np.array(all_train_mses)

    test_acc_error = np.std(all_test_accs, axis=0, ddof=1)
    test_mse_error = np.std(all_test_mses, axis=0, ddof=1)
    train_acc_error = np.std(all_train_accs, axis=0, ddof=1)
    train_mse_error = np.std(all_train_mses, axis=0, ddof=1)

    test_acc_mean = np.mean(all_test_accs, axis=0)
    test_mse_mean = np.mean(all_test_mses, axis=0)
    train_acc_mean = np.mean(all_train_accs, axis=0)
    train_mse_mean = np.mean(all_train_mses, axis=0)

    real_accs = []
    real_mses = []
    for df in real_data:
        true = df["fitness"].to_numpy()
        pred = df["growth_pred"].to_numpy()
        true = true[~np.isnan(true)]
        pred = pred[~np.isnan(pred)]

        acc = _get_acc(true, pred, 0.25)
        mse = mean_squared_error(true, pred)
        real_accs.append(acc)
        real_mses.append(mse)

    x = np.arange(len(test_acc_error)) + 1

    fig, axs = plt.subplots(
        nrows=1, ncols=2, sharex=False, sharey=False, figsize=(6, 3)
    )

    axs[0].errorbar(x, test_acc_mean, yerr=test_acc_error, color="r", label="NN test")
    axs[0].errorbar(
        x, train_acc_mean, yerr=train_acc_error, color="b", label="NN train"
    )
    axs[0].plot(x, real_accs, "k.-", label="experiment")
    # axs[0].legend()
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Round")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(x)

    axs[1].errorbar(x, test_mse_mean, yerr=test_mse_error, color="r", label="NN test")
    axs[1].errorbar(
        x, train_mse_mean, yerr=train_mse_error, color="b", label="NN train"
    )
    axs[1].plot(x, real_mses, "k.-", label="experiment")
    axs[1].set_ylabel("MSE")
    axs[1].set_xlabel("Round")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(x)

    legend_elements = [
        Line2D([0], [0], markersize=0, linewidth=3, color="red", label="NN test"),
        Line2D([0], [0], markersize=0, linewidth=3, color="blue", label="NN train"),
        Line2D(
            [0],
            [0],
            marker=".",
            markersize=5,
            linewidth=0,
            color="k",
            label="Experiment",
        ),
    ]
    axs[0].legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(1.1, -0.25),
        ncol=3,
    )
    plt.subplots_adjust(
        left=0.1, right=0.98, top=0.9, bottom=0.3, wspace=0.35, hspace=0.1
    )

    # fig.tight_layout()
    fig.savefig("simulation_control_summary.png", dpi=400)


def main_with_replicates(args):
    with open(args.path) as f:
        config = json.load(f)

    EXPT_FOLDER = config["experiment_path"]
    DATA_PATH = config["data_path"]
    BATCH_SIZE = config["batch_size"]
    N_BAGS = config["n_bags"]

    n_replicates = 6

    for replicate_n in range(n_replicates):
        folder = os.path.join(EXPT_FOLDER, f"replicate_{replicate_n}")
        if not os.path.exists(folder):
            os.makedirs(folder)

        all_test_data = []
        all_train_data = []

        NP_RAND_STATE = utils.seed_numpy_state(replicate_n)
        batches = load_data_and_batch(
            DATA_PATH, BATCH_SIZE, max_batches=8, exclude_LOs=True
        )
        trained_models = train_on_batches(batches, N_BAGS, folder)
        test_results, train_results = get_test_train_data(folder, batches)

        for y_true_test, preds_test in test_results:
            print(y_true_test.shape, preds_test.shape)
            all_test_data += [y_true_test.tolist(), preds_test.tolist()]

        for y_true_train, preds_train in train_results:
            all_train_data += [y_true_train.tolist(), preds_train.tolist()]

        column_names = []
        for i in range(len(all_test_data) // 2):
            column_names += [f"test_true_R{i+1}", f"test_pred_R{i+1}"]

        test_df = pd.DataFrame(
            all_test_data,
        )
        test_df = test_df.transpose()
        test_df.columns = column_names

        test_df.to_csv(
            f"simulation_nn_test_replicate_{replicate_n}_SMU.csv", index=False
        )

        column_names = []
        for i in range(len(all_train_data) // 2):
            column_names += [f"train_true_R{i+1}", f"train_pred_R{i+1}"]
        train_df = pd.DataFrame(
            all_train_data,
        )
        train_df = train_df.transpose()
        train_df.columns = column_names
        train_df.to_csv(
            f"simulation_nn_train_replicate_{replicate_n}_SMU.csv", index=False
        )

    plot_summary_with_replicates(n_replicates)


def main(args):
    with open(args.path) as f:
        config = json.load(f)

    EXPT_FOLDER = config["experiment_path"]
    DATA_PATH = config["data_path"]
    BATCH_SIZE = config["batch_size"]
    N_BAGS = config["n_bags"]

    batches = load_data_and_batch(DATA_PATH, BATCH_SIZE, exclude_LOs=True)
    # trained_models = train_on_batches(batches, N_BAGS, EXPT_FOLDER)
    test_results, train_results = get_test_train_data(EXPT_FOLDER, batches)
    results = list(zip(test_results, train_results))
    # test_train_plot(results)

    batch_fitnesses_1 = [d["growth"].to_numpy().flatten() for d in batches]

    print(len(batch_fitnesses_1))
    results = load_experiment_batch_data("experiments/05-31-2021_7")
    batch_fitnesses_2 = [
        d["fitness"].to_numpy().flatten() for d in results[: len(batch_fitnesses_1)]
    ]

    fitness_order_plot(batch_1=batch_fitnesses_1, batch_2=batch_fitnesses_2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BacterAI Control Simulation")

    parser.add_argument(
        "path",
        type=str,
        help="The path to the configuration file (.json)",
    )

    args = parser.parse_args()

    # main(args)
    main_with_replicates(args)
