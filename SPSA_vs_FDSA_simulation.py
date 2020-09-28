"""
# GOAL

- [] determine the optimal number of SPSA gradients to avg over, e.g. find the least number of 
samples we need for an (semi-)accurate gradient. 
- [] determine the optimal scaling factor to use

## Plan
- [] perform SPSA using many samples (20 or so)
- [] compare performance to FDSA (collect 2 samples)
- [] determine where we get deminishing returns on gradient accuracy 
- [] repeat using different scaling factors

"""
import argparse
import copy
import csv
import datetime
import logging
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import spsa
import neural_pretrain as neural


def run_FDSA_SPSA(
    n_components,
    experiments_csv,
    scaling_factor=0.20,
    perturbs_output_file="SPSAvFDSA_expt/spsa_perturbations_simulation.csv",
    spsa_output_file="SPSAvFDSA_expt/spsa_results.csv",
    fdsa_output_file="SPSAvFDSA_expt/fdsa_results.csv",
):

    if not os.path.isdir("SPSAvFDSA_expt"):
        os.makedirs("SPSAvFDSA_expt")

    experiments = np.genfromtxt(experiments_csv, delimiter=",")

    net = neural.PredictNet(
        n_test=0,
        exp_id=0,
        parent_logdir="SPSAvFDSA_expt/tf_logs",
        save_model_path="SPSAvFDSA_expt/neural_model",
        n_epochs=10,
    )

    # Train growth predict net
    x, y = neural.load_data(
        filepath=f"models/iSMU-test/data_20_extrapolated.csv", starting_index=0,
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.001)
    net.train(x_train, y_train, x_test, y_test)
    # net.save()

    spsa_perturbations = []
    spsa_results = []
    fdsa_results = []
    for e, expt_binary in enumerate(experiments):
        print(f"\n\nExperiments #{e}")
        remaining_indexes = np.where(expt_binary == 1)[0]
        n_remaining = remaining_indexes.shape[0]
        s = spsa.SPSA(W=np.ones(n_remaining))

        fdsa_experiments = s.gen_fdsa_experiments()
        spsa_experiments, perturbations = s.gen_spsa_experiments(n_remaining)
        spsa_perturbations.append((n_remaining, perturbations))

        n_expts = len(fdsa_experiments)
        print("FDSA", n_expts)
        print("SPSA", len(spsa_experiments))
        # print("n_remaining", n_remaining)

        fdsa_inputs = np.zeros((2 * n_expts, n_components), dtype=np.float)
        for idx, (e_plus, e_minus) in enumerate(fdsa_experiments):
            fdsa_inputs[idx, remaining_indexes] = 1
            fdsa_inputs[idx, remaining_indexes] *= e_plus
            fdsa_inputs[idx + n_expts, remaining_indexes] = 1
            fdsa_inputs[idx + n_expts, remaining_indexes] *= e_minus

        # print(fdsa_inputs)

        spsa_inputs = np.zeros((2 * n_expts, n_components), dtype=np.float)
        for idx, (e_plus, e_minus) in enumerate(spsa_experiments):
            spsa_inputs[idx, remaining_indexes] = 1
            spsa_inputs[idx, remaining_indexes] *= e_plus
            spsa_inputs[idx + n_expts, remaining_indexes] = 1
            spsa_inputs[idx + n_expts, remaining_indexes] *= e_minus

        # print(spsa_inputs)

        fdsa_predictions = net.predict_probability(fdsa_inputs).reshape((n_expts * 2,))
        spsa_predictions = net.predict_probability(spsa_inputs).reshape((n_expts * 2,))

        # print(fdsa_predictions)
        # print(spsa_predictions)

        fdsa_results.append([n_expts] + fdsa_predictions.tolist())
        spsa_results.append([n_expts] + fdsa_predictions.tolist())

        # print(fdsa_results)
        # print(spsa_results)

    with open(spsa_output_file, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["n_experiments", "results_plus", "results_minus"])
        for r in spsa_results:
            writer.writerow(r)

    with open(fdsa_output_file, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["n_experiments", "results_plus", "results_minus"])
        for r in fdsa_results:
            writer.writerow(r)

    with open(perturbs_output_file, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["n_grads", "perturbations"])
        for n, p in spsa_perturbations:
            perturbs = [n]
            perturbs += [" ".join(map(str, x.tolist())) for x in p]
            writer.writerow(perturbs)


def generate_random_experiments(
    n_experiments,
    n_components,
    output_file="SPSAvFDSA_expt/experiments_random_simulation.csv",
):
    """
    Generates random binary experiments.
    """

    experiments = np.ones((n_experiments, n_components))

    expts_used = set()
    for idx, row in enumerate(experiments):
        while True:
            n_to_remove = np.random.randint(1, n_components)
            indexes_to_remove = np.random.choice(
                range(n_components), size=n_to_remove, replace=False
            )
            index_tup = tuple(sorted(indexes_to_remove.tolist()))
            if index_tup not in expts_used:
                expts_used.add(index_tup)
                break

        experiments[idx, indexes_to_remove] = 0

    with open(output_file, "w") as f:
        np.savetxt(f, experiments, fmt="%i", delimiter=",")


def run_simulation():
    n_components = 20
    expt_output = "SPSAvFDSA_expt/experiments_random_simulation.csv"
    perturb_output = "SPSAvFDSA_expt/spsa_perturbations_simulation.csv"
    spsa_results_csv = "SPSAvFDSA_expt/spsa_results.csv"
    fdsa_results_csv = "SPSAvFDSA_expt/fdsa_results.csv"

    generate_random_experiments(50, n_components, expt_output)
    run_FDSA_SPSA(
        n_components,
        expt_output,
        0.20,
        perturb_output,
        spsa_results_csv,
        fdsa_results_csv,
    )
    all_angles = spsa.compare_gradients(
        0.20, spsa_results_csv, perturb_output, fdsa_results_csv
    )
    # Graph results

    for i, y in enumerate(all_angles):
        plt.plot(range(len(y)), y, "-", linewidth=2, alpha=0.50)
    # plt.legend(
    #     [f"Expt #{i}" for i in range(len(all_angles))],
    #     bbox_to_anchor=(1.05, 1.0),
    #     loc="upper left",
    # )
    plt.xlim(left=0)
    plt.ylim(bottom=0, top=180)
    plt.xlabel("# of grads")
    plt.ylabel("Angular Distance")
    plt.title("SPSA vs. FDSA")
    # plt.tight_layout()
    plt.savefig("SPSA_vs_FDSA_simulation.png")


if __name__ == "__main__":
    run_simulation()
