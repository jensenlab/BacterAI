import collections
import os
import time

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import confusion_matrix, log_loss

from global_vars import *
import utils


class GeneticSolver:
    def __init__(self, n_groups, rule_choices, population_size, data):
        self.n_groups = n_groups
        self.rule_choices = rule_choices  # np.arange(21)
        self.current_generation = np.random.choice(
            rule_choices, size=(population_size, self.n_groups * 4), replace=True
        )

        # np.array first 20 cols are input AAs (binary), last col is growth
        self.data = data

        self.generation_stats = {"fitness_avg": [], "fitness_max": []}

    def fitness_score(self, rule, include_metrics=False):
        results = self.apply_rule(rule)

        results_grow = results == 1
        data_grow = self.data[:, -1] == 1
        tp = np.sum(np.logical_and(results_grow, data_grow))
        tn = np.sum(np.logical_and(~results_grow, ~data_grow))
        fp = np.sum(np.logical_and(results_grow, ~data_grow))
        fn = np.sum(np.logical_and(~results_grow, data_grow))

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        balanced_accuracy = (tpr + tnr) / 2
        acc = (tp + tn) / len(results)
        # cross entropy
        # score = -log_loss(self.data[:, -1], results)

        n_zero_percent = (rule == 0).sum() / len(rule)
        lamb = 0.95
        # score = (balanced_accuracy * lamb) + (n_zero_percent * (1 - lamb))
        score = balanced_accuracy
        if include_metrics:
            metrics = {
                "TPR": tpr,
                "TNR": tnr,
                "accuracy": acc,
                "balanced_accuracy": balanced_accuracy,
            }
            return score, metrics

        return score

    def apply_rule(self, rule):
        rule_as_idxes = self.split_rule(rule - 1)

        r = rule_as_idxes[0]
        results = np.any(self.data[:, r[r >= 0]], axis=1)
        for r in rule_as_idxes[1:]:
            results = results & np.any(self.data[:, r[r >= 0]], axis=1)

        return results

    def split_rule(self, rule):
        rule = np.array_split(rule.astype(int), len(rule) // 4)
        return rule

    def final_rule_clean(self, rule):
        og_score, og_metrics = self.fitness_score(rule, include_metrics=True)
        for x in range(len(rule)):
            rule_copy = rule.copy()
            rule_copy[x] = 0
            score, metrics = self.fitness_score(rule_copy, include_metrics=True)

            print(x, metrics)
            if metrics["balanced_accuracy"] >= og_metrics["balanced_accuracy"]:
                rule = rule_copy
                print("Better!")

            print()

        return rule

    def clean_rule(self, rule):
        clean = np.zeros(self.n_groups * 4)
        seen_group = set()
        for i in range(0, self.n_groups * 4, 4):
            unique = np.unique(rule[i : i + 4])
            key = tuple(unique)
            if not key in seen_group:
                seen_group.add(key)
                clean[i : i + len(unique)] = unique

        return clean

    def perform_group_crossover(self, pair):
        cross_idx = np.random.choice(np.arange(4), size=1, replace=False)[0] * 4
        pair[[0, 1], cross_idx : cross_idx + 4] = pair[
            [1, 0], cross_idx : cross_idx + 4
        ]

    def perform_group_recombination(self, pair):
        before = pair.copy()
        both_grouped = np.vstack(self.split_rule(pair.flatten()))
        group_idxs = np.arange(len(both_grouped), dtype=int)
        np.random.shuffle(group_idxs)
        pair[0, :] = both_grouped[group_idxs[: len(group_idxs) // 2]].flatten()
        pair[1, :] = both_grouped[group_idxs[len(group_idxs) // 2 :]].flatten()

    def perform_crossover(self, pair, double_crossover=False):
        if double_crossover:
            cross_idxs = np.random.choice(
                np.arange(pair.shape[1]), size=2, replace=False
            )
            pair[[0, 1], cross_idxs[0] : cross_idxs[1]] = pair[
                [1, 0], cross_idxs[0] : cross_idxs[1]
            ]
        else:
            cross_idx = np.random.choice(
                np.arange(pair.shape[1]), size=1, replace=False
            )[0]
            pair[[0, 1], :cross_idx] = pair[[1, 0], :cross_idx]

    def perform_mutations(self, a, poisson_center=2, bias_zero_choice=False):
        idxs = np.arange(a.shape[1])
        rule_len = self.n_groups * 4
        n_choices = len(self.rule_choices)
        if bias_zero_choice:
            choice_bias = [0.75] + [0.25 / (n_choices - 1)] * (n_choices - 1)
        else:
            choice_bias = [1 / n_choices] * n_choices
        for row in a:
            n_mutations = np.random.choice(
                np.arange(0, rule_len + 1),
                1,
                p=[0.5] + [0.5 / rule_len] * rule_len,
            )
            mutation_idxs = np.random.choice(idxs, size=n_mutations, replace=False)
            row[mutation_idxs] = np.random.choice(
                self.rule_choices,
                size=n_mutations,
                replace=True,
                p=choice_bias,
            )
        # print("mut after:", a)

    def get_elite(self, p, fitnesses):
        n = int(p * len(self.current_generation))
        n -= n % 2  # ensure even number

        elite_idxs = np.argsort(fitnesses)[-n:]  # Sorted ascending Grab largest fitness
        elites = self.current_generation[elite_idxs, :]
        return elites

    def _new_pair(self, pair, mu):
        # self.perform_crossover(pair, double_crossover=True)
        # self.perform_group_recombination(pair)
        self.perform_crossover(pair)
        # self.perform_group_crossover(pair)
        # self.perform_mutations(pair, mu)
        self.perform_mutations(pair, mu, bias_zero_choice=True)
        return pair

    def get_new_generation(self, n, fitnesses, mutation_mu):
        selection_bias = utils.softmax(fitnesses)
        new_generation = np.zeros((n, self.current_generation.shape[1]))

        selections = self.select_from_population(selection_bias, n)
        with multiprocessing.Pool(processes=30) as pool:
            results = pool.starmap(
                self._new_pair,
                [(selections[[i, i + 1], :], mutation_mu) for i in range(0, n, 2)],
            )

        new_generation = np.concatenate(results)

        return new_generation

    def select_from_population(self, p, n):
        selected_idxs = np.random.choice(
            np.arange(len(self.current_generation)), size=n, replace=True, p=p
        )
        selections = self.current_generation[selected_idxs, :]

        return selections

    def solve(
        self,
        elite_p=0.20,
        mutation_mu=2,
        max_generations=None,
        fitness_threshold=None,
        timeout_seconds=None,
    ):
        if (
            max_generations is None
            and fitness_threshold is None
            and timeout_seconds is None
        ):
            raise Exception("Need 1 or more stopping conditions!")

        generation_n = 0
        starting_time = time.time()
        while True:
            if max_generations and generation_n > max_generations:
                break
            elapsed = time.time() - starting_time
            if timeout_seconds and elapsed >= timeout_seconds:
                break

            # fitnesses = np.apply_along_axis(
            #     self.fitness_score, 1, self.current_generation
            # )
            with multiprocessing.Pool(processes=30) as pool:
                self.current_generation = np.vstack(
                    pool.starmap(
                        self.clean_rule,
                        [(row,) for row in self.current_generation],
                    )
                )
                fitnesses = np.array(
                    pool.starmap(
                        self.fitness_score,
                        [(row,) for row in self.current_generation],
                    )
                )

            avg_fitness = fitnesses.sum() / len(fitnesses)
            max_fitness = fitnesses.max()
            if fitness_threshold and max_fitness >= fitness_threshold:
                break

            elites = self.get_elite(elite_p, fitnesses)
            n_remaining = len(self.current_generation) - len(elites)
            new_generation = self.get_new_generation(
                n_remaining, fitnesses, mutation_mu
            )
            new_generation = np.vstack((elites, new_generation))
            self.current_generation = new_generation
            print(
                f"Generation {generation_n:3} - Avg. Fitness: {avg_fitness*100:.2f}% - Max. Fitness: {max_fitness*100:.2f}%"
            )

            generation_n += 1
            self.generation_stats["fitness_avg"].append(avg_fitness)
            self.generation_stats["fitness_max"].append(max_fitness)

        best_idx = np.argmax(fitnesses)
        best_rule = self.current_generation[best_idx]

        best_rule = self.final_rule_clean(best_rule)

        score, metrics = self.fitness_score(best_rule, include_metrics=True)
        print()
        print("---------- FINISHED ----------")
        print(f"Stopped after {generation_n-1} generations.")
        print(f"Time elapsed: {elapsed:.0f}s")
        print(f"Max fitness: {max_fitness:.5f}.")
        print(f"Final accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Final balanced accuracy: {metrics['balanced_accuracy']*100:.2f}%")
        print(f"Final TPR: {metrics['TPR']*100:.2f}%")
        print(f"Final TNR: {metrics['TNR']*100:.2f}%")
        print(f"Best Rule: {best_rule.tolist()}:")

        rule = self.split_rule(best_rule)
        rule = [sorted([AA_SHORT[i - 1] for i in y if i != 0]) for y in rule]
        for group in rule:
            print(f"\t({' or '.join(group)})")

        return rule


def plot_hit_miss_rates():
    rule = np.array([0, 10, 15, 17, 0, 6, 7, 0, 11, 15, 16, 20, 0, 10, 11, 0])
    # rule = np.array([0, 10, 11, 0, 0, 6, 7, 0, 0, 11, 15, 20, 0, 10, 15, 0])

    round_data["n_ingredients"] = 20 - round_data.iloc[:, :20].sum(axis=1)
    round_data["rule_pred"] = solver.apply_rule(rule)
    round_data["fitness"] = round_data["fitness"].astype(bool)
    round_data["correct"] = round_data["fitness"] == round_data["rule_pred"]

    fig, axs = plt.subplots(
        nrows=2, ncols=2, sharex=False, sharey=True, figsize=(12, 10)
    )

    values = [(1, 1), (1, 0), (0, 1), (0, 0)]
    for ax, vals in zip(axs.flatten(), values):
        correct_counts = collections.Counter(
            round_data[
                (round_data["fitness"] == vals[0]) & (round_data["correct"] == vals[1])
            ]["n_ingredients"].values.tolist()
        )

        print(correct_counts)
        ax.bar(correct_counts.keys(), correct_counts.values())

    axs[0, 0].set_title("grow correct counts")
    axs[0, 1].set_title("grow incorrect counts")
    axs[1, 0].set_title("no grow correct counts")
    axs[1, 1].set_title("no grow incorrect counts")

    for ax in axs.flatten():
        ax.set_xticks(np.arange(0, 21))
        ax.set_xlabel("N ingredients removed")

    # plt.suptitle(f"Experiment: {prev_folder}")
    plt.tight_layout()
    plt.savefig("rule_solver_counts.png", dpi=400)


def solve(round_data, threshold=0.25):
    # Median all duplicated experiments since we repeat some
    round_data = round_data.groupby(
        list(round_data.columns[:20]), as_index=False
    ).median()
    fitness_data = round_data["fitness"].to_numpy()
    fitness_data[fitness_data >= threshold] = 1
    fitness_data[fitness_data < threshold] = 0

    round_data = round_data.iloc[:, :20]
    round_data.columns = list(range(1, 21))
    round_data["fitness"] = fitness_data

    choices = np.arange(21)
    pop_size = 1000
    n_groups = 4
    solver = GeneticSolver(n_groups, choices, pop_size, round_data.to_numpy())
    c = solver.solve(elite_p=0.25, mutation_mu=5, max_generations=100)


def main(folder):
    max_round_n = 7
    folders = [
        os.path.join(folder, i, "results_all.csv")
        for i in os.listdir(folder)
        if "Round" in i
    ]
    folders = sorted(folders, key=lambda x: (len(x), x))[:max_round_n]

    print(folders)
    results_data = []
    round_data = [pd.read_csv(f, index_col=None) for f in folders]
    round_data = pd.concat(round_data, ignore_index=True)
    round_data["direction"] = "DOWN"
    solve(round_data)

    # # folder = "data/SMU_data/standards"
    # # folder = "data/SMU_data/kk1"
    # # folder = "data/SMU_data/kk2"
    # # folder = "data/SMU_data/kk3"
    # folder = "data/SMU_data/randoms"
    # files = [os.path.join(folder, i) for i in os.listdir(folder)]
    # round_data = [pd.read_csv(f, index_col=None) for f in files]

    # # for i, d in enumerate(round_data):
    # #     print(files[i], d[d["environment"].isnull()])

    # round_data = pd.concat(round_data, ignore_index=True)
    # round_data = round_data.replace(
    #     {"Anaerobic Chamber @ 37 C": "anaerobic", "5% CO2 @ 37 C": "aerobic"}
    # )
    # # print(pd.unique(round_data["environment"]))
    # round_data = round_data[round_data["environment"] == "aerobic"]
    # round_data["direction"] = "DOWN"
    # round_data["solution_id_hex"] = None
    # print(round_data.shape)
    # round_data = round_data[round_data["good_data"] != False]
    # print(round_data.shape)

    # drops = [i for i in ["good_data", "reason", "reasons"] if i in round_data.columns]
    # round_data = round_data.drop(columns=drops)
    # round_data = round_data.reset_index(drop=True)
    # round_data.to_csv("all_smu_data.csv", index=False)
    # round_data, _, _ = utils.process_mapped_data("all_smu_data.csv")
    # solve(round_data)


if __name__ == "__main__":
    # experiment_folder = "experiments/04-07-2021 copy"
    experiment_folder = "experiments/04-30-2021_3/both"
    main(experiment_folder)
