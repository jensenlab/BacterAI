import os
import time

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
        rule_as_idxes = self.np_to_rule(rule - 1)

        results = np.empty((len(self.data), len(rule_as_idxes)))
        for idx, r in enumerate(rule_as_idxes):
            r = r[r >= 0]
            result = np.any(self.data[:, r], axis=1)
            results[:, idx] = result
        results = np.all(results, axis=1)

        # balanced accuracy
        tn, fp, fn, tp = confusion_matrix(self.data[:, -1], results).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        balanced_accuracy = (tpr + tnr) / 2
        acc = (tp + tn) / len(results)
        # cross entropy
        # score = -log_loss(self.data[:, -1], results)

        # non_zero_percent = sum(n_non_zero) / self.n_groups / 4
        n_zero_percent = (rule == 0).sum() / len(rule)
        lamb = 0.90
        score = (balanced_accuracy * lamb) + (n_zero_percent * (1 - lamb))
        # score = balanced_accuracy
        if include_metrics:
            metrics = {
                "TPR": tpr,
                "TNR": tnr,
                "accuracy": acc,
                "balanced_accuracy": balanced_accuracy,
            }
            return score, metrics

        return score

    def np_to_rule(self, np_rule):
        np_rule = np_rule.astype(int)
        rule = [np_rule[i : i + 4] for i in range(0, self.n_groups * 4, 4)]

        return rule

    def clean_rule(self, rule):
        clean = np.zeros(self.n_groups * 4)
        for i in range(0, self.n_groups * 4, 4):
            unique = np.unique(rule[i : i + 4])
            clean[i : i + len(unique)] = unique

        return clean

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

    def perform_mutations(self, a, poisson_center=2):
        # print("mut before:", a)
        idxs = np.arange(a.shape[1])
        rule_len = self.n_groups * 4
        for row in a:
            # n_mutations = poisson.rvs(poisson_center, size=1)
            n_mutations = np.random.choice(
                np.arange(0, rule_len + 1),
                1,
                p=[0.5] + [0.5 / rule_len] * rule_len,
            )
            mutation_idxs = np.random.choice(idxs, size=n_mutations, replace=False)
            row[mutation_idxs] = np.random.choice(
                self.rule_choices, size=n_mutations, replace=True
            )
        # print("mut after:", a)

    def get_elite(self, p, fitnesses):
        n = int(p * len(self.current_generation))
        n -= n % 2  # ensure even number

        elite_idxs = np.argsort(fitnesses)[-n:]  # Sorted ascending Grab largest fitness
        elites = self.current_generation[elite_idxs, :]
        return elites

    def _new_pair(self, pair, mu):
        # pair = self.select_from_population(p)
        # print(f"\npair:\n {pair}")
        self.perform_crossover(pair)
        # print(f"pair cross:\n {pair}")
        self.perform_mutations(pair, mu)
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

        # for i in range(0, n, 2):
        #     pair = selections[[i, i + 1], :]
        #     # print(f"\npair:\n {pair}")
        #     self.perform_crossover(pair)
        #     # print(f"pair cross:\n {pair}")
        #     self.perform_mutations(pair, mutation_mu)
        #     # print(f"pair mutations:\n {pair}")
        #     new_generation[[i, i + 1], :] = pair

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
        print(f"Best Rule: {best_rule}:")

        rule = self.np_to_rule(best_rule)
        rule = [sorted([AA_SHORT[i - 1] for i in y if i != 0]) for y in rule]
        for group in rule:
            print(f"\t({' or '.join(group)})")

        return rule


def main(folder, threshold=0.25):
    folders = [
        os.path.join(folder, i, "results_all.csv")
        for i in os.listdir(folder)
        if "Round" in i
    ]
    folders = sorted(folders, key=lambda x: (len(x), x))

    print(folders)
    results_data = []

    round_data = [pd.read_csv(f, index_col=None) for f in folders]
    round_data = pd.concat(round_data, ignore_index=True)
    round_data["direction"] = "DOWN"

    # Mean all duplicated experiments since we repeat some
    round_data = round_data.groupby(
        list(round_data.columns[:20]), as_index=False
    ).mean()
    fitness_data = round_data["fitness"].to_numpy()
    fitness_data[fitness_data >= threshold] = 1
    fitness_data[fitness_data < threshold] = 0

    round_data = round_data.iloc[:, :20]
    round_data.columns = list(range(1, 21))
    round_data["fitness"] = fitness_data
    round_data = round_data.to_numpy()

    choices = np.arange(21)
    pop_size = 100
    n_groups = 4
    solver = GeneticSolver(n_groups, choices, pop_size, round_data)
    rule = solver.solve(elite_p=0.25, mutation_mu=5, max_generations=200)


if __name__ == "__main__":
    # experiment_folder = "experiments/04-07-2021 copy"
    experiment_folder = "experiments/04-30-2021_3/both"
    main(experiment_folder)
