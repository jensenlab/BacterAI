import collections
import datetime
import os
import time

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import KFold

from global_vars import *
import utils

VERBOSE = True


class GeneticSolver:
    def __init__(
        self,
        run_name,
        output_folder,
        n_groups,
        rule_choices,
        population_size,
        data,
        fitness_lambda=1,
    ):
        self.run_name = run_name
        self.output_folder = output_folder
        self.n_groups = n_groups
        self.rule_choices = rule_choices  # np.arange(21)
        self.current_generation = np.random.choice(
            rule_choices, size=(population_size, self.n_groups * 4), replace=True
        )

        # np.array first 20 cols are input AAs (binary), last col is growth
        self.data = data

        self.generation_stats = {"fitness_avg": [], "fitness_max": []}
        self.generation_n = 0
        self.fitness_lambda = fitness_lambda

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

        # n_zero_percent = (rule == 0).sum() / len(rule)
        # # lamb = 0.95
        # score = (balanced_accuracy * self.fitness_lambda) + (
        #     n_zero_percent * (1 - self.fitness_lambda)
        # )
        score = balanced_accuracy
        # score = tp + tn - fp - fn

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
            cols = r[r >= 0]
            if cols.size > 0:
                results = results & np.any(self.data[:, cols], axis=1)

        return results

    def split_rule(self, rule):
        rule = np.array_split(rule.astype(int), len(rule) // 4)
        return rule

    def final_rule_clean(self, rule):
        og_score = self.fitness_score(rule)
        for x in range(len(rule)):
            rule_copy = rule.copy()
            if rule_copy[x] == 0:
                continue
            rule_copy[x] = 0
            score = self.fitness_score(rule_copy)
            if score >= og_score:
                rule = rule_copy

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
            zero_bias = 0.75
            choice_bias = [zero_bias] + [(1 - zero_bias) / (n_choices - 1)] * (
                n_choices - 1
            )
        else:
            choice_bias = [1 / n_choices] * n_choices

        # zero_mutation_bias = 0.5
        zero_mutation_bias = 0.25
        mutation_probability = [zero_mutation_bias] + [
            (1 - zero_mutation_bias) / rule_len
        ] * rule_len
        choices = np.arange(0, rule_len + 1)
        for row in a:
            n_mutations = np.random.choice(choices, 1, p=mutation_probability)
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

    def _build_summary_output(self, rule):
        _, metrics = self.fitness_score(rule, include_metrics=True)
        rule_split = self.split_rule(rule)

        individual_scores = []
        diff_rule = np.zeros(len(rule))
        for idx, group in enumerate(rule_split):

            diff_rule[len(group) * idx : len(group) * (idx + 1)] = group
            score, _ = self.fitness_score(diff_rule, include_metrics=True)
            current_sum = sum(individual_scores)
            individual_scores.append(score - current_sum)

        rule_split = [
            sorted([AA_SHORT[i - 1] for i in y if i != 0]) for y in rule_split
        ]

        output = [
            f"\n>> {self.run_name} ------------------------\n",
            f"Stopped after {self.generation_n-1} generations.\n",
            f"Final accuracy: {metrics['accuracy']*100:.2f}%\n",
            f"Final balanced accuracy: {metrics['balanced_accuracy']*100:.2f}%\n",
            f"Final TPR: {metrics['TPR']*100:.2f}%\n",
            f"Final TNR: {metrics['TNR']*100:.2f}%\n",
            f"Best Rule: {rule.tolist()}:\n",
        ]
        for group, score_diff in zip(rule_split, individual_scores):
            output.append(f"\t({' or '.join(group)}) {score_diff*100:+.2f}%\n")

        return output

    def summarize_score(self, rule):
        output = self._build_summary_output(rule)
        cleaned_rule = self.final_rule_clean(rule)
        clean_output = self._build_summary_output(cleaned_rule)

        current_date = datetime.datetime.now().isoformat().replace(":", ".")
        output_file = os.path.join(self.output_folder, "genetic_solver_output.txt")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        with open(output_file, "a") as f:
            f.writelines(output)
            f.writelines(clean_output)

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

        self.generation_n = 0
        starting_time = time.time()
        while True:
            if max_generations and self.generation_n > max_generations:
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
            if VERBOSE:
                print(
                    f"Generation {self.generation_n:3} - Avg. Fitness: {avg_fitness*100:.2f}% - Max. Fitness: {max_fitness*100:.2f}%"
                )

            self.generation_n += 1
            self.generation_stats["fitness_avg"].append(avg_fitness)
            self.generation_stats["fitness_max"].append(max_fitness)

        best_idx = np.argmax(fitnesses)
        best_rule = self.current_generation[best_idx].astype(int)

        print(f"Time elapsed: {elapsed:.0f}s")
        print(f"Max fitness: {max_fitness:.5f}.")

        self.summarize_score(best_rule)

        return best_rule


def plot_hit_miss_rates(solver, round_data, rule):
    print(len(round_data))
    print(len(set(map(tuple, round_data.to_numpy()[:, :20]))))

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


def solve(round_data, run_name, output_folder, threshold=0.25):
    # Median all duplicated experiments since we repeat some
    round_data = round_data.groupby(
        list(round_data.columns[:20]), as_index=False
    ).median()

    fitness_data = round_data["fitness"].to_numpy()
    fitness_data[fitness_data >= threshold] = 1
    fitness_data[fitness_data < threshold] = 0

    if "round_n" in round_data.columns:
        round_nums = round_data["round_n"]

    round_data = round_data.iloc[:, :20]
    round_data.columns = list(range(1, 21))
    round_data["fitness"] = fitness_data
    if "round_n" in round_data.columns:
        round_data["round_n"] = round_nums

    choices = np.arange(21)
    pop_size = 1000
    n_groups = 4
    solver = GeneticSolver(
        run_name, output_folder, n_groups, choices, pop_size, round_data.to_numpy()
    )
    rule = solver.solve(elite_p=0.25, mutation_mu=5, max_generations=150)

    # rule = np.array([0, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # plot_hit_miss_rates(solver, round_data, rule)


def solve_kfold_regularization(round_data, run_name, threshold=0.25):
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
    round_data = round_data.to_numpy()

    fitness_lambdas = [1.0, 0.99, 0.975, 0.95, 0.9, 0.5]
    kf = KFold(n_splits=len(fitness_lambdas), shuffle=True)
    choices = np.arange(21)
    pop_size = 1000
    n_groups = 4

    final_scores = {}
    for lamb, (train_index, test_index) in zip(fitness_lambdas, kf.split(round_data)):
        # split_train, split_test = round_data[train_index], X[test_index]
        all_indexes = np.hstack((train_index, test_index))
        split_data = round_data[all_indexes]
        solver = GeneticSolver(
            run_name,
            n_groups,
            choices,
            pop_size,
            split_data,
            fitness_lambda=lamb,
        )
        rule = solver.solve(elite_p=0.25, mutation_mu=5, max_generations=100)
        print(f"Finished Solving with Lambda: {lamb}")
        print()


def evaluate_rule(rule, round_data, run_name, threshold=0.25):
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
    solver = GeneticSolver(run_name, n_groups, choices, pop_size, round_data.to_numpy())

    solver.summarize_score(rule)


def main(folder, max_round_n):
    folders = [
        os.path.join(folder, i, "results_all.csv")
        for i in os.listdir(folder)
        if "Round" in i
    ]
    folders = sorted(folders, key=lambda x: (len(x), x))[:max_round_n]

    print(folders)
    round_data = []
    for i, f in enumerate(sorted(folders)):
        data = utils.normalize_ingredient_names(pd.read_csv(f, index_col=None))
        data["round_n"] = i
        round_data.append(data)

    round_data = pd.concat(round_data, ignore_index=True)
    # round_data.to_csv("roundata.csv", index=False)
    round_data["direction"] = "DOWN"
    print(f"Round data: {round_data.shape[0]}")
    run_name = f"{folder} - Round {max_round_n}"
    output_folder = os.path.join(folder, "rule_results")
    solve(round_data, run_name, output_folder)
    # solve_kfold_regularization(round_data)


# def main2():
#     folder = "data/SMU_data/standards"
#     # folder = "data/SMU_data/kk1"
#     # folder = "data/SMU_data/kk2"
#     # folder = "data/SMU_data/kk3"
#     # folder = "data/SMU_data/randoms"
#     files = [os.path.join(folder, i) for i in os.listdir(folder)]

#     # files = ["data/SMU_data/standards/L1I-L2I SMU a438.csv", "data/SMU_data/standards/L1O-L2O SMU 3e31.csvs"]
#     # files = [
#     #     # "data/SMU_data/standards/L1IO-L2IO-Rand SMU UA159 69be.csv",
#     #     # "data/SMU_data/standards/L1IO-L2IO-Rand SMU UA159 (3) 9b54.csv",
#     #     "data/SMU_data/standards/L1I-L2I SMU a438.csv",
#     # ]

#     round_data = [
#         utils.normalize_ingredient_names(pd.read_csv(f, index_col=None)) for f in files
#     ]
#     round_data = pd.concat(round_data, ignore_index=True)
#     round_data = round_data.replace(
#         {"Anaerobic Chamber @ 37 C": "anaerobic", "5% CO2 @ 37 C": "aerobic"}
#     )
#     round_data = round_data[round_data["environment"] == "aerobic"]
#     round_data["direction"] = "DOWN"
#     round_data["solution_id_hex"] = None
#     print(round_data.shape)
#     round_data = round_data[round_data["good_data"] != False]
#     print(round_data.shape)

#     drops = [i for i in ["good_data", "reason", "reasons"] if i in round_data.columns]
#     round_data = round_data.drop(columns=drops)
#     round_data = round_data.reset_index(drop=True)
#     round_data.to_csv("data.csv", index=False)
#     round_data, _, _ = utils.process_mapped_data("data.csv")
#     # round_data["depth"] = round_data.iloc[:, :20].sum(axis=1)
#     # round_data = round_data[round_data["depth"] <= 2]

#     # print(round_data["depth"])
#     # print(round_data.shape)

#     # rule = np.array([0, 6, 7, 0, 0, 10, 11, 20, 0, 0, 0, 0, 0, 0, 0, 0])
#     # rule = np.array([0, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
#     # evaluate_rule(rule, round_data)
#     solve(round_data)
#     # solve_kfold_regularization(round_data)


if __name__ == "__main__":
    VERBOSE = True
    # experiment_folder = "experiments/04-07-2021 copy"
    # experiment_folder = "experiments/04-30-2021_3/both"
    # experiment_folder = "experiments/05-31-2021_7/"
    # experiment_folder = "experiments/05-31-2021_8/"
    # experiment_folder = "experiments/05-31-2021_9/"
    experiment_folder = "experiments/07-26-2021_11"

    for i in range(1, 13):
        print()
        print(i, "-----------------------------")
        main(experiment_folder, i)

    # main(experiment_folder, 6)
    # main(experiment_folder, 7)

    # main2()
