import collections
from enum import Enum
import os
import time
import datetime
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from constants import *
import utils

VERBOSE = False
N_PROCESSES = 10


class CrossoverMethod(Enum):
    SINGLE_CROSSOVER = 0
    DOUBLE_CROSSOVER = 1
    GROUP_CROSSOVER = 2
    GROUP_RECOMBINATION = 3


class GeneticSolver:
    def __init__(
        self,
        run_name,
        output_folder,
        n_groups,
        rule_choices,
        population_size,
        data,
        summary_file_info,
        ingredient_names,
        no_test_train_split=False,
        bias_zero_choice=True,
    ):
        self.run_name = run_name
        self.output_folder = output_folder
        self.n_groups = n_groups
        self.rule_choices = rule_choices.astype("int16")
        self.current_generation = np.random.choice(
            rule_choices, size=(population_size, self.n_groups * 4), replace=True
        )

        # np.array first 20 cols are input AAs (binary), last col is growth
        data = data.astype("int16")
        if no_test_train_split:
            self.train_data = data
        else:
            self.train_data, self.test_data = train_test_split(data, test_size=0.25)

        self.generation_stats = {"fitness_avg": [], "fitness_max": []}
        self.generation_n = 0
        self.summary_file_info = summary_file_info
        self.ingredient_names = ingredient_names

        # mutation metrics
        self.rule_len = self.n_groups * 4
        self.mut_idxs = np.arange(self.rule_len, dtype="int16")
        n_choices = len(self.rule_choices)
        zero_bias = 0.75
        mut_choice_bias_zero = [zero_bias] + [(1 - zero_bias) / (n_choices - 1)] * (
            n_choices - 1
        )
        mut_choice_bias = [1 / n_choices] * n_choices
        self.mut_bias = mut_choice_bias_zero if bias_zero_choice else mut_choice_bias
        zero_mutation_bias = 0.25
        self.mutation_probability = [zero_mutation_bias] + [
            (1 - zero_mutation_bias) / self.rule_len
        ] * self.rule_len
        self.mut_choices = np.arange(0, self.rule_len + 1, dtype="int16")

    def fitness_score(self, rule, include_metrics=False, use_test_data=False):
        data = self.test_data if use_test_data else self.train_data

        results = self.apply_rule(rule, data)

        pred_grow = results == 1
        true_grow = data[:, -1] == 1

        tp = np.sum(np.logical_and(pred_grow, true_grow))
        tn = np.sum(np.logical_and(~pred_grow, ~true_grow))
        fp = np.sum(np.logical_and(pred_grow, ~true_grow))
        fn = np.sum(np.logical_and(~pred_grow, true_grow))

        denom = tp + fn
        tpr = 0 if denom == 0 else tp / denom
        denom = tn + fp
        tnr = 0 if denom == 0 else tn / denom
        balanced_accuracy = (tpr + tnr) / 2

        if include_metrics:
            acc = (tp + tn) / len(results)
            metrics = {
                "TPR": tpr,
                "TNR": tnr,
                "accuracy": acc,
                "balanced_accuracy": balanced_accuracy,
            }
            return balanced_accuracy, metrics

        return balanced_accuracy

    # @decoratortimer(5)
    def apply_rule(self, rule, data):
        rule_as_idx = rule - 1
        i = 0
        cols = rule_as_idx[i : i + 4]
        cols = cols[cols >= 0]
        results = np.any(data[:, cols], axis=1)
        for i in range(4, self.n_groups * 4 + 1, 4):
            cols = rule_as_idx[i : i + 4]
            cols = cols[cols >= 0]
            if cols.size:
                results = results & np.any(data[:, cols], axis=1)

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
        clean = np.zeros(self.n_groups * 4, dtype="int16")
        seen_group = set()
        for i in range(0, self.n_groups * 4, 4):
            unique = np.unique(rule[i : i + 4])
            key = tuple(sorted(unique.tolist()))
            if not key in seen_group:
                seen_group.add(key)
                clean[i : i + len(unique)] = unique

        return clean

    def perform_group_crossover(self, pair):
        cross_idx = (
            np.random.choice(np.arange(4, dtype="int16"), size=1, replace=False)[0] * 4
        )
        pair[[0, 1], cross_idx : cross_idx + 4] = pair[
            [1, 0], cross_idx : cross_idx + 4
        ]

    def perform_group_recombination(self, pair):
        both_grouped = np.vstack(self.split_rule(pair.flatten()), dtype="int16")
        group_idxs = np.arange(len(both_grouped), dtype="int16")
        np.random.shuffle(group_idxs)
        pair[0, :] = both_grouped[group_idxs[: len(group_idxs) // 2]].flatten()
        pair[1, :] = both_grouped[group_idxs[len(group_idxs) // 2 :]].flatten()

    def perform_double_crossover(self, pair):
        cross_idxs = np.random.choice(
            np.arange(pair.shape[1], dtype="int16"), size=2, replace=False
        )
        pair[[0, 1], cross_idxs[0] : cross_idxs[1]] = pair[
            [1, 0], cross_idxs[0] : cross_idxs[1]
        ]

    def perform_crossover(self, pair, cross_idx=None):
        if cross_idx == None:
            cross_idx = np.random.choice(
                np.arange(pair.shape[1], dtype="int16"), size=1, replace=False
            )[0]
        pair[[0, 1], :cross_idx] = pair[[1, 0], :cross_idx]
        return pair

    def perform_mutations(self, single, n_mutations):
        mutation_idxs = np.random.choice(self.mut_idxs, size=n_mutations, replace=False)
        single[mutation_idxs] = np.random.choice(
            self.rule_choices,
            size=n_mutations,
            replace=True,
            p=self.mut_bias,
        )
        return single

    def get_elite(self, p, fitnesses):
        n = int(p * len(self.current_generation))
        n -= n % 2  # ensure even number

        elite_idxs = np.argsort(fitnesses)[-n:]  # Sorted ascending Grab largest fitness
        elites = self.current_generation[elite_idxs, :]
        return elites

    def _new_pair(self, pair, method=CrossoverMethod.SINGLE_CROSSOVER):
        if method == CrossoverMethod.SINGLE_CROSSOVER:
            self.perform_crossover(pair)
        elif method == CrossoverMethod.DOUBLE_CROSSOVER:
            self.perform_double_crossover(pair)
        elif method == CrossoverMethod.GROUP_CROSSOVER:
            self.perform_group_crossover(pair)
        elif method == CrossoverMethod.GROUP_RECOMBINATION:
            self.perform_group_recombination(pair)
        return pair

    def get_new_generation(self, n, fitnesses):
        selection_bias = utils.softmax(fitnesses)
        selections = self.select_from_population(selection_bias, n)
        n_pairs = len(selections) // 2
        selection_pairs = np.split(selections, n_pairs)

        cross_idxs = np.random.choice(
            np.arange(self.rule_len, dtype="int16"), size=n_pairs, replace=True
        )
        new_generation = list(map(self.perform_crossover, selection_pairs, cross_idxs))
        new_generation = np.vstack(new_generation)

        n_mutations = np.random.choice(
            self.mut_choices, len(selections), p=self.mutation_probability
        )
        new_generation = list(map(self.perform_mutations, new_generation, n_mutations))
        return new_generation

    def select_from_population(self, p, n):
        selected_idxs = np.random.choice(
            np.arange(len(self.current_generation), dtype="int16"),
            size=n,
            replace=True,
            p=p,
        )
        selections = self.current_generation[selected_idxs, :]
        return selections

    def _build_summary_output(self, rule):
        _, train_metrics = self.fitness_score(
            rule, include_metrics=True, use_test_data=False
        )
        score, test_metrics = self.fitness_score(
            rule, include_metrics=True, use_test_data=True
        )
        rule_split = self.split_rule(rule)

        add_rule = np.zeros(len(rule), dtype="int16")

        add_scores = []
        remove_scores = []
        for idx, group in enumerate(rule_split):

            add_rule[0 : len(group)] = group
            add_score = self.fitness_score(add_rule, use_test_data=True)
            add_scores.append(add_score)

            remove_rule = np.copy(rule)
            remove_rule[len(group) * idx : len(group) * (idx + 1)] = 0
            remove_score = self.fitness_score(remove_rule, use_test_data=True)
            remove_scores.append(remove_score)

        rule_split = sorted(
            [
                sorted([self.ingredient_names[i - 1] for i in y if i != 0])
                for y in rule_split
            ],
            key=len,
        )
        rule_split_filtered = [x for x in rule_split if len(x)]
        rule_str = "".join([f"({' or '.join(group)})" for group in rule_split])
        rule_str_filtered = "".join(
            [f"({' or '.join(group)})" for group in rule_split_filtered]
        )

        metrics = {
            "n_generations": self.generation_n,
            "train_acc": train_metrics["accuracy"],
            "train_bal_acc": train_metrics["balanced_accuracy"],
            "train_tpr": train_metrics["TPR"],
            "train_tnr": train_metrics["TNR"],
            "test_acc": test_metrics["accuracy"],
            "test_bal_acc": test_metrics["balanced_accuracy"],
            "test_tpr": test_metrics["TPR"],
            "test_tnr": test_metrics["TNR"],
            "rule_str": rule_str_filtered,
            "rule_raw_idx": rule.tolist(),
        }

        output = [
            f"\n>> {self.run_name} ------------------------\n",
            f"Stopped after {self.generation_n-1} generations.\n",
            f"Number of groups: {self.n_groups}\n",
            f"Final train accuracy: {train_metrics['accuracy']*100:.2f}%\n",
            f"Final test accuracy: {test_metrics['accuracy']*100:.2f}%\n",
            f"Final test balanced accuracy: {test_metrics['balanced_accuracy']*100:.2f}%\n",
            f"Final TPR: {test_metrics['TPR']*100:.2f}%\n",
            f"Final TNR: {test_metrics['TNR']*100:.2f}%\n",
            f"\t{rule_str_filtered}\n\n",
            f"Best Rule:\n",
            f"\t{rule_str}\n\n",
            f"\t{rule.tolist()}\n\n",
        ]

        for group, add_score, remove_score in zip(
            rule_split, add_scores, remove_scores
        ):
            output.append(
                f"\t({' or '.join(group)}) - Added: {add_score*100:+.2f}%, Removed: {(remove_score-score)*100:+.2f}%\n"
            )

        for line in output:
            print(line, end="")
        return output, metrics

    def summarize_score(self, rule, clean_only=True):
        if not clean_only:
            output, _ = self._build_summary_output(rule)
        cleaned_rule = self.final_rule_clean(rule)
        clean_output, metrics = self._build_summary_output(cleaned_rule)

        output_file = os.path.join(
            self.output_folder,
            f"genetic_solver_output-{self.n_groups}_groups-{self.summary_file_info}.txt",
        )
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        with open(output_file, "a") as f:
            if not clean_only:
                f.writelines(output)
            f.writelines(clean_output)

        return metrics

    def solve(
        self,
        elite_p=0.20,
        max_generations=None,
        fitness_threshold=None,
        timeout_seconds=None,
        tqdm_offset=0
    ):
        if (
            max_generations is None
            and fitness_threshold is None
            and timeout_seconds is None
        ):
            raise Exception("Need 1 or more stopping conditions!")

        self.generation_n = 0
        starting_time = time.time()
        for _ in (pbar := tqdm(range(max_generations), position=tqdm_offset)):
            elapsed = time.time() - starting_time

            self.current_generation = np.apply_along_axis(
                self.clean_rule, 1, self.current_generation
            )
            fitnesses = np.apply_along_axis(
                self.fitness_score, 1, self.current_generation
            )

            avg_fitness = fitnesses.sum() / len(fitnesses)
            max_fitness = fitnesses.max()
            if fitness_threshold and max_fitness >= fitness_threshold:
                break

            elites = self.get_elite(elite_p, fitnesses)
            n_remaining = len(self.current_generation) - len(elites)
            new_generation = self.get_new_generation(n_remaining, fitnesses)
            new_generation = np.vstack((elites, new_generation))
            self.current_generation = new_generation
            if VERBOSE:
                print(
                    f"Generation {self.generation_n:3} - Avg. Fitness: {avg_fitness*100:.2f}% - Max. Fitness: {max_fitness*100:.2f}%"
                )

            self.generation_n += 1
            self.generation_stats["fitness_avg"].append(avg_fitness)
            self.generation_stats["fitness_max"].append(max_fitness)
            pbar.set_description(f"{self.summary_file_info:<10} - Avg: {avg_fitness*100:.2f}%, Max: {max_fitness*100:.2f}%")

        best_idx = np.argmax(fitnesses)
        best_rule = self.current_generation[best_idx].astype(int)

        print(f"Time elapsed: {elapsed:.0f}s")
        print(f"Max fitness: {max_fitness:.5f}.")

        metrics = self.summarize_score(best_rule)

        return best_rule, metrics


def plot_hit_miss_rates(solver, round_data, rule, n_ingredients):
    print(len(round_data))
    print(len(set(map(tuple, round_data.to_numpy()[:, :n_ingredients]))))

    round_data["n_ingredients"] = n_ingredients - round_data.iloc[
        :, :n_ingredients
    ].sum(axis=1)
    round_data["rule_pred"] = solver.apply_rule(
        rule, round_data.iloc[:, :n_ingredients].to_numpy()
    )
    round_data["fitness"] = round_data["fitness"].astype(bool)
    round_data["correct"] = round_data["fitness"] == round_data["rule_pred"]

    _, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=True, figsize=(12, 10))

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
        ax.set_xticks(np.arange(0, n_ingredients + 1))
        ax.set_xlabel("N ingredients removed")

    # plt.suptitle(f"Experiment: {prev_folder}")
    plt.tight_layout()
    plt.savefig("rule_solver_counts.png", dpi=400)


def solve(
    round_data,
    run_name,
    output_folder,
    ingredient_names,
    n_groups,
    summary_file_info,
    threshold=0.25,
    max_generations=250,
    tqdm_offset=0,
):
    # Median all duplicated experiments since we repeat some
    n_ingredients = len(ingredient_names)
    round_data = round_data.groupby(
        list(round_data.columns[:n_ingredients]), as_index=False
    ).agg({"fitness": "median"})

    fitness_data = round_data["fitness"].to_numpy()
    fitness_data[fitness_data >= threshold] = 1
    fitness_data[fitness_data < threshold] = 0

    if "round_n" in round_data.columns:
        round_nums = round_data["round_n"]

    round_data = round_data.iloc[:, :n_ingredients]
    round_data.columns = list(range(1, n_ingredients + 1))
    round_data["fitness"] = fitness_data
    if "round_n" in round_data.columns:
        round_data["round_n"] = round_nums

    choices = np.arange(n_ingredients + 1, dtype="int16")
    pop_size = 1000
    solver = GeneticSolver(
        run_name,
        output_folder,
        n_groups,
        choices,
        pop_size,
        round_data.to_numpy(),
        summary_file_info,
        ingredient_names,
    )
    rule, metrics = solver.solve(elite_p=0.25, max_generations=max_generations, tqdm_offset=tqdm_offset)
    # plot_hit_miss_rates(solver, round_data, rule, n_ingredients)
    return metrics


def evaluate_rule(
    rule,
    round_data,
    run_name,
    output_folder,
    n_ingredients,
    threshold=0.25,
    export_summary=True,
):
    # Median all duplicated experiments since we repeat some
    round_data = round_data.groupby(
        list(round_data.columns[:n_ingredients]), as_index=False
    ).median()

    fitness_data = round_data["fitness"].to_numpy()
    fitness_data[fitness_data >= threshold] = 1
    fitness_data[fitness_data < threshold] = 0

    round_data = round_data.iloc[:, :n_ingredients]
    round_data.columns = list(range(1, n_ingredients + 1))
    round_data["fitness"] = fitness_data
    round_data = round_data.to_numpy()

    train_set, test_set = train_test_split(round_data, test_size=0.25)

    choices = np.arange(n_ingredients + 1, dtype="int16")
    pop_size = 1000
    n_groups = 4
    solver = GeneticSolver(
        run_name, output_folder, n_groups, choices, pop_size, train_set
    )

    if export_summary:
        solver.summarize_score(rule)


def main(
    folder,
    ingredient_names,
    max_round_n,
    n_groups,
    summary_file_info,
    max_generations,
    include_nots=False,
    tqdm_offset=0,
    **kwargs,
):
    folders = [
        os.path.join(folder, i, "results_all.csv")
        for i in os.listdir(folder)
        if "Round" in i
    ]
    folders = sorted(folders, key=lambda x: (len(x), x))[:max_round_n]

    round_data = []
    for i, f in enumerate(sorted(folders)):
        data = utils.normalize_ingredient_names(pd.read_csv(f, index_col=None))
        data["round_n"] = i
        round_data.append(data)

    round_data = pd.concat(round_data, ignore_index=True)
    round_data["direction"] = "DOWN"

    if include_nots:
        half_n = len(ingredient_names) // 2
        og_data = round_data.iloc[:, :half_n].to_numpy()
        not_data = pd.DataFrame(1 - og_data, columns=ingredient_names[half_n:])
        round_data = pd.concat(
            (round_data.iloc[:, :half_n], not_data, round_data.iloc[:, half_n:]), axis=1
        )
    
    if VERBOSE:
        print(f"Round data: {round_data.shape[0]}, {list(round_data.columns)}")

    run_name = f"{folder} - Round {max_round_n}"
    output_folder = os.path.join(folder, "rule_results")
    metrics = solve(
        round_data=round_data,
        run_name=run_name,
        output_folder=output_folder,
        ingredient_names=ingredient_names,
        n_groups=n_groups,
        summary_file_info=summary_file_info,
        max_generations=max_generations,
        tqdm_offset=tqdm_offset,
    )
    return metrics


from itertools import repeat

def starmap_with_kwargs(pool, fn, kwargs_iter):
    args_for_starmap = zip(repeat(fn), kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, kwargs):
    return fn(**kwargs)


if __name__ == "__main__":
    MAX_GENERATIONS = 2000

    aa = AA_SHORT
    aa_nots = AA_SHORT + AA_SHORT_NOTS
    cdm = AA_SHORT + BASE_NAMES
    cdm_nots = AA_SHORT + BASE_NAMES + AA_SHORT_NOTS + BASE_NAMES_NOTS

    configs_kwargs = [
        {
            "folder": "experiments/2021-07-26_10",
            "ingredient_names": aa,
            "max_round_n": 13,
            "n_groups": 20,
            "include_nots": False,
            "summary_file_info": "10_AA",
            "max_generations": MAX_GENERATIONS,
            "key": "SGO_AA_aerobic",
        },
        {
            "folder": "experiments/2021-08-20_12",
            "ingredient_names": aa,
            "max_round_n": 11,
            "n_groups": 20,
            "include_nots": False,
            "summary_file_info": "11_AA",
            "max_generations": MAX_GENERATIONS,
            "key": "SSA_AA_aerobic",
        },
        {
            "folder": "experiments/2022-01-17_19",
            "ingredient_names": aa,
            "max_round_n": 4,
            "n_groups": 20,
            "include_nots": False,
            "summary_file_info": "19_AA",
            "max_generations": MAX_GENERATIONS,
            "key": "SGO_AA_aerobic_TL",
        },
        {
            "folder": "experiments/2022-02-08_24",
            "ingredient_names": aa,
            "max_round_n": 3,
            "n_groups": 20,
            "include_nots": False,
            "summary_file_info": "24_AA",
            "max_generations": MAX_GENERATIONS,
            "key": "SSA_AA_anaerobic_TL",
        },
        {
            "folder": "experiments/2022-04-18_25",
            "ingredient_names": cdm,
            "max_round_n": 7,
            "n_groups": 20,
            "include_nots": False,
            "summary_file_info": "25_CDM",
            "max_generations": MAX_GENERATIONS,
            "key": "SSA_CDM_aerobic_TL",
        },
        # NOTS
        {
            "folder": "experiments/2021-07-26_10",
            "ingredient_names": aa_nots,
            "max_round_n": 13,
            "n_groups": 20,
            "include_nots": True,
            "summary_file_info": "10_AA_NOT",
            "max_generations": MAX_GENERATIONS,
            "key": "SGO_AA_aerobic_NOTS",
        },
        {
            "folder": "experiments/2021-08-20_12",
            "ingredient_names": aa_nots,
            "max_round_n": 11,
            "n_groups": 20,
            "include_nots": True,
            "summary_file_info": "11_AA_NOT",
            "max_generations": MAX_GENERATIONS,
            "key": "SSA_AA_aerobic_NOTS",
        },
        {
            "folder": "experiments/2022-01-17_19",
            "ingredient_names": aa_nots,
            "max_round_n": 4,
            "n_groups": 20,
            "include_nots": True,
            "summary_file_info": "19_AA_NOT",
            "max_generations": MAX_GENERATIONS,
            "key": "SGO_AA_aerobic_TL_NOTS",
        },
        {
            "folder": "experiments/2022-02-08_24",
            "ingredient_names": aa_nots,
            "max_round_n": 3,
            "n_groups": 20,
            "include_nots": True,
            "summary_file_info": "24_AA_NOT",
            "max_generations": MAX_GENERATIONS,
            "key": "SSA_AA_anaerobic_TL_NOTS",
        },
        {
            "folder": "experiments/2022-04-18_25",
            "ingredient_names": cdm_nots,
            "max_round_n": 7,
            "n_groups": 20,
            "include_nots": True,
            "summary_file_info": "25_CDM_NOT",
            "max_generations": MAX_GENERATIONS*2,
            "key": "SSA_CDM_aerobic_TL_NOTS",
        },
    ]

    current_date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    export_filename = f"rule_solver_results_{current_date}.csv"

    for i, configs_kwarg in enumerate(configs_kwargs):
        configs_kwarg["tqdm_offset"] = i

    with multiprocessing.Pool(processes=N_PROCESSES) as pool:
        results = starmap_with_kwargs(pool, main, configs_kwargs)
    
    results_accum = {}
    for k, r in zip(configs_kwargs, results):
        results_accum[k["key"]] = r
        results = pd.DataFrame.from_dict(results_accum, orient="index")
        results.to_csv(export_filename)
