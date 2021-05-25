from copy import deepcopy
import itertools
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, log_loss

from utils import decoratortimer
from global_vars import *

# @decoratortimer(2)
def score_rule(rule, data):
    # rule = rule.astype(int).tolist()
    # rule = [rule[0:4], rule[4:8], rule[8:12], rule[12:]]

    print(rule)
    # rule = [[1, 3, 4, 0], [1, 5, 10, 0], [3, 14, 18, 20], [6, 7, 0, 0]]
    # rule = np.array([[1,3,4,0], [1,5,10, 0], [3, 14, 18, 20], [6,7, 0, 0]])
    total_experiments = len(data)

    results = np.empty((total_experiments, len(rule[0])))
    for idx, r in enumerate(rule):
        r = [i - 1 for i in r if i > 0]
        result = np.any(data[:, r], axis=1)
        results[:, idx] = result

    results = np.all(results, axis=1)
    # print(results)

    # balanced accuracy
    tn, fp, fn, tp = confusion_matrix(data[:, -1], results).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    balanced_accuracy = (tpr + tnr) / 2
    acc = (tp + tn) / len(results)
    # cross entropy
    # score = -log_loss(self.data[:, -1], results)

    # non_zero_percent = sum(n_non_zero) / self.n_groups / 4

    # rule = np.array(rule).flatten()
    # n_zero_percent = (rule == 0).sum() / len(rule)
    # lamb = 0.90
    score = balanced_accuracy
    # score = (balanced_accuracy * lamb) + (n_zero_percent * (1 - lamb))

    # # ce = log_loss(data[:, -1], results)
    # print(
    #     f"ACCURACY: {acc*100:.0f}% both, {tpr*100:.0f}% G, {tnr*100:.0f}% NG, {balanced_accuracy*100:.0f}% balanced"
    # )
    # # print("F1:", f1)

    # ce = log_loss(data[:, -1], results)
    print(
        f"ACCURACY: {acc*100:.0f}% both, {tpr*100:.0f}% G, {tnr*100:.0f}% NG, {balanced_accuracy*100:.0f}% balanced"
    )
    # print("F1:", f1)
    # print(f"ce: {ce}")

    return score
    # return ce
    # return f1


# @decoratortimer(2)
def rollout(action_key, shared_scores, n, rule, max_group_length, data):
    scores = np.zeros(n)
    for i in range(n):
        test_rule = deepcopy(rule)
        while True:
            available_actions = get_available_actions(test_rule, max_group_length)
            # print(f"available_actions: {available_actions}")
            if available_actions is None:
                break

            action = np.random.choice(available_actions, 1)[0]
            # print(f"action: {action}")
            test_rule = update_rule(action, test_rule, max_group_length)
            # print(f"new_rule: {rule}")
        scores[i] = score_rule(test_rule, data)

    score = scores.mean()
    shared_scores[action_key] = score


# @decoratortimer(2)
def get_available_actions(rule, max_group_length):
    all_actions = set(range(0, 21))
    for group in rule:
        length = len(group)
        if length == max_group_length:
            continue
        # elif length == 0:
        #     start = 0
        # else:
        #     start = max(group)
        # return [0] + list(range(start + 1, 21))
        return list(all_actions - set(group))

    # Rule is full
    return None


# @decoratortimer(2)
def update_rule(action, rule, max_group_length):
    new_rule = deepcopy(rule)
    for group in new_rule:
        # print(group, new_rule)
        length = len(group)
        if length == max_group_length:
            continue
        if action == 0:
            # Fill remaining with zeros
            group += [0] * (max_group_length - length)
        else:
            group.append(action)
        break

    return new_rule


@decoratortimer(2)
def solve(data, n_rollouts, max_groups, max_group_length):
    rule = [[] for _ in range(max_groups)]

    while True:
        available_actions = get_available_actions(rule, max_group_length)
        # print(available_actions)
        if available_actions is None:
            break

        with mp.Manager() as manager:
            scores = manager.dict()
            # scores = [0] * len(available_actions)
            processes = []
            for action in available_actions:
                new_rule = update_rule(action, rule, max_group_length)
                # print(new_rule)
                p = mp.Process(
                    target=rollout,
                    args=(action, scores, n_rollouts, new_rule, max_group_length, data),
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            # print(scores)
            scores = [scores[i] for i in available_actions]
            # print(scores)
        # scores = np.zeros(len(available_actions))
        # for idx, action in enumerate(available_actions):
        #     new_rule = update_rule(action, rule, max_group_length)
        #     rollout_score = rollout(n_rollouts, new_rule, max_group_length, data)
        #     scores[idx] = rollout_score

        best_action = available_actions[np.argmax(scores)]
        # best_action = available_actions[np.argmin(scores)]

        rule = update_rule(best_action, rule, max_group_length)

    score = score_rule(rule, data)
    return rule, score


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

    # combos = list(itertools.combinations_with_replacement(range(1, 11), 2))
    combos = [(4, 4)]
    results = {"max_groups": [], "max_group_length": [], "score": [], "rule": []}

    for i, (g, gl) in enumerate(combos):
        print(f"{i+1:2}/{len(combos)} - max_groups: {g}, max_group_length: {gl}")
        rule, score = solve(
            round_data, n_rollouts=100, max_groups=g, max_group_length=gl
        )

        print(f"Score: {score*100:.2f}%")
        rule = [sorted([AA_SHORT[i - 1] for i in y if i != 0]) for y in rule]
        for group in rule:
            # g = [AA_SHORT[i - 1] for i in group if i != 0]
            print(" or ".join(group))

        print()

        results["max_groups"].append(g)
        results["max_group_length"].append(gl)
        results["score"].append(score)
        results["rule"].append(rule)

    results = pd.DataFrame(results)
    # results = results.sort_values(by="score", ascending=False)
    # print(results)

    # results.to_csv("rule_solver_results.csv", columns=False)
    # The best solution found:
    #  [ 0.  0. 12.  7.  5.  7.  7.  7.  0.  0.  0.  0.  5.  3.  4.  0.]
    # Best Rule: [ 0  0 11 10 17 15 10 10  7  0  7  6 20 15 11 16].
    # score_rule(
    #     [
    #         [0, 0, 11, 10],
    #         [17, 15, 10, 10],
    #         [7, 0, 7, 6],
    #         [
    #             20,
    #             15,
    #             11,
    #             16,
    #         ],
    #     ],
    #     round_data,
    # )
    # rule = [[] for _ in range(4)]
    # s = rollout(200, rule, 4, round_data.to_numpy())
    # print(s)


if __name__ == "__main__":
    # experiment_folder = "experiments/04-30-2021_3/both"
    experiment_folder = "experiments/04-07-2021 copy"
    main(experiment_folder)
