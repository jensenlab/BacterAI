
import csv
import datetime
import itertools
import os
import string

import pandas as pd
import numpy as np
from sympy.logic.boolalg import And, Or, to_dnf, to_cnf, simplify_logic

from global_vars import *
import models
import utils

SEED = 0
NP_RAND_STATE = utils.seed_numpy_state(SEED)

def rule_to_sympy(rule):
    ingredients = np.unique(rule)
    alpha_symbols = string.ascii_lowercase[: len(ingredients)]
    ingredient_to_sym = dict(zip(ingredients, alpha_symbols))
    sym_to_ingredient = dict(zip(alpha_symbols, ingredients))

    # print(rule)
    # print(ingredient_to_sym)

    ands = []
    for row in rule:
        ors = " | ".join([ingredient_to_sym[i] for i in row])
        ands.append(f"({ors})")
    rule = " & ".join(ands)
    return rule, sym_to_ingredient


def sympy_to_rule(sympy_rule, sym_to_ingredient):
    sympy_rule = str(sympy_rule).replace("(", "").replace(")", "")
    ors = sympy_rule.split(" | ")
    rule = [list(map(lambda a: sym_to_ingredient[a], o.split(" & "))) for o in ors]
    return rule


def simplify(rule):
    rule, sym_to_ingredient = rule_to_sympy(rule)
    print(rule)
    rule = to_cnf(rule)
    print(rule)

    rule = simplify_logic(rule, form="cnf")
    print(rule)
    rule = sympy_to_rule(rule, sym_to_ingredient)
    # print(rule)
    return rule


def rule_to_dnf(rule):
    rule = rule_to_sympy(rule)
    # print(ands)

    dnf_form = to_dnf(rule)
    simplified = simplify_logic(dnf_form)
    # print(dnf_form)
    # print(simplified)

    rule = sympy_to_rule(simplified)

    # print(rule)
    return rule


def make_ingredient_combos(rule, parent_path="", use_tempest=True):
    rule = np.array(rule)
    rule = np.vstack(np.array_split(rule.astype(int), len(rule) // 4))
    print(rule)

    ingredients = set(np.unique(rule)) - set([0])

    all_combos = []
    for i in range(1, len(ingredients) + 1):
        print(i)
        c = list(itertools.combinations(ingredients, i))

        all_combos += list(c)
        for x in c:
            print(x)

    ingredient_names = AA_NAMES_TEMPEST if use_tempest else AA_NAMES_2
    ingredients = {i + 1: n for i, n in enumerate(ingredient_names)}
    ingredient_names = set(ingredient_names)

    date = datetime.datetime.now().isoformat().replace(":", ".")


    file_path = os.path.join(parent_path, f"rule_verification_dp_{date}.csv")
    with open(file_path, "w") as f:
        writer = csv.writer(f, delimiter=",")
        for combo in all_combos:
            leave_ins = [ingredients[x] for x in combo]
            leave_outs = sorted(ingredient_names - set(leave_ins))
            writer.writerow(leave_outs)

    print()
    print(f"Exported batch ({len(all_combos)} experiments) at:\n\t{file_path}")

def find_violations(rule, mapped_data_path, threshold=0.25):
    rule = np.array(rule)
    def _apply_rule(rule, data):
        rule_as_idxes = _split_rule(rule - 1)
        r = rule_as_idxes[0]
        results = np.any(data[:, r[r >= 0]], axis=1)
        for r in rule_as_idxes[1:]:
            cols = r[r >= 0]
            if cols.size > 0:
                results = results & np.any(data[:, cols], axis=1)

        return results

    def _split_rule(rule):
        rule = np.array_split(rule.astype(int), len(rule) // 4)
        return rule

    data = utils.process_mapped_data(mapped_data_path)[0]
    fitness = data["fitness"]
    data = data[list(data.columns)[:20]]
    data["grow_pred"] = _apply_rule(rule, data.iloc[:, :20].values)
    data["grow_true"] = False
    data.loc[fitness >= threshold, "grow_true"] = True

    pred_grow = data["grow_pred"].values
    true_grow = data["grow_true"].values

    tp = np.sum(np.logical_and(pred_grow, true_grow))
    tn = np.sum(np.logical_and(~pred_grow, ~true_grow))
    fp = np.sum(np.logical_and(pred_grow, ~true_grow))
    fn = np.sum(np.logical_and(~pred_grow, true_grow))
    
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    print(f"{tp=} {tn=} {fp=} {fn=}")

    tp /= len(data)
    tn /= len(data)
    fp /= len(data)
    fn /= len(data)
    print(f"{tp=:.3f} {tn=:.3f} {fp=:.3f} {fn=:.3f}")
    print(f"{tpr=:.3f} {tnr=:.3f}")
    print(f"{tn + tp:.3f} {fn + fp:.3f}")

    # ng = data[data["fitness"] < threshold] 
    # g = data[data["fitness"] >= threshold] 
    # print(ng)
    # print(g)
    # print(g.columns)

    # sm = ['arg', 'phe', 'ser', 'tyr']
    # for row in g.to_numpy()[:, :20]:
    #     present = []
    #     for i, n in zip(row, g.columns[:20]):
    #         if i:
    #             present.append(n)

    #     print(sorted(present))


    # for row in ng.to_numpy():
    #     present = []
    #     for i, n in zip(row[:20], g.columns[:20]):
    #         if i:
    #             present.append(n)

    #     present = sorted(present)

    #     flag = True
    #     for x in sm:
    #         if x not in present:
    #             flag = False
    #             break
    #     if flag:
    #         print(round(row[-1],2))

            
def make_G_NG_experiments(n_grow, n_no_grow, rule, exclude, output_folder, use_tempest=True):
    rule = np.array(rule) - 1
    rule = np.array_split(rule.astype(int), len(rule) // 4)


    def _apply_rule(state):
        r = rule[0]
        result = np.any(state[r[r >= 0]])
        for r in rule[1:]:
            cols = r[r >= 0]
            if cols.size > 0:
                result = result & np.any(state[cols])

        return result

    exclude = set([tuple(x) for x in exclude])

    grows = []
    no_grows = []
    count = 0
    while True:
        candidate_state = NP_RAND_STATE.choice([0, 1], 20, replace=True)
        count += 1

        if tuple(candidate_state) in exclude:
            print("Skip:", tuple(candidate_state))
            continue

        result = _apply_rule(candidate_state)
        if len(grows) < n_grow and result:
            grows.append(candidate_state)
            exclude.add(tuple(candidate_state.tolist()))
        elif len(no_grows) < n_no_grow and not result:
            no_grows.append(candidate_state)
            exclude.add(tuple(candidate_state.tolist()))
        elif len(grows) >= n_grow and len(no_grows) >= n_no_grow:
            break

        # print(len(grows), len(no_grows))

    all_experiments = grows + no_grows
    grows = pd.DataFrame(np.vstack(grows))
    no_grows = pd.DataFrame(np.vstack(no_grows))
    print(all_experiments)
    print((n_grow+n_no_grow)/count)
    # print(grows)
    # print(no_grows)

    ingredient_names = AA_NAMES_TEMPEST if use_tempest else AA_NAMES_2
    ingredients = {i: n for i, n in enumerate(ingredient_names)}

    date = datetime.datetime.now().isoformat().replace(":", ".")


    file_path = os.path.join(output_folder, f"rule_verification_SGO_10_{date}.csv")
    with open(file_path, "w") as f:
        writer = csv.writer(f, delimiter=",")
        for exp in all_experiments:
            leave_outs = [ingredients[idx] for idx, x in enumerate(exp) if x == 0]
            writer.writerow(leave_outs)


    grows["grow"] = True
    no_grows["grow"] = False
    file_path = os.path.join(output_folder, f"rule_verification_experiments_SGO_10_{date}.csv")
    out = pd.concat((grows, no_grows), ignore_index=False)
    out.to_csv(file_path, index=False)

def eval_G_NG_experiments(mapped_data_path, expected_results_path, nn_models_path, threshold=0.25):
    data = utils.process_mapped_data(mapped_data_path)[0]
    aa_names = list(data.columns)[:20]
    data = data[aa_names + ["fitness"]]
    data["grow_true"] = False
    data.loc[data["fitness"] >= threshold, "grow_true"] = True
    # data = data.drop(columns=["fitness"])

    expected = pd.read_csv(expected_results_path, index_col=None)
    expected.columns = aa_names + ["grow_pred"]

    combined = data.merge(expected, on=aa_names)

    model = models.NeuralNetModel.load_trained_models(nn_models_path)
    combined["grow_nn_pred"] = model.evaluate(combined.iloc[:, :20].values)[0]
    combined.loc[combined["grow_nn_pred"] >= threshold, "grow_nn_pred"] = True
    combined.loc[combined["grow_nn_pred"] < threshold, "grow_nn_pred"] = False

    correct = (combined["grow_pred"] == combined["grow_true"]).sum()/len(combined)
    correct_nn = (combined["grow_nn_pred"] == combined["grow_true"]).sum()/len(combined)

    tp = np.sum(np.logical_and(combined["grow_true"], combined["grow_pred"]))
    tn = np.sum(np.logical_and(~combined["grow_true"], ~combined["grow_pred"]))
    fp = np.sum(np.logical_and(combined["grow_true"], ~combined["grow_pred"]))
    fn = np.sum(np.logical_and(~combined["grow_true"], combined["grow_pred"]))

    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    
    tp /= len(combined)
    tn /= len(combined)
    fp /= len(combined)
    fn /= len(combined)
    print(f"{tp=} {tn=} {fp=} {fn=}")
    print(f"{tpr=} {tnr=}")
    print(f"Correct: {correct*100:.2f}%")
    print(f"Correct NN: {correct_nn*100:.2f}%")
    # print(expected)
    print(combined)


if __name__ == "__main__":
    # Expt 7R6
    # (ile or leu or val)
    # (gln or glu)
    # (leu or pro)
    # rule = [0, 10, 11, 20, 0, 6, 7, 0, 0, 0, 0, 0, 0, 11, 15, 0]

    # Expt 8R6
    # (ile)
    # (val)
    # (gln or glu)
    # rule = [0, 10, 0, 0, 0, 20, 0, 0, 0, 6, 7, 0, 0, 0, 0, 0]

    # Expt 9R6
    # (leu or val)
    # (ile or leu)
    # (gln or glu)
    # rule = [0, 11, 20, 0, 0, 10, 11, 0, 0, 6, 7, 0, 0, 0, 0, 0]

    # Expt 10R14
    # rule = [0, 19, 0, 0, 0, 14, 0, 0, 0, 16, 0, 0, 0, 11, 0, 0, 0, 2, 0, 0, 0, 5, 20, 0, 0, 6, 7, 0]
    # EXPT 10R13
    rule = [0, 2, 0, 0, 0, 11, 0, 0, 0, 16, 0, 0, 0, 19, 0, 0, 0, 5, 20, 0, 0, 14, 0, 0, 0, 6, 7, 0]
    path = "experiments/07-26-2021_10/rule_results"

    # exclude = pd.read_csv("experiments/07-26-2021_10/Round14/train_pred.csv", index_col=None)
    
    # .iloc[:, :20].values.tolist()

    # print(exclude[exclude["y_true"] >= 0.25])
    # print(exclude[exclude["y_true"] < 0.25])

    # n = 500
    # make_G_NG_experiments(n, n, rule, exclude, path)
    # eval_G_NG_experiments("experiments/07-26-2021_10/rule_results/BacterAI SGO CH1 (10R13) rule verify 12fb mapped_data.csv", 
    # "experiments/07-26-2021_10/rule_results/rule_verification_experiments_SGO_10_2021-10-03T23.53.10.505987.csv",
    # "experiments/07-26-2021_10/Round13/nn_models")

    # make_ingredient_combos(rule, path)

    # mapped_data_name = "BacterAI SGO CH1 (10R14) rule verify 2963 mapped_data.csv"
    mapped_data_name = "Randoms (1) SGO CH1 17f3 mapped_data.csv"
    find_violations(rule, os.path.join(path, mapped_data_name))


    # rule = [0, 11, 15, 20, 0, 10, 11, 20, 0, 6, 7, 0, 0, 10, 11, 15]
    # rule = np.array_split(np.array(rule).astype(int), len(rule) // 4)
    # simplify(rule)

    # rule_to_dnf(split_rule)
