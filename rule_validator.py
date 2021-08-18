
import csv
import datetime
import itertools
import os
import string

import numpy as np
from sympy.logic.boolalg import And, Or, to_dnf, to_cnf, simplify_logic

from global_vars import *
import utils


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

    ingredients = set(np.unique(rule)) - set([0])

    all_combos = []
    for i in range(1, len(ingredients) + 1):
        c = itertools.combinations(ingredients, i)
        all_combos += list(c)

    ingredient_names = AA_NAMES_TEMPEST if use_tempest else AA_NAMES_2
    ingredients = {i + 1: n for i, n in enumerate(ingredient_names)}
    ingredient_names = set(ingredient_names)

    date = datetime.datetime.now().isoformat().replace(":", ".")
    file_path = os.path.join(parent_path, f"rule_verification_dp_{date}.csv")
    with open(file_path, "w") as f:
        writer = csv.writer(f, delimiter=",")
        for combo in all_combos:
            leave_ins = [tempest_ingredients[x] for x in combo]
            leave_outs = sorted(ingredient_names - set(leave_ins))
            writer.writerow(leave_outs)

    print()
    print(f"Exported batch ({len(all_combos)} experiments) at:\n\t{file_path}")

def find_violations(rule, mapped_data_path, threshold=0.25):
    data = utils.process_mapped_data(mapped_data_path)[0]
    data = data[list(data.columns)[:20] + ["fitness"]]
    ng = data[data["fitness"] < threshold] 
    g = data[data["fitness"] >= threshold] 
    print(ng)
    print(g)
    

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
    rule = [0, 11, 20, 0, 0, 10, 11, 0, 0, 6, 7, 0, 0, 0, 0, 0]
    
    make_ingredient_combos(rule)
    # mapped_data_path = "experiments/05-31-2021_7/Round7/BacterAI SMU UA159 (7R7) a919 mapped_data.csv"
    # find_violations(rule, mapped_data_path)


    # rule = [0, 11, 15, 20, 0, 10, 11, 20, 0, 6, 7, 0, 0, 10, 11, 15]
    # rule = np.array_split(np.array(rule).astype(int), len(rule) // 4)
    # simplify(rule)

    # rule_to_dnf(split_rule)
