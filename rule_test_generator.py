import itertools
import string

import numpy as np
import sympy
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
    rule = simplify_logic(rule, sym_to_ingredient)
    rule = sympy_to_rule(rule)
    print(rule)
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


def make_ingredient_combos(rule):
    ingredients = set(np.unique(rule)) - set([0])

    all_combos = []
    for i in range(1, len(ingredients) + 1):
        c = itertools.combinations(ingredients, i)
        all_combos += list(c)

    print(len(all_combos))
    for c in all_combos:
        print(c)


def main(rule):
    rule = np.array(rule)
    split_rule = np.vstack(np.array_split(rule.astype(int), len(rule) // 4))

    # rule_to_dnf(split_rule)

    make_ingredient_combos(rule)


if __name__ == "__main__":
    # rule = [0, 10, 15, 17, 0, 6, 7, 0, 11, 15, 16, 20, 0, 10, 11, 0]
    # main(rule)

    rule = [
        [
            0,
            10,
            15,
            0,
        ],
        [
            0,
            6,
            7,
            0,
        ],
        [
            1,
            11,
            15,
            20,
        ],
        [0, 10, 11, 0],
    ]
    simplify(rule)
