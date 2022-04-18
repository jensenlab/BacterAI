import os
import time

import numpy as np
import pandas as pd

from global_vars import *


def seed_numpy_state(seed):
    return np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))


def process_mapped_data(path, ingredients=AA_SHORT):
    """Processes DeepPhenotyping data. It normalizes the
    change in OD (delta OD) to their plate controls' mean delta OD.
    """

    n_ingredients = len(ingredients)
    data = pd.read_csv(path, index_col=None).fillna("")
    if "bad" not in data.columns:
        data["bad"] = False
        print("Added 'bad' column")

    data = normalize_ingredient_names(data)
    plate_control_indexes = data[data["plate_control"]].index
    plate_blank_indexes = data[data["plate_blank"]].index

    data["delta_od"] = data["final_od"] - data["initial_od"]

    leave_out_cols = [c for c in data.columns if "leave_out" in c]

    plate_controls = data.loc[plate_control_indexes, :].drop(columns=leave_out_cols)
    plate_blanks = data.loc[plate_blank_indexes, :].drop(columns=leave_out_cols)
    plate_control_means = (
        plate_controls.groupby("parent_plate").mean().to_dict()["delta_od"]
    )
    data["fitness"] = data["delta_od"] / data["parent_plate"].replace(
        plate_control_means
    )

    data = data.drop(data[(data["plate_control"] | data["plate_blank"])].index)
    data = data.drop(
        columns=[
            "plate_control",
            "plate_blank",
            "parent_well",
            "parent_well_index",
            "replicate",
            "solution_id_hex",
        ]
    )
    data_grouped = data.groupby(
        by=leave_out_cols + ["environment", "strain", "parent_plate"],
        as_index=False,
    )

    data = data_grouped.median()
    # data = data_grouped.mean()

    cols = list(ingredients) + list(data.columns)

    data = pd.concat(
        (pd.DataFrame(np.ones((data.shape[0], n_ingredients), dtype=int)), data),
        axis=1,
        ignore_index=True,
    )
    data.columns = cols
    for row_idx, row in data.iterrows():
        idxs = pd.unique([i for i in row[leave_out_cols] if i != ""])
        data.loc[row_idx, idxs] = 0

    data = data.drop(columns=leave_out_cols)
    return data, plate_controls, plate_blanks


def normalize_ingredient_names(data):
    names1 = dict(zip(AA_NAMES_TEMPEST, AA_SHORT))
    names2 = dict(zip(AA_NAMES_2, AA_SHORT))
    names3 = dict(zip(BASE_NAMES_TEMPEST, BASE_NAMES))
    name_map = {**names1, **names2, **names3}

    data = data.replace(name_map)
    data = data.rename(columns=name_map, index=name_map)
    return data


def softmax(scores, k=1):
    """
    Compute softmax with random tiebreak.
    """
    exps = np.exp(k * scores)
    if np.isinf(exps.sum()):
        # perform max if sum is infinite, split among all maxes
        row_maxes = scores.max()
        matches = scores == row_maxes
        scores = np.where(matches, 1 / matches.sum(), 0)
        return scores

    softmax_scores = exps / exps.sum()
    return softmax_scores


def decoratortimer(decimal):
    def decoratorfunction(f):
        def wrap(*args, **kwargs):
            time1 = time.monotonic()
            result = f(*args, **kwargs)
            time2 = time.monotonic()
            print(
                "{:s} function took {:.{}f} ms".format(
                    f.__name__, ((time2 - time1) * 1000.0), decimal
                )
            )
            return result

        return wrap

    return decoratorfunction


def combined_round_data(experiment_folder, max_n=None, sort=True):
    paths = []
    for root, dirs, files in os.walk(experiment_folder):
        models = []
        for name in files:
            path = os.path.join(root, name)
            if "bad_runs" in path:
                continue
            if "results_all" in name:
                paths.append(path)

    paths = sorted(paths, key=lambda x: (len(x), x), reverse=False)
    if max_n:
        paths = paths[:max_n]
    all_results = []
    for idx, path in enumerate(paths):
        print(path)
        results = normalize_ingredient_names(pd.read_csv(path, index_col=None))
        if sort:
            results = results.sort_values(by="growth_pred").reset_index(drop=True)
        if "is_redo" in results.columns:
            results = results[~results["is_redo"]]
        # results["round"] = idx + 1
        all_results.append(results)

    all_results = pd.concat(all_results, ignore_index=True)
    return all_results


if __name__ == "__main__":
    m = softmax(np.array([0.1, 0.8, 0.6, 1, 0.9]), 0.2)
    print(m, m.sum())
