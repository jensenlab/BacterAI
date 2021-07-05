import time

import numpy as np
import pandas as pd

from global_vars import *


def process_mapped_data(path):
    """Processes DeepPhenotyping data. It normalizes the
    change in OD (delta OD) to their plate controls' mean delta OD.
    """

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

    cols = list(AA_SHORT) + list(data.columns)

    data = pd.concat(
        (pd.DataFrame(np.ones((data.shape[0], 20), dtype=int)), data),
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
    names1 = {AA_NAMES_TEMPEST[i]: AA_SHORT[i] for i in range(len(AA_NAMES_TEMPEST))}
    names2 = {AA_NAMES_2[i]: AA_SHORT[i] for i in range(len(AA_NAMES_2))}
    name_map = dict(names1, **names2)

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


if __name__ == "__main__":
    m = softmax(np.array([0.1, 0.8, 0.6, 1, 0.9]), 0.2)
    print(m, m.sum())
