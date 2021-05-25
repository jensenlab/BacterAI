
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
        print("Added \'bad\' column")

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
    data = data.drop(columns=["plate_control", "plate_blank",  "parent_well", "parent_well_index", "replicate", "solution_id_hex"])
    data_grouped = data.groupby(
        by=leave_out_cols + ["environment", "strain", "parent_plate"],
        as_index=False,
    )

    data = data_grouped.mean()
    cols = list(AA_NAMES) + list(data.columns)
    data = pd.concat((pd.DataFrame(np.ones((data.shape[0], 20), dtype=int)), data), axis=1, ignore_index=True)
    data.columns = cols
    ingredient_locs = {name: i for i, name in enumerate(AA_NAMES)}
    for row_idx, row in data.iterrows():
        idxs = [i for i in row[leave_out_cols] if i != ""]
        data.loc[row_idx, idxs] = 0
    
    data = data.drop(columns=leave_out_cols)

    return data, plate_controls, plate_blanks


def softmax(scores):
    """
    Compute softmax with random tiebreak.
    """

    exps = np.exp(scores)
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