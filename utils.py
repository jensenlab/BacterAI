import csv
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

from protocols import CDM


def get_LXO(n_reagents, X=1):
    # n_reactions - int: number of reactions
    # X - int: number to leave out for leave-X-out experiments

    all_indexes = np.arange(n_reagents)
    combos = itertools.combinations(all_indexes, X)
    remove_indexes = [list(c) for c in combos]
    remove_arrs = np.empty((len(remove_indexes), n_reagents))
    for i, to_remove in enumerate(remove_indexes):
        remove_arr = np.ones(n_reagents)
        remove_arr[to_remove] = 0
        remove_arrs[i, :] = remove_arr
    return remove_arrs


def parse_data_map(name_mappings_csv, mapped_data_csv, components):
    """Processes DeepPhenotyping data for use in BacterAI framework. It normalizes the 
    change in OD (delta OD) to their plate controls' mean delta OD. It parses the 
    `leave_out` columns into a representative binary np.array of components for that 
    experiment.
    """
    with open(name_mappings_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)
        name_mappings = {row[0]: row[1] for row in reader}

    component_locations = {c: idx for idx, c in enumerate(components)}

    data = pd.read_csv(mapped_data_csv, header=0).fillna("")
    plate_control_indexes = data[data["plate_control"]].index
    plate_blank_indexes = data[data["plate_blank"]].index

    data = data.drop(index=plate_blank_indexes)

    data["delta_od"] = data["final_od"] - data["initial_od"]

    leave_out_cols = [c for c in data.columns if "leave_out" in c]

    plate_controls = data.loc[plate_control_indexes, :].drop(columns=leave_out_cols)
    plate_control_means = (
        plate_controls.groupby("parent_plate").mean().to_dict()["delta_od"]
    )

    data = data.groupby(
        # by=leave_out_cols + ["environment", "strain", "parent_plate"], as_index=False
        by=leave_out_cols + ["environment", "strain", "parent_plate"],
        as_index=False,
    ).mean()

    data = data.drop(index=plate_control_indexes)
    data["delta_od_normalized"] = data["delta_od"] / data["parent_plate"].replace(
        plate_control_means
    )

    def _f(row):
        media = np.ones(len(components)).astype(int)
        for c in leave_out_cols:
            if row[c] == "":
                continue
            cleaned_name = row[c].split(" ")[0]  # removes any '(n)' that are present
            idx = component_locations[name_mappings[cleaned_name]]
            media[idx] = 0
        return media

    data["media"] = data.apply(_f, axis=1)
    data = data.drop(
        columns=["initial_od", "final_od", "plate_blank", "plate_control", "replicate",]
    )

    growth_cutoff = 0.25
    data.loc[data["delta_od_normalized"] >= growth_cutoff, "delta_od_normalized"] = 1.0
    data.loc[data["delta_od_normalized"] < growth_cutoff, "delta_od_normalized"] = 0.0

    print(data)
    data_growth = data["delta_od_normalized"].to_frame()
    data = data["media"].apply(pd.Series)
    print(data)
    print(data, data_growth)

    return data, data_growth


def match_original_data(original_data, new_data, new_data_labels=None):
    new_data = new_data.rename(
        columns={n: f"col_{n}" for n in new_data.columns.to_list()}
    )
    original_data = original_data.rename(
        columns={n: f"col_{n}" for n in original_data.columns.to_list()}
    )

    new_data = new_data.merge(original_data.reset_index()).set_index("index")
    if new_data_labels is not None:
        new_data_labels = new_data_labels.set_index(new_data.index)
        return new_data, new_data_labels
    return new_data


def batch_to_deep_phenotyping_protocol(
    batch_name, batch, components, name_mappings_csv
):
    with open(name_mappings_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)
        name_mappings = {row[1]: row[0] for row in reader}

    batch_removals = list()
    components = np.array([name_mappings[c] for c in components])
    for row in batch.to_numpy():
        c = components[np.invert(row.astype(bool))].tolist()
        batch_removals.append(c)

    CDM.from_batch_list(batch_name, batch_removals)


if __name__ == "__main__":
    components = [
        "ala_exch",
        "gly_exch",
        "arg_exch",
        "asn_exch",
        "asp_exch",
        "cys_exch",
        "glu_exch",
        "gln_exch",
        "his_exch",
        "ile_exch",
        "leu_exch",
        "lys_exch",
        "met_exch",
        "phe_exch",
        "ser_exch",
        "thr_exch",
        "trp_exch",
        "tyr_exch",
        "val_exch",
        "pro_exch",
    ]
    parse_data_map(
        "files/name_mappings_aa.csv",
        "/home/lab/Downloads/mapped_data_SGO_2.csv",
        components,
    )

