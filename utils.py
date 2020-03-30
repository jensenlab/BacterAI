import csv
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd

from protocols import CDM_NH4_rescale as protocol


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


def parse_data_map(
    name_mappings_csv, mapped_data_csv, components, binary_threshold=False
):
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
        by=leave_out_cols + ["environment", "strain", "parent_plate"], as_index=False,
    ).mean()
    plate_control_indexes = data[data["plate_control"]].index
    data = data.drop(index=plate_control_indexes)
    data["delta_od_normalized"] = data["delta_od"] / data["parent_plate"].replace(
        plate_control_means
    )

    data = data.groupby(
        by=leave_out_cols + ["environment", "strain"], as_index=False,
    ).mean()

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

    if isinstance(binary_threshold, float):
        data.loc[
            data["delta_od_normalized"] >= binary_threshold, "delta_od_normalized"
        ] = 1.0
        data.loc[
            data["delta_od_normalized"] < binary_threshold, "delta_od_normalized"
        ] = 0.0
    else:
        data.loc[data["delta_od_normalized"] >= 1, "delta_od_normalized"] = 1.0
        data.loc[data["delta_od_normalized"] < 0, "delta_od_normalized"] = 0.0

    data_growth = data["delta_od_normalized"].to_frame()
    experiments = data["media"].apply(pd.Series)
    experiments.columns = components
    experiments["aerobic"] = data["environment"]
    return experiments, data_growth


def match_original_data(original_data, new_data, new_data_labels=None):
    """Intersection of keys from both frames (inner join). 
    Set index to `original_data` index"""

    new_data.columns = [f"col_{n}" for n in range(new_data.shape[1])]
    original_data.columns = [f"col_{n}" for n in range(original_data.shape[1])]
    new_data = new_data.merge(original_data.reset_index()).set_index("index")

    if new_data_labels is not None:
        new_data_labels = new_data_labels.set_index(new_data.index)
        return new_data, new_data_labels
    return new_data


def batch_to_deep_phenotyping_protocol(
    batch_name, batch, components, name_mappings_csv, development=True
):
    """Calls Deepphenotyping to create an experiment with `batch`"""
    with open(name_mappings_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)
        name_mappings = {row[1]: row[0] for row in reader}

    batch_removals = list()
    components = np.array([name_mappings[c] for c in components])
    for row in batch.to_numpy():
        c = components[np.invert(row.astype(bool))].tolist()
        batch_removals.append(c)

    protocol.from_batch_list(batch_name, batch_removals, development=development)


def convex_extrapolation(data_filepaths, inputs_filepath, output_filepath):
    """Take existing growth data and extrapolate those results to a set of inputs, 
    assuming metabolic convexity (i.e. if a set of media components is removed and 
    grows to a certain point, we can assume all medias with those same coomponents removed 
    result in that same growth).
    
    We define two directions to extrapolate: `up` and `down`. The up direction should
    be used for data that has less components removed (i.e Leave-out data). In the up 
    direction, the matching inputs will take the data that has a smaller cardinality since
    it has more 'metabolic information.' Conversely, the down direction should be used
    for data that has more components removed (i.e. Leave-in data). In the down direction,
    the matching inputs will take the data that has a larger cardinality, for the same 
    reason as above.
    
    Inputs
    ------
    data_filepaths: dict(str: List(str))
        Paths to the data files used which contain the growth data to be extrapolated.
        Order the lists from least importance to most importance. Data will from the 
        more important lists will trump the 'lesser' data. Data files should be CSVs 
        where each row is a boolean array of media components (where columns are the
        components) with a column 'grow' with growth data for that media config.
        Formatted in this way:
            {
                "up" : [filepaths, to, data, files],
                "down" : [filepaths, to, data, files]
            }
            
    inputs_filepath: str
        Path to inputs to extrapolate data onto. Input should be a CSV
        where each row is a boolean array of media components (where columns are the
        components).
        
    output_filepath: str
        Path to save the extrapolated data to.
    
    Outputs
    -------
    CSV file at `output_filepath`.
                    

    """
    # TODO: add support for aerobic
    inputs = pd.read_csv(inputs_filepath)
    inputs["grow"] = 1

    # Ensuring order: `up` direction goes first.
    data_filepaths = {
        "up": data_filepaths.get("up", []),
        "down": data_filepaths.get("down", []),
    }

    for direction, filepaths in data_filepaths.items():
        for filepath in filepaths:
            data = pd.read_csv(filepath)
            media_components = data.drop(columns=["grow"]).columns.to_list()
            n_components = len(media_components)
            data["card"] = data.iloc[:, :-1].sum(axis=1)

            if direction == "up":
                ascending = False
            elif direction == "down":
                ascending = True
            data = data.sort_values(by=["card"], ascending=ascending)

            unique_cards = pd.unique(data["card"])
            for card in unique_cards:
                combos = [
                    list(c)
                    for c in itertools.combinations(
                        media_components, n_components - card
                    )
                ]
                for c in combos:
                    grow_result = data[data[c].eq(0).all(axis=1)]["grow"].to_list()[0]
                    matches = inputs[c].eq(0).all(1)
                    inputs.loc[matches, "grow"] = grow_result

    inputs.to_csv(output_filepath)


if __name__ == "__main__":
    # components = [
    #     "ala_exch",
    #     "gly_exch",
    #     "arg_exch",
    #     "asn_exch",
    #     "asp_exch",
    #     "cys_exch",
    #     "gln_exch",
    #     "his_exch",
    #     "ile_exch",
    #     "leu_exch",
    #     "lys_exch",
    #     "met_exch",
    #     "phe_exch",
    #     "ser_exch",
    #     "thr_exch",
    #     "trp_exch",
    #     "tyr_exch",
    #     "val_exch",
    #     "pro_exch",
    # ]
    # parse_data_map(
    #     "files/name_mappings_aa.csv",
    #     "/home/lab/Downloads/mapped_data_SGO_2.csv",
    #     components,
    # )

    convex_extrapolation(
        {
            "up": ["data/iSMU-test/initial_data/train_set_L1OL2O.csv"],
            "down": ["data/iSMU-test/initial_data/train_set_L1IL2I.csv"],
        },
        "models/iSMU-test/data_20_clean.csv",
        "models/iSMU-test/data_20_extrapolated.csv",
    )
