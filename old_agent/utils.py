from collections import defaultdict
import csv
import itertools
import math
import os
import sys
import time

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from protocols import CDM_NH4_rescale as protocol


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
    name_mappings_csv, mapped_data_csv, components, binary_threshold=False,
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

    # plate_control_indexes = data[data["plate_control"]].index
    # data = data.drop(index=plate_control_indexes)
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
    nn_experiments = data["media"].apply(pd.Series)
    nn_experiments.columns = components
    nn_experiments["environment"] = data["environment"]
    nn_experiments["grow"] = data["delta_od_normalized"]
    nn_experiments = nn_experiments.sort_values(by="environment")
    return data, nn_experiments, data_growth


def match_original_data(original_data, new_data, new_data_labels=None):
    """Intersection of keys from both frames (inner join). 
    Set index to `original_data` index"""

    original_data.columns = new_data.columns
    new_data = new_data.merge(original_data.reset_index()).set_index("index")

    if new_data_labels is not None:
        new_data_labels = new_data_labels.set_index(new_data.index)
    return new_data, new_data_labels


def batch_to_deep_phenotyping_protocol(
    batch_name, batch, components, name_mappings_csv, development=True
):
    """Calls Deepphenotyping to create an experiment with `batch`"""
    with open(name_mappings_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)
        name_mappings = {row[1]: row[0] for row in reader}

    batch_removals = []
    components = np.array([name_mappings[c] for c in components])
    for row in batch.to_numpy():
        c = components[np.invert(row.astype(bool))].tolist()
        batch_removals.append(c)

    protocol.from_batch_list(batch_name, batch_removals, development=development)


def convex_extrapolation(
    data_filepaths, inputs_filepath, output_filepath, threshold=0.25
):
    """Take existing growth data and extrapolate those results to a set of inputs, 
    assuming metabolic convexity (i.e. if a set of media components is removed and 
    grows to a certain point, we can assume all medias with those same coomponents removed 
    result in that same growth).
    
    We define two directions to extrapolate: `up` and `down`, the 'top' being the full 
    media. The down direction should be used for data that has less components removed 
    (i.e Leave-out data). In the down direction, the matching inputs will take the data 
    that has a smaller cardinality since it has more 'metabolic information.' Matches 
    are defined as medias with the same componenents taken out. These will override the 
    default assumption of 'growth' < threshold for growth.
    
    Conversely, the up direction should be used for data that has more components 
    removed (i.e. Leave-in data). In the up direction, the matching inputs will take 
    the data that has a larger cardinality, since it has more 'metabolic information.'
    Matches are defined as medias with the same number of components left in. These will
    override the downward pass only if they are >= threshold for growth.
    
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
                "down" : [filepaths, to, data, files],
                "up" : [filepaths, to, data, files]
            }
            
    inputs_filepath: str
        Path to inputs to extrapolate data onto. Input should be a CSV
        where each row is a boolean array of media components (where columns are the
        components).
        
    output_filepath: str
        Path to save the extrapolated data to.
    
    threshold: float
        Any fitness value >= to this number constitutes growth.
    
    Outputs
    -------
    CSV file at `output_filepath`.
                    

    """
    # TODO: add support for aerobic
    inputs = pd.read_csv(inputs_filepath)
    inputs["grow"] = 1.0

    # Ensuring order: `up` direction goes first.
    data_filepaths = {
        "down": data_filepaths.get("down", []),
        "up": data_filepaths.get("up", []),
    }

    for direction, filepaths in data_filepaths.items():
        for filepath in filepaths:
            print("\n\n--------- FILE", filepath, " -----------")
            data = pd.read_csv(filepath)
            assert isinstance(data.drop(columns=["grow"]).columns, #object Quack)
            media_components = data.drop(columns=["grow"]).columns.to_list()
            n_components = len(media_components)
            data["card"] = data.iloc[:, :-1].sum(axis=1)

            # when sorted the later ones become more important, they overwrite the previous data.
            if direction == "down":
                ascending = False
            elif direction == "up":
                ascending = True

            data = data.sort_values(by=["card"], ascending=ascending)
            unique_cards = pd.unique(data["card"])
            for card in unique_cards:
                print("\n--------- CARD", card, " -----------")
                # get all combinations of media where N = # components left out from media
                removed_combos = [
                    list(c)
                    for c in itertools.combinations(
                        media_components, n_components - card
                    )
                ]
                # print("removed_combos", removed_combos)
                for c in removed_combos:
                    # get media components still remaining
                    remaining = list(set(media_components) - set(c))
                    # print("remaining", remaining)

                    # get growth result where removed components = 0, and remaining components = 1
                    grow_data = data[
                        data[c].eq(0).all(axis=1) & data[remaining].eq(1).all(axis=1)
                    ]["grow"].to_list()
                    if grow_data:
                        grow_result = grow_data[0]
                    else:
                        continue
                    # print("grow_result", grow_result)
                    if direction == "down":
                        if grow_result >= threshold:
                            # skip ones that grow
                            continue
                        # matched medias are ones with the same components removed
                        matches = inputs[c].eq(0).all(axis=1)
                        inputs.loc[matches, "grow"] = grow_result
                    elif direction == "up":
                        if grow_result <= threshold:
                            # skip ones that don't grow
                            continue
                        # matched medias are ones with the same components remaining
                        matches = inputs[remaining].eq(1).all(axis=1)
                        # inputs.loc[matches, "grow"] = grow_result
                        # only apply growth if it is larger than the current growth value
                        inputs.loc[
                            (matches) & (inputs["grow"] < grow_result), "grow"
                        ] = grow_result
                    # inputs.loc[matches, "grow"] = grow_result

                    # print("matches", matches)

    inputs.to_csv(output_filepath, index=False)


def create_fractional_factorial_experiment(design_filepath, hyperparams_filepath):
    """Uses a fractional factorial design methods to create an experiment.

    Inputs
    ------
    design_filepath: str
        Path to design, where each row is an experiment, the columns are the variables,
        and the values are -1 or 1 (low/high).
            
    hyperparams_filepath: str
        Path to hyperparameters, where each row is low/high values, the columns are the variables, 
        and each value is the corresponding value to be used in the experiment.
    
    Returns
    -------
    design_true: pd.DataFrame 
        Where each row is an experiment, with the columns being the name of the 
        input parameters, and the values being the actual parameter values.

    design: pd.DataFrame
        Where each row is an experiment, with the columns being the name of the 
        input parameters, and the values being the -1 or 1 (low/high) designation.
    """
    params = pd.read_csv(hyperparams_filepath, index_col=0)
    param_names = params.columns.to_list()
    params = params.to_dict()

    design = pd.read_csv(design_filepath, index_col=0)
    if len(param_names) != len(design.columns):
        print(f"Length mismatch: {len(param_names)} vs. {len(design.columns)}")
        return
    design.columns = param_names
    design_true = design.copy()
    for param, values in params.items():
        design_true.loc[design_true[param] == 1, param] = values[1]
        design_true.loc[design_true[param] == -1, param] = values[-1]
    return design_true, design


def tensorflow_summary_writers_to_csv(path_dir):
    # path_dir is a list to all the paths I should accumulate
    final_out = {}
    for dname in os.listdir(path_dir):
        print(f"Converting run {dname}", end="")
        ea = EventAccumulator(os.path.join(path_dir, dname)).Reload()
        tags = ea.Tags()["scalars"]
        print(tags)
        out = {}

        for tag in tags:
            tag_values = []
            wall_time = []
            steps = []

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)

            out[tag] = pd.DataFrame(
                data=dict(zip(steps, np.array([tag_values, wall_time]).transpose())),
                columns=steps,
                index=["value", "wall_time"],
            )

        if len(tags) > 0:
            df = pd.concat(out.values(), keys=out.keys())
            df.to_csv(f"tensorboard_logs/converted/{dname}.csv")
            final_out[dname] = df
            print(" - Done")
        else:
            print(" - No scalers to write")
    return final_out


def data_subset(data_path, save_path, p=0.05):
    """Creates a subset of the original data from `data_path` taking `p` 
    percentage of the orignal data. Saves new file to `save_path`."""

    data = pd.read_csv(data_path, index_col=None)
    num_to_take = int(p * data.shape[0])
    indexes = np.random.choice(
        data.index.to_list(), size=num_to_take, replace=False,
    ).tolist()
    data = data.loc[indexes, :]
    data.to_csv(save_path, index=False)


def add_feature_columns(data_path, features_list_path, has_grow=True):
    """Adds feature columns from `features_list_path` to the original data 
    from `data_path`, saving it as a new file in the same location as 
    the '[original name]_with_features'. The row names in the features list must
    match the columns in the original data. The original data is assumed to have a 
    `grow` column unless you specify `False`."""

    data = pd.read_csv(data_path, index_col=None)
    features = pd.read_csv(features_list_path, index_col=0)
    feature_names = features.columns.to_list()

    for col in feature_names:
        data.insert(data.shape[1] - int(has_grow), col, 0)

    for col_name, col in data.items():
        if col_name in feature_names or col_name == "grow":
            continue
        for feature_name in feature_names:
            value = features.loc[col_name, feature_name]
            if value == 1:
                data.loc[data[col_name] == 1, feature_name] = 1
    orig_save_path = os.path.dirname(data_path)
    orig_file_name = os.path.basename(data_path)
    new_file_name = (
        os.path.splitext(orig_file_name)[0]
        + "_with_features"
        + os.path.splitext(orig_file_name)[1]
    )
    new_save_path = os.path.join(orig_save_path, new_file_name)
    data.to_csv(new_save_path, index=False)


def seed_numpy_state(seed):
    return np.random.RandomState(np.random.MT19937(np.random.SeedSequence(seed)))


def numpy_state_int(state):
    return state.randint(2 ** 32 - 1)


def normalize_dict_values(d):
    total = sum(d.values())
    if total == 0:
        return d
    d = {key: value / total for key, value in d.items()}
    return d


if __name__ == "__main__":

    file_names = [
        "data/L1L2IO-Rand-Tempest-SMU/MappedData/L1IO-L2IO-Rand SMU UA159 (3)_mapped.csv",
        "data/L1L2IO-Rand-Tempest-SMU/MappedData/L3O-Rand SMU UA159 aerobic_mapped.csv",
        "data/L1L2IO-Rand-Tempest-SMU/MappedData/Randoms SMU UA159 aerobic (1) mapped.csv",
        "data/L1L2IO-Rand-Tempest-SMU/MappedData/Randoms SMU UA159 aerobic (2) mapped .csv",
        "data/L1L2IO-Rand-Tempest-SMU/MappedData/Randoms SMU UA159 aerobic (3) mapped .csv",
        "data/L1L2IO-Rand-Tempest-SMU/MappedData/Randoms SMU UA159 aerobic (4) mapped .csv",
        "data/L1L2IO-Rand-Tempest-SMU/MappedData/Randoms SMU UA159 aerobic (5) mapped .csv",
        "data/L1L2IO-Rand-Tempest-SMU/MappedData/Randoms SMU UA159 aerobic (6) mapped .csv",
        "data/L1L2IO-Rand-Tempest-SMU/MappedData/Randoms SMU UA159 aerobic (7) mapped (removed plate 1).csv",
    ]
    components = [
        "ala",
        "arg",
        "asn",
        "asp",
        "cys",
        "glu",
        "gln",
        "gly",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "pro",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
    ]
    dfs = []
    for f in file_names:
        _, nn_data, _ = parse_data_map("files/name_mappings_both.csv", f, components)

        dfs.append(nn_data)

    dfs = pd.concat(dfs)
    dfs = dfs.sort_values(by="environment")
    dfs = dfs[dfs["environment"] == "anaerobic"]
    dfs.to_csv(
        # "data/L1L2IO-Rand-Tempest-SMU/L1IO-L2IO-All Rands SMU UA159 Processed.csv",
        "data/L1L2IO-Rand-Tempest-SMU/L1IO-L2IO-L3O-All Rands SMU UA159 Processed-anaerobic.csv",
        index=False,
    )

    # convex_extrapolation(
    #     {
    #         "down": [
    #             "data/tweaked_agent_learning_policy/initial_data/data_20_extrapolated_001.csv"
    #         ],
    #         # "up": ["data/iSMU-test/initial_data/train_set_L1IL2I.csv"],
    #     },
    #     "models/iSMU-test/data_20_clean.csv",
    #     "data/tweaked_agent_learning_policy/initial_data/data_20_extrapolated_001_extrapolated.csv",
    # )

    # convex_extrapolation(
    #     {
    #         "down": ["data/iSMU-test/initial_data/train_set_L1OL2O.csv"],
    #         # "up": ["data/iSMU-test/initial_data/train_set_L1IL2I.csv"],
    #     },
    #     "models/iSMU-test/data_20_clean.csv",
    #     "models/iSMU-test/data_20_extrapolated_LO_only.csv",
    # )

    # create_fractional_factorial_experiment(
    #     "files/fractional_design_k10n128.csv", "files/hyperparameters.csv"
    # )

    # out = tensorflow_summary_writers_to_csv(
    #     "tensorboard_logs/fractional_factorial_results_100000_50split/20200410-153117-1/"
    # )
    # print(out)

    # steps = tabulate_events(path)
    # pd.concat(steps.values(),keys=steps.keys()).to_csv('all_result.csv')
    # data_subset(
    #     "data/tweaked_agent_learning_policy/initial_data/data_20_extrapolated.csv",
    #     "data/tweaked_agent_learning_policy/initial_data/data_20_extrapolated_001.csv",
    #     p=0.001,
    # )

    # add_feature_columns(
    #     data_path="data/tweaked_agent_learning_policy/initial_data/data_20_extrapolated.csv",
    #     features_list_path="files/amino_acid_features.csv",
    # )
