import os
import sys

import pandas as pd
import numpy as np

sys.path.append("../")
import utils
import global_vars



def process_mapped_data(path, ingredients):
    """Processes DeepPhenotyping data. It normalizes the
    change in OD (delta OD) to their plate controls' mean delta OD.
    """

    n_ingredients = len(ingredients)
    data = pd.read_csv(path, index_col=None).fillna("")
    if "bad" not in data.columns:
        data["bad"] = False

    data = utils.normalize_ingredient_names(data)
    plate_control_indexes = data[data["plate_control"]].index
    plate_blank_indexes = data[data["plate_blank"]].index

    data["delta_od"] = data["final_od"] - data["initial_od"]

    leave_out_cols = [c for c in data.columns if "leave_out" in c]

    plate_controls = data.loc[plate_control_indexes, :].drop(columns=leave_out_cols)
    plate_blanks = data.loc[plate_blank_indexes, :].drop(columns=leave_out_cols)
    plate_control_means = (
        plate_controls.groupby("parent_plate").agg(
            delta_od_mean=('delta_od', np.mean),
        ).to_dict()["delta_od_mean"]
    )
    data["fitness"] = data["delta_od"] / data["parent_plate"].replace(
        plate_control_means
    )
    data = data.drop(data[data["plate_control"] | data["plate_blank"]].index)
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

    data = data_grouped.agg(
        fitness_mean=('fitness', np.mean),
        fitness_std=('fitness', np.std),
        fitness_median=('fitness', np.median),
    )

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
    return data


def main(max_day, ingredient_names, experiment_path):

    full_cdm = len(ingredient_names) > 20

    # data = import.combined_round_data(path, max_n=11, sort=False)
    round_folders = sorted([f for f in os.listdir(experiment_path) if "Round" in f], key=lambda x: (len(x), x))[:max_day]
    print(round_folders)

    all_round_data = []
    for folder_name in round_folders:
        folder_path = os.path.join(experiment_path, folder_name)
        folder_content = os.listdir(folder_path)
        
        if full_cdm and folder_name == 'Round1':
            ingredient_names = global_vars.BASE_NAMES

        n_ingredients = len(ingredient_names)

        for i in folder_content:
            if (
                "bad_runs" in i or 
                "redo_dp" in i.lower() or 
                "redo_meta" in i.lower() or 
                "Redo Plate" in i
            ):  
                continue
            elif "mapped_data" in i:
                mapped_path = os.path.join(folder_path, i)
            elif "batch_meta" in i and "results" not in i:
                batch_path = os.path.join(folder_path, i)

        # Merge results (mapped data) with predictions + experiment metadata (batch data)
        data = process_mapped_data(mapped_path, ingredient_names)
        batch_df = utils.normalize_ingredient_names(pd.read_csv(batch_path, index_col=None))
        results = pd.merge(
            batch_df,
            data,
            how="left",
            left_on=ingredient_names,
            right_on=ingredient_names,
            sort=True,
        )
        results = results[~results['is_redo']]
        if full_cdm and folder_name == 'Round1':
            ingredient_names = global_vars.AA_SHORT + global_vars.BASE_NAMES
            ones = pd.DataFrame(np.ones((len(data), 20)), columns=global_vars.AA_SHORT)
            results = pd.concat([ones, results], axis=1)

        results[results.columns[:n_ingredients]] = results[results.columns[:n_ingredients]].astype(int)
        print(folder_name, len(results))
        all_round_data.append(results)

    data = pd.concat(all_round_data)
    data = data.reset_index(drop=True)

    # Pad with rest of CDM's ingredients
    if not full_cdm:
        ones = pd.DataFrame(np.ones((len(data), 19)), columns=global_vars.BASE_NAMES)
        data = pd.concat([data.iloc[:, :n_ingredients], ones, data.iloc[:, n_ingredients:]], axis=1)

    data[data.columns[:39]] = data[data.columns[:39]].astype(int)
    data["ingredients_in_media_count"] = data.iloc[:, :39].sum(axis=1).astype(int)

    data = data.drop(
        columns=[
            "direction",
            "is_redo",
            "parent_plate",
            "environment",
            "strain",
            "var"
        ]
    )
    if 'ammoniums_50g/l' in data.columns:
        data = data.drop(columns=['ammoniums_50g/l'])

    data = data.sort_values(['round', 'fitness_mean'])

    if full_cdm:
        data['fitness_std'] = 'N/A'

    print(list(data.columns))
    print(data.shape)
    # print(data)
    print(max_day * 336)

    out_path = os.path.join(experiment_path, "experiment_data.csv")
    data.to_csv(out_path, index=False)

if __name__ == '__main__':
    experiments = [
        {
            "max_day": 13,
            "ingredient_names": global_vars.AA_SHORT,
            "experiment_path": "../experiments/2021-07-26_10",
        },
        {
            "max_day": 11,
            "ingredient_names": global_vars.AA_SHORT,
            "experiment_path": "../experiments/2021-08-20_12",
        },
        {
            "max_day": 4,
            "ingredient_names": global_vars.AA_SHORT,
            "experiment_path": "../experiments/2022-01-17_19",
        },
        {
            "max_day": 3,
            "ingredient_names": global_vars.AA_SHORT,
            "experiment_path": "../experiments/2022-02-08_24",
        },
        {
            "max_day": 7,
            "ingredient_names": global_vars.AA_SHORT + global_vars.BASE_NAMES,
            "experiment_path": "../experiments/2022-04-18_25",
        },
    ]
    for e in experiments:
        print(e["experiment_path"])
        main(**e)