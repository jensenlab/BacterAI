import pandas as pd
import numpy as np


def fill_new_ingredients(data, original_size, fill_column_names, fill_on_right=True):
    n_new = len(fill_column_names)
    fill_df = pd.DataFrame(
        np.ones((len(data), n_new), dtype=int), columns=fill_column_names
    )

    if fill_on_right:
        left = data.iloc[:, :original_size]
        right = data.iloc[:, original_size:]
        center = fill_df
        dfs = (left, center, right)
        col_names = list(left.columns) + fill_column_names + list(right.columns)
    else:
        dfs = (fill_df, data)
        col_names = fill_column_names + list(data.columns)

    data = pd.concat(dfs, axis=1, ignore_index=True)
    data.columns = col_names
    return data


if __name__ == "__main__":
    from ..global_vars import BASE_NAMES
    from ..utils import combined_round_data

    def import_data(experiment_path, max_n=None):
        data = combined_round_data(experiment_path, max_n=max_n, sort=False)
        data = data.drop(
            columns=[
                "direction",
                "is_redo",
                "parent_plate",
                "initial_od",
                "final_od",
                "bad",
                "delta_od",
            ]
        )
        data = data.rename(
            columns={"var": "growth_pred_var", "depth": "num_aa_removed"}
        )
        return data

    path = "../experiments/08-20-2021_12"

    data = import_data(path)
    new_ingredients = BASE_NAMES
    data_filled = fill_new_ingredients(
        data, original_size=20, fill_column_names=new_ingredients, fill_on_right=True
    )

    # data_filled_2 = fill_new_ingredients(data, 20, new_ingredients, fill_on_right=False)

    print(data_filled.columns)
    print(data_filled)
