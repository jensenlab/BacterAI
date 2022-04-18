import os
import sys

sys.path.append("../")
from utils import combined_round_data

path = "../experiments/08-20-2021_12"
out_path = os.path.join(path, "aggregated_data.csv")


data = combined_round_data(path, max_n=11, sort=False)
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
data = data.rename(columns={"var": "growth_pred_var", "depth": "num_aa_removed"})
print(data.columns)
print(data)

data.to_csv(out_path, index=False)