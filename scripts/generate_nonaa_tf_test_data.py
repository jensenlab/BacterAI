import sys
import numpy as np
import pandas as pd
from torch import rand

sys.path.append("../")
from global_vars import *


## USER INPUTS

META_DATA_OUTPUT = "../experiments/08-20-2021_12_TL_testing/Round2/batch_meta_2022-04-15T02.28.59.138356-2.csv"
MAPPED_DATA_OUTPUT = "../experiments/08-20-2021_12_TL_testing/Round2/BacterAI SSA SK36 (12R1) 036a mapped_data-2.csv"
INGREDIENTS = AA_NAMES_TEMPEST + BASE_NAMES_TEMPEST


##
meta_template = "../files/tf_test_data_template_batch_meta.csv"
mapped_data_template = "../files/tf_test_data_template_mapped_data.csv"

rows_of_blanks = 144

base_col_name = "base_solution"
base_solution = "CDM_base"
leave_out_pattern = "leave_out_"


# Round 1 data
meta_template = pd.read_csv(meta_template, index_col=None)
rand_inputs = np.random.choice(
    (0, 1), (len(meta_template), len(INGREDIENTS)), replace=True
)
inputs = pd.DataFrame(rand_inputs, columns=INGREDIENTS)

meta_test = pd.concat((inputs, meta_template), axis=1)
print(f"{inputs=}")
print(f"{meta_test=}")


max_lo = 0
for row in rand_inputs:
    n_lo = len(row[row == 0])
    max_lo = max(n_lo, max_lo)

print(f"{max_lo=}")

rows = []
for _ in range(rows_of_blanks):
    blanks = [base_solution] + [""] * max_lo
    rows.append(blanks)

np_ing = np.array(INGREDIENTS)
for row in rand_inputs:
    leave_outs = np_ing[row == 0]
    leave_outs = leave_outs.tolist()
    blanks_needed = max_lo - len(leave_outs)
    blanks = [""] * blanks_needed
    leave_outs = [base_solution] + leave_outs + blanks
    for _ in range(3):
        rows.append(leave_outs)

col_names = [base_col_name] + [f"{leave_out_pattern}{i+1}" for i in range(max_lo)]
mapped_data_test = pd.DataFrame(rows, columns=col_names)
mapped_data_template = pd.read_csv(mapped_data_template, index_col=None)
mapped_data_test = pd.concat((mapped_data_test, mapped_data_template), axis=1)
print(f"{mapped_data_test=}")


meta_test.to_csv(META_DATA_OUTPUT, index=False)
mapped_data_test.to_csv(MAPPED_DATA_OUTPUT, index=False)