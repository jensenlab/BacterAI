# import csv

# count = 0
# with open(
#     "experiments/04-05-2021/Round3/batch_gpr_meta_2021-04-26T16.15.44.023964.csv", "r"
# ) as f:

#     reader = csv.reader(f, delimiter=",")
#     x = set()
#     for r in reader:
#         k = tuple(sorted(r))
#         if k in x:
#             print(r)
#             count += 1
#         else:
#             x.add(k)

# print(count)

import pandas as pd

# file = "experiments/04-30-2021/both/Round6/GPR SMU UA159 (3R6) 148d mapped_data.csv"
# # file = "experiments/04-05-2021/Round8/GPR SMU UA159 (1R8) 6b68 mapped_data.csv"

# df = pd.read_csv(file, index_col=None)

# rows = "ABCDEFGHIJKLMNOP"
# cols = list(range(10, 25))
# print(cols)

# bad_wells = []
# for r in rows:
#     for c in cols:
#         bad_wells.append(r + str(c))

# bad_idx = df["parent_well"].isin(bad_wells) & df["parent_plate"].str.contains("Plate 4")
# df["bad"] = False
# df.loc[bad_idx, "bad"] = True
# print(df)

# df.to_csv(file, index=False)


file = "experiments/04-07-2021/Round10/gpr_train_pred.csv"
df = pd.read_csv(file, index_col=None)
unique = set(map(tuple, df.to_numpy()[:, :20]))
print(len(unique))
print(len(df))
print(len(unique) / len(df))
