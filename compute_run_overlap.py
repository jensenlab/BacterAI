
import utils

# path_1 = "experiments/07-26-2021_10"
# path_2 = "experiments/07-26-2021_11"
path_1 = "experiments/08-20-2021_12"
path_2 = "experiments/08-20-2021_13"
max_round = 11

data_raw_1 = utils.combined_round_data(path_1, max_round)
data_raw_2 = utils.combined_round_data(path_2, max_round)

data_1 = set()
for row in data_raw_1.values[:, :20]:
    row = tuple(list(row))
    data_1.add(row)

data_2 = set()
for row in data_raw_2.values[:, :20]:
    row = tuple(list(row))
    data_2.add(row)

print(len(data_1))
print(len(data_2))

print(len(data_1.intersection(data_2)))