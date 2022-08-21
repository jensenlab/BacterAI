import os
import csv

dir = "experiments/2022-04-18_25/rule_results"
results = {g: {} for g in range(4, 21)}
for file in os.listdir(dir):
    if not file.endswith("txt"):
        continue

    prefix, groups, rounds = file.split("-")
    n_groups = int(groups.split("_")[0])
    n_rounds = int(rounds.split("_")[0])

    file_path = os.path.join(dir, file)
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = lines[:6]
        lines = "".join(lines).replace("\t", "").replace("\n", "\\r")

    print()
    print(file, n_groups, n_rounds)
    print(lines)
    results[n_groups][n_rounds] = lines

from pprint import pprint

pprint(results)

output_file_path = os.path.join(dir, "output.csv")
with open(output_file_path, "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    for r in range(4, 10):
        line = [results[g][r] for g in range(4, 21)]
        writer.writerow(line)
