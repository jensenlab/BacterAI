import numpy as np

import csv

import global_vars
import utils

SEED = 0
NP_RAND_STATE = utils.seed_numpy_state(SEED)


def main(n=2240, out_name="control_run_randoms_SGO.csv"):
    choices = global_vars.AA_NAMES_TEMPEST

    chosen = set()
    skipped_count = 0
    while len(chosen) < n:
        new = tuple(NP_RAND_STATE.choice(choices, size=20, replace=True).tolist())
        if new not in chosen:
            chosen.add(new)
        else:
            skipped_count += 1

    print(f"{skipped_count=}")

    with open(out_name, "w") as f:
        writer = csv.writer(f, delimiter=",")
        for row in chosen:
            writer.writerow(sorted(row))

    print(f"File saved to: {out_name}")


if __name__ == "__main__":
    main()