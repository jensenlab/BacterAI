import numpy as np
np.random.seed(0)

import csv

import constants
import utils

SEED = 0
NP_RAND_STATE = utils.seed_numpy_state(SEED)


def main(n=1120, out_name=f"control_run_randoms_SGO_seed{SEED}.csv"):
    choice_map = {i: ingred for i, ingred in enumerate(constants.AA_NAMES_TEMPEST)}

    chosen = set()
    skipped_count = 0

    mapped_chosen = []
    while len(chosen) < n:
        new = tuple(NP_RAND_STATE.choice([0, 1], size=20, replace=True).tolist())
        if new not in chosen:
            chosen.add(new)

            media = []
            for i, x in enumerate(new):
                if x == 1:
                    media.append(choice_map[i])
            mapped_chosen.append(media)
        else:
            skipped_count += 1


    print(f"{skipped_count=}")

    with open(out_name, "w") as f:
        writer = csv.writer(f, delimiter=",")
        for row in mapped_chosen:
            writer.writerow(sorted(row))

    print(f"File saved to: {out_name}")


if __name__ == "__main__":
    main()