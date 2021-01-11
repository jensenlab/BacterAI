import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

names = [
    "ala_exch",
    "arg_exch",
    "asn_exch",
    "asp_exch",
    "cys_exch",
    "glu_exch",
    "gln_exch",
    "gly_exch",
    "his_exch",
    "ile_exch",
    "leu_exch",
    "lys_exch",
    "met_exch",
    "phe_exch",
    "pro_exch",
    "ser_exch",
    "thr_exch",
    "trp_exch",
    "tyr_exch",
    "val_exch",
]


def issubset(a, b):
    """Return whether sequence `a` is a subset of sequence `b`"""
    difference = a - b
    difference[difference <= 0] = 0
    n_non_zeros = np.count_nonzero(difference)
    return n_non_zeros == 0


# convex: all growing subsets of a media that grows should be <= growth of set

# oracle_data = pd.read_csv("oracle_net_fitness_distribution.csv", index_col=0)
# oracle_data = oracle_data.drop(
#     columns=["strain", "parent_plate", "delta_od_std", "environment"]
# )
oracle_data = pd.read_csv("models/iSMU-test/data_20_extrapolated.csv", index_col=None)
indexes = np.random.choice(oracle_data.index, 5000, replace=False)
oracle_data = oracle_data.loc[indexes, :]


oracle_data = oracle_data.sort_values(by=names, ascending=False)
oracle_data = oracle_data.reset_index(drop=True)
len1 = len(oracle_data)

oracle_data.drop_duplicates(keep=False, inplace=True)

len2 = len(oracle_data)
print(len1, len2)

nonconvex_counts = {i: 0 for i in range(1, 21)}
nonconvex_growths = {i: [] for i in range(1, 21)}
prev_growth = None
skips = set()

for idx, row in oracle_data.iterrows():

    print(idx)
    prev_inputs = row.values[:20]
    prev_growth = row.values[-1]
    n_components = prev_inputs.sum()
    if idx in skips:
        print("skipping", idx)
        continue
    for idx2, row2 in oracle_data.iterrows():
        if idx2 <= idx:
            continue

        inputs = row2.values[:20]
        growth = row2.values[-1]

        is_subset = issubset(inputs, prev_inputs)
        is_nonconvex = is_subset and growth > prev_growth

        if is_nonconvex:
            nonconvex_counts[n_components] += 1
            nonconvex_growths[n_components].append(growth - prev_growth)
            skips.add(idx2)
            print("skips", skips)
            print("### Nonconvex:")
            print("prev:", prev_inputs, prev_growth)
            print("curr:", inputs, growth)
            break

print(nonconvex_counts)
for k, v in nonconvex_growths.items():
    v = np.array(v)
    print(f"{k} components:")
    print(f"\tmean: {np.mean(v)}")
    print(f"\tmedian: {np.median(v)}")

# results 5000:
# {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 10, 7: 61, 8: 157, 9: 346, 10: 714, 11: 1350, 12: 1761, 13: 1804, 14: 1769, 15: 755, 16: 761, 17: 349, 18: 0, 19: 0, 20: 0}
def nCr(n, r):
    f = math.factorial
    return f(n) / f(r) / f(n - r)


# nonconvex_counts = {
#     1: 0,
#     2: 0,
#     3: 0,
#     4: 0,
#     5: 1,
#     6: 7,
#     7: 41,
#     8: 102,
#     9: 207,
#     10: 358,
#     11: 433,
#     12: 370,
#     13: 266,
#     14: 177,
#     15: 66,
#     16: 18,
#     17: 7,
#     18: 0,
#     19: 0,
#     20: 0,
# }  # 5000 datapoints

normalized_results = {k: v / (nCr(20, k)) for k, v in nonconvex_counts.items()}
## try to normalize to number of experiments in cardinality
print(nonconvex_counts)
print(sum(list(nonconvex_counts.values())))

fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=(6, 10))

x_range = range(1, 21)
axs[0].plot(x_range, list(nonconvex_counts.values()))

ax2 = axs[0].twinx()
ax2.set_ylabel("n non-convex (normalized)", color="g")
for tl in ax2.get_yticklabels():
    tl.set_color("g")
ax2.plot(x_range, list(normalized_results.values()), "g-")

axs[0].set_xlabel("n_components")
axs[0].set_ylabel("n non-convex")
axs[0].set_xticks(x_range)

means = [np.mean(v) for k, v in nonconvex_growths.items()]
medians = [np.median(v) for k, v in nonconvex_growths.items()]
# means = [
#     None,
#     None,
#     None,
#     None,
#     0.025978920000000016,
#     0.0957370885714286,
#     0.0820058131707317,
#     0.10209014421568628,
#     0.09811842526570048,
#     0.10017133578212291,
#     0.09632747120092379,
#     0.07908491675675676,
#     0.07094994229323308,
#     0.04757036677966101,
#     0.029542378939393932,
#     0.04246838333333335,
#     0.03911459571428571,
#     None,
#     None,
#     None,
# ]
# medians = [
#     None,
#     None,
#     None,
#     None,
#     0.025978920000000016,
#     0.08642931000000001,
#     0.05621879999999996,
#     0.08402979499999999,
#     0.07816445999999996,
#     0.07591154,
#     0.07019429999999999,
#     0.046872175,
#     0.035185345000000035,
#     0.01438598999999996,
#     0.010027129999999995,
#     0.014460899999999943,
#     0.0017505599999999566,
#     None,
#     None,
#     None,
# ]


axs[1].set_xlabel("n_components")

axs[1].plot(x_range, means, x_range, medians)
axs[1].legend(["means", "medians"])
axs[1].set_xticks(x_range)


plt.tight_layout()
plt.savefig("assess_dataset_complexity.png")
