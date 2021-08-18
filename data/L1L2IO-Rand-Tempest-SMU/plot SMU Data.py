import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# df1 = pd.read_csv("L1IO-L2IO-Rand SMU UA159 Processed-1.csv")
# df1["sum"] = df1.iloc[:, :-2].sum(axis=1)
# df1 = df1[df1["sum"] <= 2]
# df1 = df1.drop(columns="sum")
# columns = df1.columns[:-2].values.tolist()
# df1 = df1.sort_values(by=columns)
# df1 = df1.reset_index()

# df2 = pd.read_csv("L1IO-L2IO-Rand SMU UA159 Processed-2.csv")
# df2["sum"] = df2.iloc[:, :-2].sum(axis=1)
# df2 = df2[df2["sum"] <= 2]
# df2 = df2.drop(columns="sum")
# df2 = df2.sort_values(by=columns)
# df2 = df2.reset_index()
# x = []
# y = []
# for row_idx, row_df in df1.iterrows():
#     g1 = row_df["grow"]
#     g2 = df2.iloc[row_idx, -1]
#     x.append(g1)
#     y.append(g2)
#     expt1 = tuple(row_df.iloc[1:-2].values)
#     expt2 = tuple(df2.iloc[row_idx, 1:-2].values)
#     s1 = sum(expt1)
#     s2 = sum(expt2)
#     # if s1 > 2:
#     #     continue
#     print(expt1 == expt2)
#     print(expt1, expt2)
#     # s = 20 - sum(expt)
#     # if s !=

# plt.scatter(x, y, s=2, alpha=0.15)
# # plt.set_xticks(ticks=range(0, 21))
# plt.xlabel("1")
# plt.ylabel("2")
# plt.savefig("SMU_L1IO-L2IO-Rand_compare.png")


df = pd.read_csv("L1IO-L2IO-L3O-All Rands SMU UA159 Processed-aerobic.csv")
all_data = set()
dup = 0

x = []
y = []
grows = {x: 0 for x in range(0, 21)}
no_grows = {x: 0 for x in range(0, 21)}

l1ins = []
for row_idx, row_df in df.iterrows():
    expt = tuple(row_df.iloc[:-2].values)
    s = 20 - sum(expt)
    # if d in all_data:
    #     print("Duplicate:", d)
    #     dup += 1
    # else:
    #     all_data.add(d)
    g = row_df["grow"]
    if g >= 0.25:
        grows[s] += 1
        if s == 19:
            l1ins.append(row_df.to_frame().T)
    else:
        no_grows[s] += 1
    x.append(s)
    y.append(g)

l1ins = pd.concat(l1ins)
print(l1ins)
l1ins.to_csv("l1ins.csv", index=None)
percents_x = []
percents_y = []

for (x1, y1), (x2, y2) in zip(grows.items(), no_grows.items()):
    if y1 + y2 == 0:
        yt = 0
    else:
        yt = y1 / (y1 + y2)
    percents_x.append(x1)
    percents_y.append(yt)


# y = np.array(y)
# x = np.array(x)
# bins_x = [x / 20 for x in list(range(0, 21))]
# bins_y = list(range(0, 21))
# heatmap, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_x])
# print(bins_y)
# print(bins_x)
# print(heatmap)
# extent = [xedges.min(), xedges.max(), yedges.min(), yedges.max()]

# plt.clf()
# plt.imshow(heatmap.T, extent=extent, origin="lower")
# plt.savefig("SMU.png")
fig, ax = plt.subplots()


# hist_y, bins = np.histogram(y, bins=100)
# bin_centers = 0.5 * (bins[1:] + bins[:-1])
ax2 = ax.twinx()

ax2.set_ylabel("% grow (>- 0.25)", color="r")
for tl in ax2.get_yticklabels():
    tl.set_color("r")

ax2.plot(percents_x, percents_y, "r-")

ax.scatter(x, y, s=2, alpha=0.05)
ax.set_xticks(ticks=range(0, 21))
ax.set_xlabel("N leave outs")
ax.set_ylabel("Fitness")
plt.savefig("SMU.png")
print(dup)
