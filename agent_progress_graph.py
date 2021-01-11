import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

main_folder = "data/"
# main_folder = "data/agent_state_save_ROLL1"
# main_folder = "data/agent_state_save_GF1"
expt_names = [
    # "agent_state_save_testing",
    # "agent_state_save_ROLL1_2",
    # "agent_state_save_ROLL1_3",
    # "agent_state_save_GF1",
    # "agent_state_save_GF1_2",
    # "agent_state_save_fixed_reinforce",
    "agent_state_save_fixed_reinforce4",
    # "data/agent_state_save_GF1_2",
]

folders = [os.path.join("data", f) for f in expt_names]
line_colors = iter(["r", "g", "b", "c", "m", "y", "k"])
line_styles = []
plt.figure(figsize=(12, 6))
for i, folder in enumerate(folders):
    all_old_policies = []
    states = []
    for folder_name in sorted(os.listdir(folder), key=lambda x: int(x.split("_")[-1])):
        summary_info = os.path.join(folder, folder_name, "summary_info.txt")
        if not os.path.exists(summary_info):
            continue

        with open(summary_info, "r") as f:
            lines = f.readlines()
            new_policy = [
                float(lines[1].split(":")[1]),
                float(lines[2].split(":")[1]),
                float(lines[3].split(":")[1]),
                float(lines[4].split(":")[1]),
            ]  # , float(lines[2].split(":")[1])]
            old_policy = [
                float(lines[6].split(":")[1]),
                float(lines[7].split(":")[1]),
                float(lines[8].split(":")[1]),
                float(lines[9].split(":")[1]),
            ]  # , float(lines[5].split(":")[1])]

            final_states = lines[11][1:].strip().strip("][").split(", ")

            # final_card = lines[9][1:]
            # print(folder_name)
            # # print(new_policy)
            # print(old_policy)
            # print()
            all_old_policies.append(old_policy)
            states.append(final_states)

    all_old_policies = np.array(all_old_policies)
    length = all_old_policies.shape[0]
    its = range(0, length)
    print("policy history:")
    print(all_old_policies)
    print()
    print("policy deltas:")
    for x in range(length - 1):
        print(all_old_policies[x + 1, :] - all_old_policies[x, :])
    print()

    # r = all_old_policies[:, 0]
    # for x in range(length - 1):
    #     print(r[x + 1] - r[x])

    states = np.array(states).astype(int)
    print("final states:")
    print(states)
    print(states.sum(axis=1))
    print()

    color = next(line_colors)

    # plt.plot(
    #     its, all_old_policies[:, 0], "--", its, all_old_policies[:, 1], "-", c=color,
    # )
    plt.plot(its, all_old_policies[:, 0], "--", c=color)
    plt.plot(its, all_old_policies[:, 1], "-", c=color)
    plt.plot(its, all_old_policies[:, 2], "-.", c=color)
    plt.plot(its, all_old_policies[:, 3], ":", c=color)

    line_styles.append(Line2D([0], [0], color=color, label=expt_names[i]))


# plt.legend(["avg_rollout", "gf_1"] * len(folders))

legend_elements = [
    Line2D([0], [0], color="black", linestyle="--", label="lambda - mu rollout"),
    Line2D([0], [0], color="black", linestyle="-", label="beta 1 - removal_progress"),
    Line2D([0], [0], color="black", linestyle="-.", label="beta 2 - growth"),
    Line2D([0], [0], color="black", linestyle=":", label="beta 3 - agreement"),
]


# plt.legend(handles=legend_elements)
legend1 = plt.legend(
    handles=legend_elements, bbox_to_anchor=(1, 1), loc="upper left", ncol=1
)
plt.legend(handles=line_styles, bbox_to_anchor=(1, 0.85), loc="upper left", ncol=1)
plt.gca().add_artist(legend1)

plt.ylabel("param value")
plt.xlabel("policy iteration")
plt.tight_layout()
plt.savefig(f"agent_progress_graph.png")


# final states results (unclipped using dist)
# [3 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 2 3 2 2 2 3 2 2 2 2 2 2 2 3 2 2 2 2 3 2 3 2 2 3 3 2 2 2 2 3 2 2 3 2 2 2 2 2]
