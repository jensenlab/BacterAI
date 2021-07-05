import datetime
import os
import json
import statistics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# main_folder = "data/experiment-2021-02-14T02:00:39.072492"
# exp_folder = "experiment-2021-03-05T16:02:08.194718"
folders = [
    f
    for f in os.listdir("data/agent_logs")
    if os.path.isdir(os.path.join("data/agent_logs", f))
]

exp_folder = sorted(
    folders, key=lambda x: datetime.datetime.fromisoformat(x.split("experiment-")[1])
)[-1]

main_folder = os.path.join("data/agent_logs", exp_folder)
# main_folder = os.path.join("data", exp_folder)
expt_names = [
    f
    for f in sorted(os.listdir(main_folder))
    if os.path.isdir(os.path.join(main_folder, f))
]
print(expt_names)
folders = [os.path.join(main_folder, fold) for fold in expt_names]
line_colors = iter(["r", "g", "b", "c", "m", "y", "k"])
line_styles = []
fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(16, 10))

for i, folder in enumerate(folders):
    new_policies = []
    policy_deltas = []
    final_cardinalities = []
    mean_eval_reward = []
    for folder_name in sorted(os.listdir(folder), key=lambda x: int(x.split("_")[-1])):
        if "eval" not in folder_name:
            summary_info = os.path.join(folder, folder_name, "summary_info.json")
            with open(summary_info, "r") as f:
                data = json.load(f)
                policy = list(data["new_policy"].values())
                policy_delta = list(data["policy_delta"].values())
                media_cardinalities = data["media_cardinalities"][0]

                new_policies.append(policy)
                policy_deltas.append(policy_delta)
                final_cardinalities.append(media_cardinalities)
        else:
            eval_info = os.path.join(folder, folder_name, "summary_info_eval.json")
            with open(eval_info, "r") as f:
                data = json.load(f)

                r = []
                for eval_n, cards in data.items():
                    r += cards

                mean_eval_reward.append(statistics.mean(r))

    new_policies = np.array(new_policies)
    policy_deltas = np.array(policy_deltas)
    final_reward = 20 - np.array(final_cardinalities).astype(int)
    mean_eval_reward = np.array(mean_eval_reward)

    print()
    print("New new_policies:", new_policies)
    print("policy_deltas:", policy_deltas)
    print("final_reward:", final_reward)
    print("mean_eval_reward:", mean_eval_reward)

    length = new_policies.shape[0]
    x_range = range(0, length)
    color = next(line_colors)

    axs[0, 0].plot(x_range, new_policies[:, 0], "--", c=color)
    axs[0, 0].plot(x_range, new_policies[:, 1], "-", c=color)
    axs[0, 0].plot(x_range, new_policies[:, 2], "-.", c=color)
    axs[0, 0].plot(x_range, new_policies[:, 3], ":", c=color)

    axs[1, 0].plot(x_range, policy_deltas[:, 0], "--", c=color)
    axs[1, 0].plot(x_range, policy_deltas[:, 1], "-", c=color)
    axs[1, 0].plot(x_range, policy_deltas[:, 2], "-.", c=color)
    axs[1, 0].plot(x_range, policy_deltas[:, 3], ":", c=color)

    n_evals = len(mean_eval_reward)
    skip = int(len(new_policies) / n_evals) if n_evals else 0
    if skip > 0:
        axs[1, 1].plot(
            range(0, len(mean_eval_reward) * skip, skip), mean_eval_reward, "-", c=color
        )
    axs[1, 1].plot(x_range, final_reward, "-.", c=color)

    line_styles.append(Line2D([0], [0], color=color, label=expt_names[i]))


# plt.legend(["avg_rollout", "gf_1"] * len(folders))

legend_elements = [
    Line2D([0], [0], color="black", linestyle="--", label="lambda - mu rollout"),
    Line2D([0], [0], color="black", linestyle="-", label="beta 1 - removal_progress"),
    Line2D([0], [0], color="black", linestyle="-.", label="beta 2 - growth"),
    Line2D([0], [0], color="black", linestyle=":", label="beta 3 - agreement"),
]


# plt.legend(handles=legend_elements)
legend1 = axs[0, 0].legend(
    handles=legend_elements, bbox_to_anchor=(1, 1), loc="upper left", ncol=1
)
axs[0, 0].legend(
    handles=line_styles, bbox_to_anchor=(1, 0.75), loc="upper left", ncol=1
)
axs[0, 0].add_artist(legend1)

axs[0, 0].set_ylabel("Parameter Values")
axs[0, 0].set_xlabel("Policy Iteration")

axs[1, 0].set_ylabel("Parameter Deltas")
axs[1, 0].set_xlabel("Policy Iteration")

axs[1, 1].set_ylabel("Mean Eval Reward")
axs[1, 1].set_xlabel("Policy Iteration")
axs[1, 1].set_yticks(range(0, 21))

axs[0, 1].remove()
plt.tight_layout()
plt.savefig(f"agent_progress_graph-{exp_folder}.png")


# axs[0].plot(x_range, list(nonconvex_counts.values()))

# ax2 = axs[0].twinx()
# ax2.set_ylabel("n non-convex (normalized)", color="g")
# for tl in ax2.get_yticklabels():
#     tl.set_color("g")
# ax2.plot(x_range, list(normalized_results.values()), "g-")

# axs[0].set_xlabel("n_components")
# axs[0].set_ylabel("n non-convex")
# axs[0].set_xticks(x_range)
#
