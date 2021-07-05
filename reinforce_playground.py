rewards = [0] * 17 + [14]
T = len(rewards)  # length of episode
gamma = 0.99
n_removable_ingredients = 12  # or 18
gJ = 0  # np.zeros(len(policy))
for t in range(0, T):
    print()
    reward = rewards[t]

    baseline = n_removable_ingredients * (gamma ** (n_removable_ingredients - 1))
    print(
        f"t: {t}\tT: {T}\tbaseline: {baseline}\tN_removable: {n_removable_ingredients}"
    )

    # compute discounted return with baseline
    all_returns = [rewards[t_i] * (gamma ** (t_i - t)) for t_i in range(t, T)]
    sum_returns = sum(all_returns)
    score = sum_returns - baseline
    print(f"reward: {reward}\tsum: {sum_returns}\tscore: {score}")
    print(f"all_returns: {all_returns}")
