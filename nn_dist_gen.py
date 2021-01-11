import pandas as pd
import numpy as np
import neural_pretrain as neural


n = neural.PredictNet.from_save("models/SMU_NN_oracle")

data = np.random.choice([0, 1], size=(5000, 20))
sums = 20 - np.sum(data, axis=1)
grow = n.predict_probability(data)

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
df = pd.DataFrame(data, columns=names)
df["n_components_removed"] = sums
df["delta_od_mean"] = grow
df["strain"] = "NN"
df["parent_plate"] = "n/a"
df["delta_od_std"] = 0
df["environment"] = "sim"

df.to_csv("oracle_net_fitness_distribution.csv")
print(df)
