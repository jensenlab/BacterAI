import pandas as pd
import numpy as np
import os
import tensorflow as tf

 import neural
 import utils

############ Analyzing which ones grew ###############

batch, batch_labels = utils.parse_data_map(
    "files/name_mappings_aa_alt.csv",
    "SPSAvFDSA_expt/mapped_data_preexpt_fdsa_v_spsa.csv",
    [
        "ala_exch",
        "arg_exch",
        "asp_exch",
        "asn_exch",
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
        "cys_exch",
    ],
    # [
    #     "ala_exch",
    #     "gly_exch",
    #     "arg_exch",
    #     "asn_exch",
    #     "asp_exch",
    #     "cys_exch",
    #     "glu_exch",
    #     "gln_exch",
    #     "his_exch",
    #     "ile_exch",
    #     "leu_exch",
    #     "lys_exch",
    #     "met_exch",
    #     "phe_exch",
    #     "ser_exch",
    #     "thr_exch",
    #     "trp_exch",
    #     "tyr_exch",
    #     "val_exch",
    #     "pro_exch",
    # ],
)

batch["fitness"] = batch_labels
batch = batch[batch.fitness >= 0.50]
batch.to_csv("SPSAvFDSA_expt/preexpt_fdsa_v_spsa_candidates.csv")

# batch = batch.drop(columns=["aerobic"])
# batch["grow"] = batch_labels
# # batch["card"] = batch.sum(axis=1)
# # batch = batch[batch["card"] <= 2]
# # batch = batch.drop(columns=["aerobic", "card"])

# batch = batch.groupby(
#     by=[
#         "ala_exch",
#         "gly_exch",
#         "arg_exch",
#         "asn_exch",
#         "asp_exch",
#         "cys_exch",
#         "glu_exch",
#         "gln_exch",
#         "his_exch",
#         "ile_exch",
#         "leu_exch",
#         "lys_exch",
#         "met_exch",
#         "phe_exch",
#         "ser_exch",
#         "thr_exch",
#         "trp_exch",
#         "tyr_exch",
#         "val_exch",
#         "pro_exch",
#     ],
#     as_index=False,
# ).mean()

# cutoff = 0.25
# batch = batch[batch["grow"] >= cutoff]
# batch = batch.sort_values(by=["card", "grow"], ascending=[True, False])
# batch.to_csv("train_set_L1IL2I.csv")

# ############ Comparing L1O/L2O trained to L1O only trained AI batches #############3
# batch_12 = pd.read_csv("data/iSMU-022720/batches/batch_C1.csv", index_col=0)
# batch_1 = pd.read_csv("data/iSMU-022720_binary_L2O/batches/batch_C1.csv", index_col=0)

# data_12 = pd.read_csv("analyze_cycle.csv", index_col=0)
# og_data = pd.read_csv("models/iSMU-022720/data_20.csv")
# batch, labels = utils.match_original_data(
#     og_data.loc[:, "ala_exch":"pro_exch"],
#     data_12.loc[:, "ala_exch":"pro_exch"],
#     data_12.loc[:, "grow"].to_frame(),
# )

# batch["grow"] = labels

# print(batch)
# filtered = set(batch_12.index.to_list()).intersection(batch_1.index.to_list())
# print(len(batch_12.index))
# print(len(filtered), sorted(list(filtered)))

# filtered_data = set(batch.index.to_list()).intersection(batch_1.index.to_list())
# print(len(filtered_data), sorted(list(filtered_data)))

# batch = batch.loc[filtered_data]
# batch["card"] = batch.drop(columns=["grow"]).sum(axis=1)
# batch = batch.sort_values(by=["card", "grow"], ascending=[True, False])
# batch.to_csv("L2O_match_binary.csv")
# # batch.to_csv("L1O_match.csv")


############ Analyzing NN performance ###############


def analyze(name):
    df = pd.read_csv("media_combos.csv")
    df.columns = [
        "ala",
        "gly",
        "arg",
        "asn",
        "asp",
        "cys",
        "glu",
        "gln",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
        "pro",
    ]
    df["card"] = df.sum(axis=1)
    df["pred"] = pd.read_csv("predictions.csv")
    s = df["pred"].sum()
    print(f"G:\t{s}/{df.shape[0]} \t- {s/df.shape[0]}")
    print(f"NG:\t{df.shape[0]-s}/{df.shape[0]} \t- {(df.shape[0] - s)/df.shape[0]}")

    stats = []
    for x in range(0, 21):
        print(f"#### CARDINALITY ({x})####")
        sub = df[df["card"] == x]
        s = sub["pred"].sum()
        print(f"G:\t{s}/{sub.shape[0]} \t- {s/sub.shape[0]}")
        print(
            f"NG:\t{sub.shape[0]-s}/{sub.shape[0]} \t- {(sub.shape[0] - s)/sub.shape[0]}"
        )
        stats.append((s, sub.shape[0]))

    out = pd.DataFrame(stats)
    out.columns = ["pred true", "total num"]
    out["decimal"] = out["pred true"] / out["total num"]
    out.to_csv(f"nn_distribution_{name}.csv")

    # batch, batch_labels = utils.parse_data_map(
    #     "files/name_mappings_aa.csv",
    #     "data/iSMU-test/initial_data/bacterAI_SMU_C1.csv",
    #     [
    #         "ala_exch",
    #         "gly_exch",
    #         "arg_exch",
    #         "asn_exch",
    #         "asp_exch",
    #         "cys_exch",
    #         "glu_exch",
    #         "gln_exch",
    #         "his_exch",
    #         "ile_exch",
    #         "leu_exch",
    #         "lys_exch",
    #         "met_exch",
    #         "phe_exch",
    #         "ser_exch",
    #         "thr_exch",
    #         "trp_exch",
    #         "tyr_exch",
    #         "val_exch",
    #         "pro_exch",
    #     ],
    #     binary_threshold=0.25,
    # )

    # batch = batch.drop(columns=["aerobic"])
    # batch["grow"] = batch_labels

    # batch = batch.groupby(
    #     by=[
    #         "ala_exch",
    #         "gly_exch",
    #         "arg_exch",
    #         "asn_exch",
    #         "asp_exch",
    #         "cys_exch",
    #         "glu_exch",
    #         "gln_exch",
    #         "his_exch",
    #         "ile_exch",
    #         "leu_exch",
    #         "lys_exch",
    #         "met_exch",
    #         "phe_exch",
    #         "ser_exch",
    #         "thr_exch",
    #         "trp_exch",
    #         "tyr_exch",
    #         "val_exch",
    #         "pro_exch",
    #     ],
    #     as_index=False,
    # ).mean()
    # print(batch)
    # batch.to_csv("test_set_L1IL2I.csv")

    # batch["card"] = batch.sum(axis=1)
    batch = pd.read_csv("models/iSMU-test/data_20_extrapolated.csv")
    batch["grow"] = batch["grow"] >= 0.25
    no_grows = batch[batch["grow"] == 0]
    grows = batch[batch["grow"] == 1]
    # n_nogrows_true = no_grows.shape[0]
    # n_grows_true = grows.shape[0]

    # print("DF\n", df)
    # print("BATCH NG\n", no_grows)
    # print("BATCH G\n", grows)

    df_ng, _ = utils.match_original_data(
        df.loc[:, "ala":"pro"], no_grows.loc[:, "ala_exch":"pro_exch"],
    )

    df_g, _ = utils.match_original_data(
        df.loc[:, "ala":"pro"], grows.loc[:, "ala_exch":"pro_exch"],
    )

    ng_pred = df.loc[df_ng.index, "pred"]
    tn = ng_pred[ng_pred == 0].shape[0]
    fp = ng_pred[ng_pred == 1].shape[0]

    g_pred = df.loc[df_g.index, "pred"]
    fn = g_pred[g_pred == 0].shape[0]
    tp = g_pred[g_pred == 1].shape[0]

    print("TN:", tn, tn / (fn + tn))
    print("FN:", fn, fn / (fn + tn))
    print("TP:", tp, tp / (fp + tp))
    print("FP:", fp, fp / (fp + tp))
    print("ACCURACY", (tp + tn) / (tp + fp + tn + fn))

    df_out = pd.DataFrame(
        [
            ["True negative", f"{tn}", tn * 100 / (fn + tn),],
            ["False negative", f"{fn}", fn * 100 / (fn + tn),],
            ["False positive", f"{fp}", fp * 100 / (fp + tp),],
            ["True positive", f"{tp}", tp * 100 / (fp + tp),],
        ]
    )
    df_out.columns = ["", "proportion", "%"]
    df_out.to_csv(f"nn_performance_{name}.csv")


# if __name__ == "__main__":
#     analyze("tweaked_agent_learning_policy")


def num_grow():
    data = pd.read_csv("models/iSMU-test/data_20_extrapolated_LO_only.csv")

    cutoff = 0.25
    g = data[data["grow"] >= cutoff]
    ng = data[data["grow"] < cutoff]
    print("GROWS", g.shape[0])
    print("NO GROWS", ng.shape[0])


# num_grow()
# def filter_LO_LI():
#     data = pd.read_csv("data/iSMU-test/initial_data/train_set_L1OL2OL1IL2I.csv")

#     cards = data.iloc[:, :-1].sum(axis=1)

#     L_in = data[cards <= 2]
#     L_in.to_csv(
#         "data/tweaked_agent_learning_policy/initial_data/train_set_in_real.csv", index=False
#     )

#     L_out = data[cards >= 18]
#     L_out.to_csv(
#         "data/tweaked_agent_learning_policy/initial_data/train_set_out_real.csv",
#         index=False,
#     )

#     both = data.loc[(cards >= 18) | (cards <= 2)]
#     both.to_csv(
#         "data/tweaked_agent_learning_policy/initial_data/train_set_both_real.csv",
#         index=False,
#     )

#     ### Making data subsets
#     data = pd.read_csv("models/iSMU-test/data_20_extrapolated.csv")
#     for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.75]:
#         indexes = np.random.choice(
#             data.index.to_list(), size=int(p * len(data.index.to_list())), replace=False
#         )
#         data.loc[indexes, :].to_csv(
#             f"data/tweaked_agent_learning_policy/initial_data/data_20_extrapolated_{int(p*100)}.csv",
#             index=False,
#         )


def analyze_batch_cardinalities(path):
    data = pd.read_csv(path, index_col=0)
    data["card"] = data.sum(axis=1)
    all_cards = set(data["card"].to_list())
    for c in all_cards:
        print(f"Cardinality: {c} -> {data.loc[data['card'] == c, :].shape[0]}")


# analyze_batch_cardinalities("data/tweaked_agent_learning_policy/batches/batch_C1.csv")


def card_distribution(save_location):
    # import agent

    # agent_cont = agent.Agent.load(
    #     agent_path=os.path.join(save_location, f"agents/agent_state_C{cycle_n}.pkl"),
    #     predictor_path=os.path.join(save_location, f"neural_nets/NN_20_C{cycle_n}.h5"),
    # )

    predictor = neural.PredictNet.from_save(
        "data/neuralpy_optimization_expts/052220-sparcity-3/working_model/"
    )
    # predictor = neural.PredictNet(
    #     exp_id="1",
    #     n_test=10000,
    #     parent_logdir="tensorboard_logs/misc",
    #     save_model_path="data/neuralpy_optimization_expts/052220-sparcity-3/working_model/",
    # )
    # predictor.model = tf.keras.models.load_model(
    #     os.path.join(
    #         save_location,
    #         f"data/neuralpy_optimization_expts/052220-sparcity-3/working_model/model.h5",
    #     )
    # )
    data = pd.read_csv(
        "data/tweaked_agent_learning_policy/initial_data/data_20_extrapolated.csv",
    )

    # x, y = neural.load_data(
    #     filepath=f"models/iSMU-test/data_20_extrapolated.csv", starting_index=0,
    # )
    # from sklearn.model_selection import train_test_split

    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.50)
    # agent.predictor.train(x_train, y_train, x_test, y_test)

    predictions = predictor.predict_class(data.to_numpy()[:, :-1])
    data["pred"] = predictions

    data["card"] = data.drop(columns=["grow", "pred"]).sum(axis=1)

    data["grow_correct"] = np.where(
        (data["grow"] >= 0.25) & (data["pred"] == 1), True, False,
    )
    data["no_grow_correct"] = np.where(
        (data["grow"] < 0.25) & (data["pred"] == 0), True, False,
    )
    print(data)

    g = data[data["pred"] == 1].shape[0]
    g_correct = data["grow_correct"].sum()
    ng = data[data["pred"] == 0].shape[0]
    ng_correct = data["no_grow_correct"].sum()

    print(f"GROWS: {g / (ng +g)} \t- {g_correct/g}")
    print(f"NO GROWS: {ng / (ng +g)} \t- {ng_correct/ng}",)
    print(f"Accuracy: {(g_correct + ng_correct)/(g + ng)}",)

    for x in range(1, 21):
        print(f"#### CARDINALITY ({x})####")
        sub = data[data["card"] == x]
        g_correct = sub["grow_correct"].sum()
        ng_correct = sub["no_grow_correct"].sum()
        s = sub["pred"].sum()
        n = sub.shape[0]
        print(f"G:\t{s}/{n} \t- {s/n} \t- {g_correct/s}")
        print(f"NG:\t{n-s}/{n} \t- {(n - s)/n} \t- {ng_correct/(n-s)}")


# card_distribution("data/tweaked_agent_learning_policy")
