import pandas as pd
import utils

############ Analyzing which ones grew ###############

batch, batch_labels = utils.parse_data_map(
    "files/name_mappings_aa.csv",
    "data/iSMU-test/initial_data/bacterAI_SMU_C1.csv",
    [
        "ala_exch",
        "gly_exch",
        "arg_exch",
        "asn_exch",
        "asp_exch",
        "cys_exch",
        "glu_exch",
        "gln_exch",
        "his_exch",
        "ile_exch",
        "leu_exch",
        "lys_exch",
        "met_exch",
        "phe_exch",
        "ser_exch",
        "thr_exch",
        "trp_exch",
        "tyr_exch",
        "val_exch",
        "pro_exch",
    ],
)


batch = batch.drop(columns=["aerobic"])
batch["grow"] = batch_labels
# batch["card"] = batch.sum(axis=1)
# batch = batch[batch["card"] <= 2]
# batch = batch.drop(columns=["aerobic", "card"])

batch = batch.groupby(
    by=[
        "ala_exch",
        "gly_exch",
        "arg_exch",
        "asn_exch",
        "asp_exch",
        "cys_exch",
        "glu_exch",
        "gln_exch",
        "his_exch",
        "ile_exch",
        "leu_exch",
        "lys_exch",
        "met_exch",
        "phe_exch",
        "ser_exch",
        "thr_exch",
        "trp_exch",
        "tyr_exch",
        "val_exch",
        "pro_exch",
    ],
    as_index=False,
).mean()

# cutoff = 0.25
# batch = batch[batch["grow"] >= cutoff]
# batch = batch.sort_values(by=["card", "grow"], ascending=[True, False])
batch.to_csv("train_set_L1IL2I.csv")

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


# ############ Analyzing NN performance ###############


# def analyze(name):
#     df = pd.read_csv("media_combos.csv")
#     df.columns = [
#         "ala",
#         "gly",
#         "arg",
#         "asn",
#         "asp",
#         "cys",
#         "glu",
#         "gln",
#         "his",
#         "ile",
#         "leu",
#         "lys",
#         "met",
#         "phe",
#         "ser",
#         "thr",
#         "trp",
#         "tyr",
#         "val",
#         "pro",
#     ]
#     df["card"] = df.sum(axis=1)
#     df["pred"] = pd.read_csv("predictions.csv")
#     s = df["pred"].sum()
#     print(f"G:\t{s}/{df.shape[0]} \t- {s/df.shape[0]}")
#     print(f"NG:\t{df.shape[0]-s}/{df.shape[0]} \t- {(df.shape[0] - s)/df.shape[0]}")

#     stats = list()
#     for x in range(0, 21):
#         print(f"#### CARDINALITY ({x})####")
#         sub = df[df["card"] == x]
#         s = sub["pred"].sum()
#         print(f"G:\t{s}/{sub.shape[0]} \t- {s/sub.shape[0]}")
#         print(
#             f"NG:\t{sub.shape[0]-s}/{sub.shape[0]} \t- {(sub.shape[0] - s)/sub.shape[0]}"
#         )
#         stats.append((s, sub.shape[0]))

#     out = pd.DataFrame(stats)
#     out.columns = ["pred true", "total num"]
#     out["decimal"] = out["pred true"] / out["total num"]
#     out.to_csv(f"nn_distribution_{name}.csv")

#     batch, batch_labels = utils.parse_data_map(
#         "files/name_mappings_aa.csv",
#         "data/iSMU-test/initial_data/bacterAI_SMU_C1.csv",
#         [
#             "ala_exch",
#             "gly_exch",
#             "arg_exch",
#             "asn_exch",
#             "asp_exch",
#             "cys_exch",
#             "glu_exch",
#             "gln_exch",
#             "his_exch",
#             "ile_exch",
#             "leu_exch",
#             "lys_exch",
#             "met_exch",
#             "phe_exch",
#             "ser_exch",
#             "thr_exch",
#             "trp_exch",
#             "tyr_exch",
#             "val_exch",
#             "pro_exch",
#         ],
#         binary_threshold=0.25,
#     )

#     batch = batch.drop(columns=["aerobic"])
#     batch["grow"] = batch_labels

#     batch = batch.groupby(
#         by=[
#             "ala_exch",
#             "gly_exch",
#             "arg_exch",
#             "asn_exch",
#             "asp_exch",
#             "cys_exch",
#             "glu_exch",
#             "gln_exch",
#             "his_exch",
#             "ile_exch",
#             "leu_exch",
#             "lys_exch",
#             "met_exch",
#             "phe_exch",
#             "ser_exch",
#             "thr_exch",
#             "trp_exch",
#             "tyr_exch",
#             "val_exch",
#             "pro_exch",
#         ],
#         as_index=False,
#     ).mean()
#     print(batch)
#     # batch.to_csv("test_set_L1IL2I.csv")

#     # batch["card"] = batch.sum(axis=1)

#     no_grows = batch[batch["grow"] == 0]
#     grows = batch[batch["grow"] == 1]
#     n_nogrows_true = no_grows.shape[0]
#     n_grows_true = grows.shape[0]

#     # print("DF\n", df)
#     # print("BATCH NG\n", no_grows)
#     # print("BATCH G\n", grows)

#     df_ng = utils.match_original_data(
#         df.loc[:, "ala":"pro"], no_grows.loc[:, "ala_exch":"pro_exch"],
#     )

#     df_g = utils.match_original_data(
#         df.loc[:, "ala":"pro"], grows.loc[:, "ala_exch":"pro_exch"],
#     )

#     ng_pred = df.loc[df_ng.index, "pred"]
#     ng_pred_0 = ng_pred[ng_pred == 0].shape[0]
#     ng_pred_1 = ng_pred[ng_pred == 1].shape[0]

#     g_pred = df.loc[df_g.index, "pred"]
#     g_pred_0 = g_pred[g_pred == 0].shape[0]
#     g_pred_1 = g_pred[g_pred == 1].shape[0]

#     print("TN", ng_pred_0, n_nogrows_true, ng_pred_0 / n_nogrows_true)
#     print("FN", ng_pred_1, n_nogrows_true, ng_pred_1 / n_nogrows_true)
#     print("TP", g_pred_1, n_grows_true, g_pred_1 / n_grows_true)
#     print("FP", g_pred_0, n_grows_true, g_pred_0 / n_grows_true)

#     df_out = pd.DataFrame(
#         [
#             [
#                 "True negative",
#                 f"{ng_pred_0}/{n_nogrows_true}",
#                 ng_pred_0 * 100 / n_nogrows_true,
#             ],
#             [
#                 "False negative",
#                 f"{ng_pred_1}/{n_nogrows_true}",
#                 ng_pred_1 * 100 / n_nogrows_true,
#             ],
#             [
#                 "False positive",
#                 f"{g_pred_0}/{n_grows_true}",
#                 g_pred_0 * 100 / n_grows_true,
#             ],
#             [
#                 "True positive",
#                 f"{g_pred_1}/{n_grows_true}",
#                 g_pred_1 * 100 / n_grows_true,
#             ],
#         ]
#     )
#     df_out.columns = ["", "proportion", "%"]
#     df_out.to_csv(f"nn_performance_{name}.csv")


# if __name__ == "__main__":
#     analyze("solo")
