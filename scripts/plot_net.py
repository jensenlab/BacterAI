PATH = "model.pkl"
BATCH_SIZE = 360
EPOCHS = 50
LR = 0.001

if __name__ == "__main__":
    # # data_path = "data/L1IO-L2IO-L3O All Rands SMU UA159 Processed-Aerobic.csv"
    # splits = [0.05, 0.10, 0.25, 0.50]  # [0.01, 0.05, 0.10, 0.25, 0.50]
    # n_bags = 25
    # n_boost = 50
    # bag_proportions = [0.05, 0.10, 0.25, 0.50, 1.00]
    # boost_proportions = [0.01, 0.05, 0.10, 0.25, 0.50]

    train_split = 0.25
    train_path = f"GPRvNN_train_pred_{train_split:.2f}.csv"
    train_set = pd.read_csv(train_path, index_col=None)
    X_train = train_set.iloc[:, :20].to_numpy()
    y_train_true = train_set.loc[:, "y_true"].to_numpy()
    model_path_folder = "bag_models"
    n_bags = 25
    bag_proportion = 1.0
    # models = train_bagged(
    #     X_train,
    #     y_train_true,
    #     model_path_folder,
    #     n_bags,
    #     bag_proportion,
    #     EPOCHS,
    #     BATCH_SIZE,
    #     LR,
    # )

    test_path = f"GPRvNN_test_pred_{train_split:.2f}.csv"
    test_set = pd.read_csv(test_path, index_col=None)
    X_test = test_set.iloc[:, :20].to_numpy()
    y_test_true = test_set.loc[:, "y_true"].to_numpy()
    preds, variances = eval_bagged(X_test, models)

    print(preds)
    print(variances)
    mean_squared_error(preds, y_test_true)

    # for idx, train_split in enumerate(splits):
    #     print(f"SPLIT {train_split:.2f}")
    #     test_path = f"GPRvNN_test_pred_{train_split:.2f}.csv"
    #     train_path = f"GPRvNN_train_pred_{train_split:.2f}.csv"
    #     test_set = pd.read_csv(test_path, index_col=None)
    #     train_set = pd.read_csv(train_path, index_col=None)

    #     X_test = test_set.iloc[:, :20].to_numpy()
    #     y_test_true = test_set.loc[:, "y_true"].to_numpy()
    #     data_test = DatasetAminoAcids(X_test, y_test_true)

    #     # BOOSTING
    #     X_train = train_set.iloc[:, :20].to_numpy()
    #     y_train_true = train_set.loc[:, "y_true"].to_numpy()
    #     data_train = DatasetAminoAcids(X_train, y_train_true)
    #     for p in boost_proportions:
    #         mean_vars_test = []
    #         mses_test = []
    #         test_preds_boosted = []
    #         train_preds_boosted = []

    #         model = NeuralNetwork(lr=LR).to(DEVICE)
    #         train(
    #             model, data_train, data_test, EPOCHS, BATCH_SIZE, LR, print_status=False
    #         )

    #         n_retrain = int(len(X_train) * p)
    #         for boost in range(n_boost):
    #             train_preds = model.evaluate(X_train)
    #             train_preds = np.clip(train_preds, 0, 1)
    #             square_error = np.power(y_train_true - train_preds, 2)
    #             order = np.argsort(square_error)[::-1][:n_retrain]
    #             X_train_retrain = X_train[order]
    #             y_train_retrain = y_train_true[order]
    #             print(
    #                 f" -> Boost {boost}\tMSE: {round(mean_squared_error(y_train_true, train_preds), 4)}"
    #             )

    #             data_retrain = DatasetAminoAcids(X_train_retrain, y_train_retrain)
    #             train(
    #                 model,
    #                 data_retrain,
    #                 data_test,
    #                 EPOCHS,
    #                 min(len(X_train_retrain) // 5, BATCH_SIZE),
    #                 LR,
    #                 print_status=False,
    #                 save_plots=False,
    #                 compute_test_stats=False,
    #             )

    #             test_preds = model.evaluate(X_test)
    #             test_preds = np.clip(test_preds, 0, 1)
    #             test_preds_boosted.append(test_preds.reshape(-1, 1))
    #             results_test = np.vstack((y_test_true, test_preds))
    #             mse = mean_squared_error(y_test_true, test_preds)
    #             var = results_test.var(axis=0).mean()
    #             mses_test.append(mse)
    #             mean_vars_test.append(var)
    #             print(f"Test Var: {var}, MSE: {mse}")

    #             train_preds_boosted.append(train_preds.reshape(-1, 1))
    #             results_train = np.vstack((y_train_true, train_preds))
    #             mse = mean_squared_error(y_train_true, train_preds)
    #             var = results_train.var(axis=0).mean()
    #             print(f"Train Var: {var}, MSE: {mse}")

    #         cols = [f"y_boost_{i+1}" for i in range(n_boost)]
    #         test_preds_boosted = np.hstack(test_preds_boosted)
    #         test_preds_boosted = pd.DataFrame(test_preds_boosted, columns=cols)
    #         test_preds_boosted["y_true"] = y_test_true
    #         test_preds_boosted.to_csv(
    #             f"boost_tests/BoostTest_{train_split:.2f}_{p:.2f}.csv", index=None
    #         )

    #         print("\nRESULTS:")
    #         for v, m in zip(mean_vars_test, mses_test):
    #             print(f"Test Var: {v}, MSE: {m}")

    #         train_preds_boosted = np.hstack(train_preds_boosted)
    #         train_preds_boosted = pd.DataFrame(train_preds_boosted, columns=cols)
    #         train_preds_boosted["y_true"] = y_train_true
    #         train_preds_boosted.to_csv(
    #             f"boost_tests/BoostTrain_{train_split:.2f}_{p:.2f}.csv", index=None
    #         )

    #     # # BAGGING
    #     # for p in bag_proportions:
    #     #     n_train_data = int(p * len(train_set))

    #     #     mean_vars_test = []
    #     #     mses_test = []
    #     #     test_preds = []
    #     #     test_preds_bagged = []
    #     #     train_preds = []
    #     #     train_preds_bagged = []

    #     #     for b in range(n_bags):
    #     #         print(f"\nBag {b}, p={p:.2f}")
    #     #         X_train = train_set.iloc[:, :20].to_numpy()
    #     #         y_train_true = train_set.loc[:, "y_true"].to_numpy()
    #     #         data_train = DatasetAminoAcids(X_train, y_train_true)

    #     #         if n_bags > 1:
    #     #             train_indexes = np.random.choice(
    #     #                 train_set.index, n_train_data, replace=True
    #     #             )
    #     #             X_train_bag = train_set.iloc[train_indexes, :20].to_numpy()
    #     #             y_train_true_bag = train_set.loc[train_indexes, "y_true"].to_numpy()
    #     #             data_train_bag = DatasetAminoAcids(X_train_bag, y_train_true_bag)
    #     #         else:
    #     #             data_train_bag = data_train

    #     #         model = NeuralNetwork(lr=LR).to(DEVICE)
    #     #         train(
    #     #             model,
    #     #             data_train_bag,
    #     #             data_test,
    #     #             EPOCHS,
    #     #             BATCH_SIZE,
    #     #             LR,
    #     #             # print_status=False,
    #     #             save_plots=False
    #     #             compute_test_stats=False,
    #     #         )

    #     #         nn_pred_test = model.evaluate(X_test)
    #     #         nn_pred_test = np.clip(nn_pred_test, 0, 1)
    #     #         test_preds.append(nn_pred_test.reshape(-1, 1))
    #     #         test_bagged = np.hstack(test_preds).mean(axis=1)
    #     #         test_preds_bagged.append(test_bagged.reshape(-1, 1))

    #     #         results_test = np.vstack((y_test_true, test_bagged))
    #     #         mse = mean_squared_error(y_test_true, test_bagged)
    #     #         var = results_test.var(axis=0).mean()
    #     #         mses_test.append(mse)
    #     #         mean_vars_test.append(var)
    #     #         print(f"Test Var: {var}, MSE: {mse}")

    #     #         train_preds = model.evaluate(X_train)
    #     #         train_preds = np.clip(train_preds, 0, 1)
    #     #         train_preds.append(train_preds.reshape(-1, 1))
    #     #         train_bagged = np.hstack(train_preds).mean(axis=1)
    #     #         train_preds_bagged.append(train_bagged.reshape(-1, 1))

    #     #         results_train = np.vstack((y_train_true, train_bagged))
    #     #         mse = mean_squared_error(y_train_true, train_bagged)
    #     #         var = results_train.var(axis=0).mean()
    #     #         print(f"Train Var: {var}, MSE: {mse}")

    #     #     cols = [f"y_bags_{i+1}" for i in range(n_bags)]
    #     #     test_preds_bagged = np.hstack(test_preds_bagged)
    #     #     test_preds_bagged = pd.DataFrame(test_preds_bagged, columns=cols)
    #     #     test_preds_bagged["y_true"] = y_test_true
    #     #     test_preds_bagged.to_csv(
    #     #         f"bag_tests/BagTest_{train_split:.2f}_{p:.2f}.csv", index=None
    #     #     )

    #     #     print("\nRESULTS:")
    #     #     for v, m in zip(mean_vars_test, mses_test):
    #     #         print(f"Test Var: {v}, MSE: {m}")

    #     #     train_preds_bagged = np.hstack(train_preds_bagged)
    #     #     train_preds_bagged = pd.DataFrame(train_preds_bagged, columns=cols)
    #     #     train_preds_bagged["y_true"] = y_train_true
    #     #     train_preds_bagged.to_csv(
    #     #         f"bag_tests/BagTrain_{train_split:.2f}_{p:.2f}.csv", index=None
    #     #     )

    # fig, axs = plt.subplots(
    #     nrows=2, ncols=len(splits), sharex=True, sharey="row", figsize=(18, 8)
    # )
    # proportions = boost_proportions
    # # proportions = bag_proportions
    # N = n_boost
    # # N = n_bags
    # for idx, train_split in enumerate(splits):
    #     if idx == 0:
    #         axs[0, idx].set_ylabel("Train MSE")
    #         axs[1, idx].set_ylabel("Test MSE")
    #     for p in proportions:
    #         train_preds = pd.read_csv(
    #             f"boost_tests/BoostTrain_{train_split:.2f}_{p:.2f}.csv", index_col=None
    #         )
    #         test_preds = pd.read_csv(
    #             f"boost_tests/BoostTrain_{train_split:.2f}_{p:.2f}.csv", index_col=None
    #         )
    #         # train_preds = pd.read_csv(
    #         #     f"bag_tests/BagTrain_{train_split:.2f}_{p:.2f}.csv", index_col=None
    #         # )
    #         # test_preds = pd.read_csv(
    #         #     f"bag_tests/BagTrain_{train_split:.2f}_{p:.2f}.csv", index_col=None
    #         # )

    #         train_y_true = train_preds["y_true"]
    #         test_y_true = test_preds["y_true"]
    #         train_mses = [
    #             mean_squared_error(train_y_true, train_preds.iloc[:, j])
    #             for j in range(N)
    #         ]
    #         test_mses = [
    #             mean_squared_error(test_y_true, test_preds.iloc[:, j]) for j in range(N)
    #         ]

    #         x = np.arange(N) + 1
    #         axs[0, idx].plot(x, train_mses, "-", label=f"P={p:.2f}")
    #         axs[1, idx].plot(x, test_mses, "-", label=f"P={p:.2f}")
    #         axs[0, idx].yaxis.set_tick_params(labelleft=True)
    #         axs[1, idx].yaxis.set_tick_params(labelleft=True)

    #     axs[1, idx].set_xlabel("N bags")
    #     axs[0, idx].set_title(f"Train Set (train={train_split:.2f})")
    #     axs[1, idx].set_title(f"Test Set (train={train_split:.2f})")

    # handles, labels = axs[1, 0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="lower center", ncol=len(proportions))
    # fig.tight_layout()
    # fig.subplots_adjust(bottom=0.1)

    # plt.savefig("boost_tests/BoostComparison.png", dpi=400)
    # # plt.savefig("bag_tests/BagComparison.png", dpi=400)


# TRAIN AND PLOT SCATTERS
# if __name__ == "__main__":
#     # LR = 1.02e-5

#     # # data_path = "data/L1IO-L2IO-L3O All Rands SMU UA159 Processed-Aerobic.csv"
#     splits = [0.01, 0.05, 0.1, 0.25, 0.50]
#     n_bags = 10
#     n_boost = 0
#     bag_proportion = 1
#     boost_proportion = 0.25
#     for idx, train_split in enumerate(splits):
#         print(f"SPLIT {train_split:.2f}")
#         test_path = f"GPRvNN_test_pred_{train_split:.2f}.csv"
#         train_path = f"GPRvNN_train_pred_{train_split:.2f}.csv"
#         test_set = pd.read_csv(test_path, index_col=None)
#         train_set = pd.read_csv(train_path, index_col=None)

#         X_test = test_set.iloc[:, :20].to_numpy()
#         y_test_true = test_set.loc[:, "y_true"].to_numpy()
#         data_test = DatasetAminoAcids(X_test, y_test_true)

#         n_train_data = int(bag_proportion * len(train_set))

#         mean_vars_test = []
#         mses_test = []
#         test_preds = []
#         for b in range(n_bags):
#             print(f"\nBag {b}")
#             if n_bags > 1:
#                 train_indexes = np.random.choice(
#                     train_set.index, n_train_data, replace=True
#                 )
#                 X_train = train_set.iloc[train_indexes, :20].to_numpy()
#                 y_train_true = train_set.loc[train_indexes, "y_true"].to_numpy()
#             else:
#                 X_train = train_set.iloc[:, :20].to_numpy()
#                 y_train_true = train_set.loc[:, "y_true"].to_numpy()

#             data_train = DatasetAminoAcids(X_train, y_train_true)
#             model = NeuralNetwork(lr=LR).to(DEVICE)
#             train(
#                 model,
#                 data_train,
#                 data_test,
#                 EPOCHS,
#                 BATCH_SIZE,
#                 LR,  # print_status=False
#             )
#             if n_boost > 0:
#                 n_retrain = int(len(X_train) * boost_proportion)
#                 for boost in range(n_boost):
#                     train_preds = model.evaluate(X_train)
#                     train_preds = np.clip(train_preds, 0, 1)
#                     square_error = np.power(y_train_true - train_preds, 2)
#                     order = np.argsort(square_error)[::-1][:n_retrain]
#                     X_train_retrain = X_train[order]
#                     y_train_retrain = y_train_true[order]
#                     print(X_train_retrain.shape)
#                     print(y_train_retrain.shape)
#                     print(
#                         f" -> Boost {boost}\tMSE: {round(mean_squared_error(y_train_true, train_preds), 4)}"
#                     )

#                     data_retrain = DatasetAminoAcids(X_train_retrain, y_train_retrain)
#                     train(
#                         model,
#                         data_retrain,
#                         data_test,
#                         EPOCHS,
#                         min(len(X_train_retrain) // 5, BATCH_SIZE),
#                         LR,
#                         # print_status=False,
#                         save_plots=False,
#                     )

#             # torch.save(model, PATH)
#             # model = torch.load(PATH)

#             # model = XGBRegressor(
#             #     max_depth=6,
#             #     learning_rate=0.01,
#             #     n_estimators=2000,
#             #     # n_jobs=-1,
#             #     colsample_bytree=0.1,
#             #     tree_method="gpu_hist",
#             #     gpu_id=0,
#             #     reg_alpha=0.01,
#             #     reg_lambda=0.01,
#             #     # seed=SEED,
#             # )
#             # model.fit(X_train, y_train_true)

#             nn_pred_test = model.evaluate(X_test)
#             # nn_pred_test = model.predict(X_test)
#             nn_pred_test = np.clip(nn_pred_test, 0, 1)
#             results_test = np.vstack((y_test_true, nn_pred_test))
#             mse = mean_squared_error(y_test_true, nn_pred_test)
#             var = results_test.var(axis=0).mean()
#             mses_test.append(mse)
#             mean_vars_test.append(var)
#             test_preds.append(nn_pred_test)
#             print(f"Test Var: {var}, MSE: {mse}")

#             train_preds = model.evaluate(X_train)
#             # train_preds = model.predict(X_train)
#             train_preds = np.clip(train_preds, 0, 1)
#             results_train = np.vstack((y_train_true, train_preds))
#             mse = mean_squared_error(y_train_true, train_preds)
#             var = results_train.var(axis=0).mean()
#             print(f"Train Var: {var}, MSE: {mse}")

#         test_preds = np.vstack(test_preds).mean(axis=0)
#         mse = mean_squared_error(y_test_true, test_preds)
#         var = results_test.var(axis=0).mean()

#         print("\nRESULTS:")
#         for v, m in zip(mean_vars_test, mses_test):
#             print(f"Test Var: {v}, MSE: {m}")
#         print(f"Bagged Test Var: {var}, MSE: {mse}")

#         test_set["y_pred_bag"] = test_preds
#         test_set.iloc[:, :20] = test_set.iloc[:, :20].astype(int)
#         test_set.to_csv(f"GPRvNN_test_pred_{train_split:.2f}.csv", index=None)

#         X_train = train_set.iloc[:, :20].to_numpy()
#         train_preds = model.evaluate(X_train)
#         train_preds = np.clip(train_preds, 0, 1)
#         train_set["y_pred_bag"] = train_preds
#         train_set.iloc[:, :20] = train_set.iloc[:, :20].astype(int)
#         train_set.to_csv(f"GPRvNN_train_pred_{train_split:.2f}.csv", index=None)

#         # print("TRUE GROW:", y_test_true.min(), y_test_true.max(), y_test_true.mean())
#         # print("GPR PRED:", gpr_pred.min(), gpr_pred.max(), gpr_pred.mean())
#         # print("NN PRED:", nn_pred.min(), nn_pred.max(), nn_pred.mean())

#     fig, axs = plt.subplots(
#         nrows=4, ncols=5, sharex=False, sharey=False, figsize=(18, 14)
#     )
#     for idx, train_split in enumerate(splits):
#         test_path = f"GPRvNN_test_pred_{train_split:.2f}.csv"
#         train_path = f"GPRvNN_train_pred_{train_split:.2f}.csv"
#         test_set = pd.read_csv(test_path, index_col=None)
#         train_set = pd.read_csv(train_path, index_col=None)

#         y_test_true = test_set.loc[:, "y_true"].to_numpy()
#         y_train_true = train_set.loc[:, "y_true"].to_numpy()

#         gpr_test_pred = test_set.loc[:, "y_pred_bag"].to_numpy()
#         gpr_train_pred = train_set.loc[:, "y_pred_bag"].to_numpy()
#         gpr_test_mse = mean_squared_error(y_test_true, gpr_test_pred)
#         gpr_train_mse = mean_squared_error(y_train_true, gpr_train_pred)

#         nn_test_pred = test_set.loc[:, "y_pred_boost_nn"].to_numpy()
#         nn_train_pred = train_set.loc[:, "y_pred_boost_nn"].to_numpy()
#         nn_test_mse = mean_squared_error(y_test_true, nn_test_pred)
#         nn_train_mse = mean_squared_error(y_train_true, nn_train_pred)

#         msize = 3

#         alpha = 0.5
#         x = range(y_train_true.shape[0])
#         order = np.argsort(y_train_true)
#         axs[0, idx].plot(
#             x,
#             nn_train_pred[order],
#             "g.",
#             markeredgewidth=0.0,
#             fillstyle="full",
#             label="NN",
#             alpha=alpha,
#             markersize=msize,
#         )
#         axs[0, idx].plot(
#             x,
#             y_train_true[order],
#             "b.",
#             markeredgewidth=0.0,
#             fillstyle="full",
#             label="True",
#             alpha=alpha,
#             markersize=msize,
#         )
#         axs[0, idx].set_title(
#             f"Train Predictions Boost NN (Train={train_split:.2f})\nMSE: {nn_train_mse:.3f}"
#         )

#         axs[1, idx].plot(
#             x,
#             gpr_train_pred[order],
#             "r.",
#             markeredgewidth=0.0,
#             fillstyle="full",
#             label="GPR",
#             alpha=alpha,
#             markersize=msize,
#         )
#         axs[1, idx].plot(
#             x,
#             y_train_true[order],
#             "b.",
#             markeredgewidth=0.0,
#             fillstyle="full",
#             label="True",
#             alpha=alpha,
#             markersize=msize,
#         )
#         axs[1, idx].set_title(
#             f"Train Predictions Bagged NN (Train={train_split:.2f})\nMSE: {gpr_train_mse:.3f}"
#         )

#         x = range(y_test_true.shape[0])
#         order = np.argsort(y_test_true)
#         alpha = 0.35
#         axs[2, idx].plot(
#             x,
#             nn_test_pred[order],
#             "g.",
#             markeredgewidth=0.0,
#             fillstyle="full",
#             label="Bagged + Boosted NN",
#             alpha=alpha,
#             markersize=msize,
#         )
#         axs[2, idx].plot(
#             x,
#             y_test_true[order],
#             "b.",
#             markeredgewidth=0.0,
#             fillstyle="full",
#             label="True",
#             alpha=alpha,
#             markersize=msize,
#         )
#         # axs[2, idx].set_title(
#         #     f"Test Predictions NN (Train={train_split:.2f})\nMSE: {nn_test_mse:.3f}"
#         # )
#         axs[2, idx].set_title(
#             f"Test Predictions Boost NN\nBag(N={n_bags}, p={bag_proportion:.2f})\nBoost(N={n_boost}, p={boost_proportion:.2f})\n(Train={train_split:.2f})\nMSE: {nn_test_mse:.3f}"
#         )

#         axs[3, idx].plot(
#             x,
#             gpr_test_pred[order],
#             "r.",
#             markeredgewidth=0.0,
#             fillstyle="full",
#             label="Bagged NN",
#             alpha=alpha,
#             markersize=msize,
#         )
#         axs[3, idx].plot(
#             x,
#             y_test_true[order],
#             "b.",
#             markeredgewidth=0.0,
#             fillstyle="full",
#             label="True",
#             alpha=alpha,
#             markersize=msize,
#         )
#         axs[3, idx].set_title(
#             f"Test Predictions Bagged NN\n(N={n_bags}, p={bag_proportion:.2f})\n(Train={train_split:.2f})\nMSE: {gpr_test_mse:.3f}"
#         )

#     handles, labels = axs[1, 0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc="lower center", ncol=len(labels))
#     fig.suptitle("BoostedNN v BaggedNN Predictions")
#     # fig.suptitle("GPR v NN Predictions")
#     fig.tight_layout()
#     fig.subplots_adjust(bottom=0.1)
#     plt.savefig(
#         f"BoostedNNvBaggedNN({n_bags}_{bag_proportion:.2f},{n_boost}_{boost_proportion:.2f}).png",
#         dpi=400,
#     )
#     plt.close("all")


#     #OTHER PLOTS
#     # plt.savefig(f"GPRvNN.png", dpi=400)

#     # plt.figure()
#     # plt.plot(gpr_pred, nn_pred, "b.", alpha=0.1)
#     # plt.xlabel("GPR Pred")
#     # plt.ylabel("NN Pred")
#     # plt.title("Predictions")
#     # plt.tight_layout()
#     # plt.savefig("result_GPRvNN.png")

#     # plt.figure()
#     # # plt.plot(x, gpr_pred[order], "b.", alpha=0.1)
#     # plt.plot(x, y_test_true[order], "b.", alpha=0.1)
#     # plt.fill_between(
#     #     x,
#     #     gpr_pred[order] - gpr_var[order],
#     #     gpr_pred[order] + gpr_var[order],
#     #     facecolor="blue",
#     #     alpha=1,
#     # )
#     # plt.xlabel("TRUE")
#     # plt.ylabel("GPR Pred")
#     # plt.title("Predictions")
#     # plt.tight_layout()
#     # plt.savefig("result_TRUEvGPR_uncertainty.png")

#     # order = np.argsort(gpr_pred)
#     # plt.figure()
#     # # plt.plot(x, gpr_pred[order], "b.", alpha=0.1)
#     # plt.plot(x, gpr_pred[order], "b", alpha=1)
#     # plt.fill_between(
#     #     x,
#     #     gpr_pred[order] - gpr_var[order],
#     #     gpr_pred[order] + gpr_var[order],
#     #     facecolor="blue",
#     #     alpha=0.10,
#     # )
#     # plt.xlabel("exp")
#     # plt.ylabel("GPR Pred")
#     # plt.title("Predictions")
#     # plt.tight_layout()
#     # plt.savefig("result_GPR_uncertainty_0.50.png")