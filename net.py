import datetime

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression


class DatasetAminoAcids(Dataset):
    def __init__(self, X, y=None, mode="train"):
        self.mode = mode
        # le = LabelEncoder()

        if self.mode == "train":
            self.X = X
            self.y = y
        else:
            self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode == "train":
            X_ = torch.from_numpy(np.array(self.X[idx])).float()
            y_ = torch.from_numpy(np.array(self.y[idx])).float()
            return X_, y_
        else:
            X_ = torch.from_numpy(np.array(self.X[idx])).float()
            return X_


def split_data(path, train_size=0.15, test_size=0.25):
    data = pd.read_csv(path, index_col=None, header=0)
    print(data)
    if "environment" in data.columns:
        data = data.drop(columns="environment")

    X, y = data[data.columns[:-1]].values, data["growth"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size
    )

    data_train = DatasetAminoAcids(X_train, y_train)
    data_test = DatasetAminoAcids(X_test, y_test)
    return data_train, data_test


class NeuralNetwork(nn.Module):
    def __init__(self, lr=0.001):
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Linear(1)
            # nn.Sigmoid(),
        )
        # self.linear_relu_stack = nn.Sequential(
        #     nn.Linear(20, 873),
        #     nn.ReLU(),
        #     nn.Linear(873, 1038),
        #     nn.ReLU(),
        #     nn.Linear(1038, 636),
        #     nn.ReLU(),
        #     nn.Linear(636, 1),
        #     nn.Sigmoid(),
        # )

        self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss()
        # self.optimizer = Adam(self.parameters(), lr=lr)
        self.optimizer = Adam(
            self.parameters(), lr=lr  # , weight_decay=0.001
        )  # weight_decay=1e-2)
        self.threshold = 0.25

    def forward(self, x):
        pred = self.linear_relu_stack(x)
        return pred

    def evaluate(self, x):
        x = torch.from_numpy(np.array(x)).float().to("cuda")
        y = self.forward(x).to("cpu").detach().numpy().flatten()
        return y


def train(
    model,
    data_train,
    data_test,
    epochs,
    batch_size,
    lr,
    print_status=True,
    save_plots=True,
    optuna_trial=None,
    compute_test_stats=True,
):
    train_loss_means = []
    train_mse_means = []
    train_acc_means = []
    test_loss_means = []
    test_mse_means = []
    test_acc_means = []
    test_mse_moving_avg = []

    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        train_loss_epoch = []
        train_mse_epoch = []
        train_acc_epoch = []
        test_loss_epoch = []
        test_mse_epoch = []
        test_acc_epoch = []

        train_loader = iter(
            DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
        )
        test_loader = iter(
            DataLoader(dataset=data_test, batch_size=batch_size, shuffle=True)
        )
        for i, (train_data, test_data) in enumerate(zip(train_loader, test_loader)):
            # Train
            train_inputs = train_data[0].to("cuda")
            train_labels = train_data[1].to("cuda").unsqueeze(1)

            # zero the parameter gradients
            model.optimizer.zero_grad()

            # forward + backward + optimize
            train_outputs = model.forward(train_inputs)
            train_loss = model.criterion(train_outputs, train_labels)

            # reg_loss = None
            # for W in model.parameters():
            #     if reg_loss is None:
            #         reg_loss = W.norm(1)
            #     else:
            #         reg_loss = reg_loss + W.norm(1)

            # factor = 0.001
            # train_loss += factor * reg_loss

            train_loss.backward()
            model.optimizer.step()

            train_mse = model.mse(train_outputs, train_labels)

            train_outputs[train_outputs >= model.threshold] = 1
            train_outputs[train_outputs < model.threshold] = 0
            train_labels[train_labels >= model.threshold] = 1
            train_labels[train_labels < model.threshold] = 0
            train_acc = (train_outputs == train_labels).sum() / train_outputs.shape[0]

            # Train
            if compute_test_stats:
                test_inputs = test_data[0].to("cuda")
                test_labels = test_data[1].to("cuda").unsqueeze(1)

                test_outputs = model.forward(test_inputs)
                test_loss = model.criterion(test_outputs, test_labels).item()
                test_mse = model.mse(test_outputs, test_labels).item()

                test_outputs[test_outputs >= model.threshold] = 1
                test_outputs[test_outputs < model.threshold] = 0
                test_labels[test_labels >= model.threshold] = 1
                test_labels[test_labels < model.threshold] = 0
                test_acc = (
                    (test_outputs == test_labels).sum() / test_outputs.shape[0]
                ).item()
            else:
                test_loss = 0
                test_mse = 0
                test_acc = 0

            train_loss_epoch.append(train_loss.item())
            train_mse_epoch.append(train_mse.item())
            train_acc_epoch.append(train_acc.item())
            test_loss_epoch.append(test_loss)
            test_mse_epoch.append(test_mse)
            test_acc_epoch.append(test_acc)

        tr_l = sum(train_loss_epoch) / len(train_loss_epoch)
        tr_mse = sum(train_mse_epoch) / len(train_mse_epoch)
        tr_acc = sum(train_acc_epoch) / len(train_acc_epoch)
        te_l = sum(test_loss_epoch) / len(test_loss_epoch)
        te_mse = sum(test_mse_epoch) / len(test_mse_epoch)
        te_acc = sum(test_acc_epoch) / len(test_acc_epoch)

        train_loss_means.append(tr_l)
        train_mse_means.append(tr_mse)
        train_acc_means.append(tr_acc)
        test_loss_means.append(te_l)
        test_mse_means.append(te_mse)
        test_acc_means.append(te_acc)

        moving_subset = test_mse_means[-10:]
        moving_avg_mse = sum(moving_subset) / len(moving_subset)
        test_mse_moving_avg.append(moving_avg_mse)

        if print_status:
            print(
                f"E{epoch}\t| Train(Loss: {round(tr_l, 4)}, MSE: {round(tr_mse, 4)}, ACC: {round(tr_acc, 4)}) | Test(Loss: {round(te_l, 4)}, MSE: {round(te_mse, 4)}, ACC: {round(te_acc, 4)})"
            )

        if optuna_trial:

            optuna_trial.report(moving_avg_mse, epoch)
            # Handle pruning based on the intermediate value.
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if save_plots:
        x = range(1, epochs + 1)
        plt.figure()
        plt.plot(x, train_loss_means)
        plt.plot(x, train_mse_means)
        plt.plot(x, train_acc_means)
        plt.plot(x, test_loss_means)
        plt.plot(x, test_mse_means)
        plt.plot(x, test_acc_means)
        plt.plot(x, test_mse_moving_avg)
        plt.legend(
            [
                "train_loss",
                "train_mse",
                "train_acc",
                "test_loss",
                "test_mse",
                "test_acc",
                "test_mse_moving_avg-10",
            ]
        )
        plt.xlabel("Epoch")
        plt.title("NN performance")
        plt.savefig("result_training_nn.png")

    final_moving_avg = test_mse_moving_avg[-1]
    if print_status:
        print(f"Final 10-Moving Avg Epoch MSE: {final_moving_avg}")

    return final_moving_avg


def optuna_objective(trial):
    data_train = pd.read_csv("data/gpr_train_pred_0.20.csv", index_col=None)
    data_test = pd.read_csv("data/gpr_test_pred_0.20.csv", index_col=None)
    data_train = DatasetAminoAcids(data_train.values[:, :-3], data_train.values[:, -3])
    data_test = DatasetAminoAcids(data_test.values[:, :-3], data_test.values[:, -3])

    # Generate the model.
    # model = define_model(trial).to("cuda")
    model = NeuralNetwork().to("cuda")

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    wd = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)
    model.optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=wd
    )

    # loss_name = trial.suggest_categorical(
    #     "loss_criterion", ["BCELoss", "MSELoss", "L1Loss"]
    # )
    # losses = {"BCELoss": nn.BCELoss(), "MSELoss": nn.MSELoss(), "L1Loss": nn.L1Loss()}
    # model.criterion = losses[loss_name]
    batch_size = trial.suggest_int("batch_size", 128, 2048)
    epochs = 250
    # batch_size = 512

    mse = train(
        model,
        data_train,
        data_test,
        epochs,
        batch_size,
        lr,
        print_status=False,
        save_plots=False,
        optuna_trial=trial,
    )
    return mse


def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 3, 8)
    layers = []

    in_features = 20
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 32, 2048)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        # p = trial.suggest_uniform("dropout_l{}".format(i), 0.05, 0.5)
        # layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, 1))
    layers.append(nn.Sigmoid())

    model = NeuralNetwork().to("cuda")
    model.linear_relu_stack = nn.Sequential(*layers)

    return model


def run_optuna(n_trials=100):
    start_date = datetime.datetime.now().isoformat().replace(":", ".")
    study = optuna.create_study(
        study_name="SMU_nn_optim_20",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
    )
    study.optimize(optuna_objective, n_trials=n_trials)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("- Number of finished trials: ", len(study.trials))
    print("- Number of pruned trials: ", len(pruned_trials))
    print("- Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("- Value: ", trial.value)

    print("- Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    df_optuna = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df_optuna.to_csv(f"optuna_trial_results_{start_date}.csv")
    print(df_optuna)

    ax = optuna.visualization.matplotlib.plot_intermediate_values(study)
    ax.legend()
    ax.figure.set_figheight(10)
    ax.figure.set_figwidth(20)
    ax.figure.savefig(f"optuna_trial_viz_int_values_{start_date}.png", dpi=400)

    print(f"Finished experiment: {start_date}")


# if __name__ == "__main__":
#     run_optuna(1000)

PATH = "model.pkl"
BATCH_SIZE = 360
EPOCHS = 50
LR = 0.001

if __name__ == "__main__":
    # # data_path = "L1IO-L2IO-L3O All Rands SMU UA159 Processed-Aerobic.csv"
    splits = [0.05, 0.10, 0.25, 0.50]  # [0.01, 0.05, 0.10, 0.25, 0.50]
    n_bags = 50
    n_boost = 50
    bag_proportions = [0.05, 0.10, 0.25, 0.50, 1.00]
    boost_proportions = [0.01, 0.05, 0.10, 0.25, 0.50]

    # for idx, train_split in enumerate(splits):
    #     print(f"SPLIT {train_split:.2f}")
    #     test_path = f"GPRvNN_test_pred_{train_split:.2f}.csv"
    #     train_path = f"GPRvNN_train_pred_{train_split:.2f}.csv"
    #     test_set = pd.read_csv(test_path, index_col=None)
    #     train_set = pd.read_csv(train_path, index_col=None)

    #     X_test = test_set.iloc[:, :20].values
    #     y_test_true = test_set.loc[:, "y_true"].values
    #     data_test = DatasetAminoAcids(X_test, y_test_true)

    #     # BOOSTING
    #     X_train = train_set.iloc[:, :20].values
    #     y_train_true = train_set.loc[:, "y_true"].values
    #     data_train = DatasetAminoAcids(X_train, y_train_true)
    #     for p in boost_proportions:
    #         mean_vars_test = []
    #         mses_test = []
    #         test_preds_boosted = []
    #         train_preds_boosted = []

    #         model = NeuralNetwork(lr=LR).to("cuda")
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
    #     #         X_train = train_set.iloc[:, :20].values
    #     #         y_train_true = train_set.loc[:, "y_true"].values
    #     #         data_train = DatasetAminoAcids(X_train, y_train_true)

    #     #         if n_bags > 1:
    #     #             train_indexes = np.random.choice(
    #     #                 train_set.index, n_train_data, replace=True
    #     #             )
    #     #             X_train_bag = train_set.iloc[train_indexes, :20].values
    #     #             y_train_true_bag = train_set.loc[train_indexes, "y_true"].values
    #     #             data_train_bag = DatasetAminoAcids(X_train_bag, y_train_true_bag)
    #     #         else:
    #     #             data_train_bag = data_train

    #     #         model = NeuralNetwork(lr=LR).to("cuda")
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

    fig, axs = plt.subplots(
        nrows=2, ncols=len(splits), sharex=True, sharey="row", figsize=(18, 8)
    )
    proportions = boost_proportions
    # proportions = bag_proportions
    N = n_boost
    # N = n_bags
    for idx, train_split in enumerate(splits):
        if idx == 0:
            axs[0, idx].set_ylabel("Train MSE")
            axs[1, idx].set_ylabel("Test MSE")
        for p in proportions:
            train_preds = pd.read_csv(
                f"boost_tests/BoostTrain_{train_split:.2f}_{p:.2f}.csv", index_col=None
            )
            test_preds = pd.read_csv(
                f"boost_tests/BoostTrain_{train_split:.2f}_{p:.2f}.csv", index_col=None
            )
            # train_preds = pd.read_csv(
            #     f"bag_tests/BagTrain_{train_split:.2f}_{p:.2f}.csv", index_col=None
            # )
            # test_preds = pd.read_csv(
            #     f"bag_tests/BagTrain_{train_split:.2f}_{p:.2f}.csv", index_col=None
            # )

            train_y_true = train_preds["y_true"]
            test_y_true = test_preds["y_true"]
            train_mses = [
                mean_squared_error(train_y_true, train_preds.iloc[:, j])
                for j in range(N)
            ]
            test_mses = [
                mean_squared_error(test_y_true, test_preds.iloc[:, j]) for j in range(N)
            ]

            x = np.arange(N) + 1
            axs[0, idx].plot(x, train_mses, "-", label=f"P={p:.2f}")
            axs[1, idx].plot(x, test_mses, "-", label=f"P={p:.2f}")
            axs[0, idx].yaxis.set_tick_params(labelleft=True)
            axs[1, idx].yaxis.set_tick_params(labelleft=True)

        axs[1, idx].set_xlabel("N bags")
        axs[0, idx].set_title(f"Train Set (train={train_split:.2f})")
        axs[1, idx].set_title(f"Test Set (train={train_split:.2f})")

    handles, labels = axs[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(proportions))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.1)

    plt.savefig("boost_tests/BoostComparison.png", dpi=400)
    # plt.savefig("bag_tests/BagComparison.png", dpi=400)


# TRAIN AND PLOT SCATTERS
# if __name__ == "__main__":
#     # LR = 1.02e-5

#     # # data_path = "L1IO-L2IO-L3O All Rands SMU UA159 Processed-Aerobic.csv"
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

#         X_test = test_set.iloc[:, :20].values
#         y_test_true = test_set.loc[:, "y_true"].values
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
#                 X_train = train_set.iloc[train_indexes, :20].values
#                 y_train_true = train_set.loc[train_indexes, "y_true"].values
#             else:
#                 X_train = train_set.iloc[:, :20].values
#                 y_train_true = train_set.loc[:, "y_true"].values

#             data_train = DatasetAminoAcids(X_train, y_train_true)
#             model = NeuralNetwork(lr=LR).to("cuda")
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

#         X_train = train_set.iloc[:, :20].values
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

#         y_test_true = test_set.loc[:, "y_true"].values
#         y_train_true = train_set.loc[:, "y_true"].values

#         gpr_test_pred = test_set.loc[:, "y_pred_bag"].values
#         gpr_train_pred = train_set.loc[:, "y_pred_bag"].values
#         gpr_test_mse = mean_squared_error(y_test_true, gpr_test_pred)
#         gpr_train_mse = mean_squared_error(y_train_true, gpr_train_pred)

#         nn_test_pred = test_set.loc[:, "y_pred_boost_nn"].values
#         nn_train_pred = train_set.loc[:, "y_pred_boost_nn"].values
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