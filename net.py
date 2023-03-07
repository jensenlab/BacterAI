import os
import random
import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DatasetAminoAcids(Dataset):
    def __init__(self, X, y=None, mode="train"):
        self.mode = mode
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

    X, y = data[data.columns[:-1]].to_numpy(), data["growth"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, test_size=test_size
    )

    data_train = DatasetAminoAcids(X_train, y_train)
    data_test = DatasetAminoAcids(X_test, y_test)
    return data_train, data_test


class NeuralNetwork(nn.Module):
    def __init__(self, lr=0.001, n_inputs=20):
        super(NeuralNetwork, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss()
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.threshold = 0.25

    def forward(self, x):
        pred = self.linear_relu_stack(x)
        return pred

    def evaluate(self, x):
        x = torch.from_numpy(np.array(x)).float().to(DEVICE)
        y = self.forward(x).to("cpu").detach().numpy().flatten()
        return y


def threshold(data, value):
    data[data >= value] = 1
    data[data < value] = 0
    return data


def accuracy(preds, labels, threshold):
    preds = threshold(preds, threshold)
    labels = threshold(labels, threshold)
    acc = ((preds == labels).sum() / preds.shape[0]).item()
    return acc


def mean(data):
    return sum(data) / len(data)


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
            train_inputs = train_data[0].to(DEVICE)
            train_labels = train_data[1].to(DEVICE).unsqueeze(1)

            # zero the parameter gradients
            model.optimizer.zero_grad()

            # forward + backward + optimize
            train_outputs = model.forward(train_inputs)
            train_loss = model.criterion(train_outputs, train_labels)
            train_loss.backward()
            model.optimizer.step()

            # compute stats
            train_mse = model.mse(train_outputs, train_labels)
            train_acc = accuracy(train_outputs, train_labels, model.threshold)

            # Test
            if compute_test_stats:
                test_inputs = test_data[0].to(DEVICE)
                test_labels = test_data[1].to(DEVICE).unsqueeze(1)

                # forward
                test_outputs = model.forward(test_inputs)
                test_loss = model.criterion(test_outputs, test_labels).item()

                # compute stats
                test_mse = model.mse(test_outputs, test_labels).item()
                test_acc = accuracy(test_outputs, test_labels, model.threshold)
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

        tr_l = mean(train_loss_epoch)
        tr_mse = mean(train_mse_epoch)
        tr_acc = mean(train_acc_epoch)
        te_l = mean(test_loss_epoch)
        te_mse = mean(test_mse_epoch)
        te_acc = mean(test_acc_epoch)

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


def train_bagged(
    X_train,
    y_train_true,
    model_path_folder,
    n_ingredients=20,
    n_bags=25,
    bag_proportion=1.0,
    epochs=50,
    batch_size=360,
    lr=0.001,
    transfer_models=[],
):

    if transfer_models:
        if len(transfer_models) != n_bags:
            raise "The number of transfer models needs to match the number of bags."
        else:
            random.shuffle(transfer_models)

    start_time = time.time()
    model_paths = []
    models = []
    avg_mse = []
    n_train_data = int(bag_proportion * len(X_train))

    for b in range(n_bags):
        print(f"\nBag {b}, p={bag_proportion:.2f}")

        train_indexes = np.random.choice(
            range(len(X_train)), n_train_data, replace=True
        )
        X_train_bag = X_train[train_indexes, :]
        y_train_true_bag = y_train_true[train_indexes]
        dataset_bag = DatasetAminoAcids(X_train_bag, y_train_true_bag)

        if transfer_models:
            print(f"Using transfer model: {b}")
            model = transfer_models[b].to(DEVICE)
        else:
            model = NeuralNetwork(lr=lr, n_inputs=n_ingredients).to(DEVICE)

        # Training Model
        for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
            train_loss_epoch = []
            train_mse_epoch = []

            for batch_data in iter(
                DataLoader(dataset=dataset_bag, batch_size=batch_size, shuffle=True)
            ):
                # Train
                train_inputs = batch_data[0].to(DEVICE)
                train_labels = batch_data[1].to(DEVICE).unsqueeze(1)

                # zero the parameter gradients
                model.optimizer.zero_grad()

                # forward + backward + optimize
                train_outputs = model.forward(train_inputs)
                train_loss = model.criterion(train_outputs, train_labels)
                train_loss.backward()
                model.optimizer.step()

                train_mse = model.mse(train_outputs, train_labels)

                train_loss_epoch.append(train_loss.item())
                train_mse_epoch.append(train_mse.item())

            tr_l = sum(train_loss_epoch) / len(train_loss_epoch)
            tr_mse = sum(train_mse_epoch) / len(train_mse_epoch)

            print(
                f"\tEPOCH {epoch:2}/{epochs} | Train Loss: {tr_l:.4f}, Train MSE: {tr_mse:.4f}"
            )

        # Save model
        if not os.path.exists(model_path_folder):
            os.makedirs(model_path_folder)

        model_path = os.path.join(model_path_folder, f"bag_model_{b}.pkl")
        torch.save(model, model_path)
        model_paths.append(model_path)
        models.append(model)
        avg_mse.append(tr_mse)

    end_time = time.time()
    print(
        f"\nAverage Training MSE ({end_time - start_time:.1f}s): {sum(avg_mse)/len(avg_mse):.4f}\n"
    )
    return models


def eval_bagged(X, models):
    preds = np.zeros((len(X), len(models)))
    for i, model in enumerate(models):
        y_pred = model.evaluate(X)
        preds[:, i] = y_pred

    bagged_pred = np.mean(preds, axis=1)
    bagged_var = np.var(preds, axis=1, ddof=0)
    return bagged_pred, bagged_var
