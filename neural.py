import csv
import math
import os
import random

import pandas as pd
import numpy as np
import scipy.stats as sp
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import tensorflow.keras.backend as K

import model


class PredictNet:
    def __init__(self, loss="custom"):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                # tf.keras.layers.Dropout(0.05, noise_shape=None, seed=None),
                tf.keras.layers.Dense(16, activation="relu"),
                # tf.keras.layers.Dropout(0.05, noise_shape=None, seed=None),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        if loss == "custom":
            loss = self.custom_loss
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=loss,
            # loss=self.custom_loss,
            # loss="binary_crossentropy",
            # loss="hinge",
            # loss="squared_hinge",
            metrics=["accuracy"],
        )

        self.model_bayes = GaussianNB()

    def train(self, data, data_labels, epochs):
        self.model.fit(data, data_labels, epochs=epochs)

    def train_bayes(self, data, data_labels):
        self.model_bayes.fit(data, data_labels)

    def predict_bayes(self, data):
        return self.model_bayes.predict(data)

    def evaluate(self, data, data_labels):
        return self.model.evaluate(data, data_labels, verbose=2)

    def predict_probability(self, data):
        return self.model.predict_proba(data)

    def predict_class(self, data):
        return self.model.predict_classes(data)

    # TODO: implement this function:
    def save_model(self):
        return NotImplementedError

    def custom_loss(self, y_true, y_pred):
        # def _f(true, pred):
        # both = K.concatenate((y_actual, y_predicted), axis=0)
        # print(both)
        bcross = K.binary_crossentropy(y_true, y_pred)
        bcross = K.print_tensor(bcross, message="ANS")
        # print(ans)
        # return ans
        # y_true = K.variable(y_true)
        # y_pred = K.variable(y_pred)
        cutoff = 0.25
        N = K.equal(y_true, 0)
        P = K.equal(y_true, 1)
        TN = tf.logical_and(K.equal(y_true, 0), K.less(y_pred, cutoff))
        TP = tf.logical_and(K.equal(y_true, 1), K.greater_equal(y_pred, cutoff))

        N = K.print_tensor(N, message="N")
        P = K.print_tensor(P, message="P")
        TN = K.print_tensor(TN, message="TN")
        TP = K.print_tensor(TP, message="TP")
        # as Keras Tensors
        N = K.sum(K.cast(N, tf.float32))
        P = K.sum(K.cast(P, tf.float32))
        TN = K.sum(K.cast(TN, tf.float32))
        TP = K.sum(K.cast(TP, tf.float32))

        N = K.print_tensor(N, message="Ns")
        P = K.print_tensor(P, message="Ps")
        TN = K.print_tensor(TN, message="TNs")
        TP = K.print_tensor(TP, message="TPs")

        G_ZZ = tf.cast(tf.logical_and(K.equal(TP, 0), K.equal(P, 0)), tf.float32)
        NG_ZZ = tf.cast(tf.logical_and(K.equal(TN, 0), K.equal(N, 0)), tf.float32)

        G = K.square(tf.math.divide_no_nan(TP, P) + G_ZZ)
        NG = K.square(tf.math.divide_no_nan(TN, N) + NG_ZZ)

        # G = TP / (P + K.epsilon())
        # NG = TN / (N + K.epsilon())
        G = K.print_tensor(G, message="G")
        NG = K.print_tensor(NG, message="NG")

        L = (-G * NG) * 0.99 + bcross * 0.01

        L = K.print_tensor(L, message="L")
        return L

        # n_true_ng = tf.cast(tf.size(K.equal(y_actual, 0)), tf.float32)
        # n_true_g = tf.cast(tf.size(K.equal(y_actual, 1)), tf.float32)
        # print(n_true_ng)
        # print(n_true_g)
        # # no_grow_matches = both[np.where((both[:, 0] == 0) * (both[:, 1] == 0))]
        # # grow_matches = both[np.where((both[:, 0] == 1) * (both[:, 1] == 1))]
        # cutoff = 0.25
        # no_grow_matches = K.equal(
        #     K.equal(y_actual, 0), K.less_equal(y_predicted, cutoff)
        # )
        # grow_matches = K.equal(
        #     K.equal(y_actual, 1), K.greater_equal(y_predicted, cutoff)
        # )
        # # grow_matches = tf.where((both[:, 0] == 1) * (both[:, 1] >= cutoff))]
        # print(no_grow_matches)
        # print(grow_matches)
        # n_matches_ng = tf.cast(tf.size(no_grow_matches), tf.float32)
        # n_matches_g = tf.cast(tf.size(grow_matches), tf.float32)

        # ng_proportion = tf.math.divide_no_nan(n_matches_ng, n_true_ng)
        # g_proportion = tf.math.divide_no_nan(n_matches_g, n_true_g)

        # # loss = (ng_proportion) * (g_proportion)
        # loss = tf.math.negative(tf.math.multiply(ng_proportion, g_proportion))
        # print("LOSS:", loss)
        # return loss

        # loss = tf.py_function(func=_f, inp=[y_actual, y_predicted], Tout=tf.float32,)
        # return loss


def generate_training_data():
    m = model.load_cobra("models/iSMUv01_CDM_LOO_v2.xml")
    max_n = 10000
    with open("CDM_leave_out_validation_01.csv", mode="a") as file:
        writer = csv.writer(file, delimiter=",")
        for _ in trange(max_n):
            # n = random.randint(0, len(model.KO_RXN_IDS))
            n = sp.poisson.rvs(5)
            grow, reactions = model.knockout_and_simulate(m, n, return_boolean=True)
            reactions = list(reactions)
            reactions.append(grow)
            writer.writerow(reactions)
            # print(grow, reactions)

        for _ in trange(max_n):
            n = random.randint(0, len(model.KO_RXN_IDS))
            grow, reactions = model.knockout_and_simulate(m, n, return_boolean=True)
            reactions = list(reactions)
            reactions.append(grow)
            writer.writerow(reactions)
            # print(grow, reactions)


def load_data(filepath, mode="train", max_n=None):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: data and the corresponding labels
    """
    print(f"\nMode: {mode}")
    if mode == "train":
        raw_train = np.genfromtxt(filepath, delimiter=",")

        data_train = raw_train[1:max_n, :-1] if max_n else raw_train[1:, :-1]

        data_train_labels = raw_train[1:max_n, -1] if max_n else raw_train[1:, -1]

        x_train, y_train = (
            data_train,
            data_train_labels,
        )
        return x_train, y_train
    elif mode == "validation":
        raw_validation = np.genfromtxt(filepath, delimiter=",")
        data_validation = (
            raw_validation[1:max_n, :-1] if max_n else raw_validation[1:, :-1]
        )
        data_validation_labels = (
            raw_validation[1:max_n, -1] if max_n else raw_validation[1:, -1]
        )
        x_valid, y_valid = (
            data_validation,
            data_validation_labels,
        )
        return x_valid, y_valid
    elif mode == "test":
        raw_test = np.genfromtxt(filepath, delimiter=",")
        data_test = raw_test[1:max_n, :-1] if max_n else raw_test[1:, :-1]
        data_test_labels = raw_test[1:max_n, -1] if max_n else raw_test[1:, -1]

        x_test, y_test = data_test, data_test_labels
        return x_test, y_test


def get_stats(results, epoch, loss_name):
    TP = y_train[y_train >= 0.25].size
    TN = y_train[y_train < 0.25].size
    P = results[results == 1].size
    N = results[results == 0].size

    grow_acc = P / TP
    no_grow_acc = N / TN
    return [epoch, loss_name, grow_acc, no_grow_acc]


def get_distribution(data, pred, epoch, loss_name):
    data = pd.DataFrame(data)
    data["card"] = data.sum(axis=1)
    data["pred"] = pred

    stats = list()
    for x in range(1, 21):
        # print(f"#### CARDINALITY ({x})####")
        sub = data[data["card"] == x]
        s = sub["pred"].sum()
        # print(f"G:\t{s}/{sub.shape[0]} \t- {s/sub.shape[0]}")
        # print(
        #     f"NG:\t{sub.shape[0]-s}/{sub.shape[0]} \t- {(sub.shape[0] - s)/sub.shape[0]}"
        # )
        stats.append(
            [
                x,
                epoch,
                loss_name,
                round(s / sub.shape[0], 3),
                round(1 - (s / sub.shape[0]), 3),
            ]
        )
    return stats


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    names = [
        "L1IL2I",
        "L1OL2O",
        "L1OL2OL1IL2I",
    ]
    for name in names:
        # x_train, y_train = load_data(
        #     filepath=f"data/iSMU-test/initial_data/train_set_{name}.csv"
        # )
        # # x_valid, y_valid = load_data(filepath, max_n=10000)

        # x_test, y_test = load_data(
        #     filepath=f"data/iSMU-test/initial_data/test_set_{name}.csv", mode="test",
        # )
        x, y = load_data(filepath=f"data/iSMU-test/initial_data/train_set_{name}.csv")

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(1000)
            .batch(32)
        )
        # test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        # print(x_train, y_train)
        # print(x_test, y_test)

        # model.train(x_train, y_train, epochs=10)

        all_combos = np.genfromtxt("models/iSMU-test/data_20.csv", delimiter=",")[
            :, :-1
        ]
        df_out_acc = pd.DataFrame()
        df_out_dist = pd.DataFrame()
        epochs = [5, 10, 20, 100, 1000]
        for e in epochs:
            model_custom_batched = PredictNet(loss="custom")
            model_bc_batched = PredictNet(loss="binary_crossentropy")

            model_custom = PredictNet(loss="custom")
            model_bc = PredictNet(loss="binary_crossentropy")

            model_custom.train(x_train, y_train, epochs=e)
            model_bc.train(x_train, y_train, epochs=e)
            for x, y in train_dataset:  # only take first element of dataset
                model_custom_batched.train(x, y, epochs=e)
                model_bc_batched.train(x, y, epochs=e)

            pred1 = model_custom.predict_class(all_combos)
            pred2 = model_bc.predict_class(all_combos)
            pred3 = model_custom_batched.predict_class(all_combos)
            pred4 = model_bc_batched.predict_class(all_combos)

            dist1 = get_distribution(all_combos, pred1, e, "custom")
            dist2 = get_distribution(all_combos, pred2, e, "binary CE")
            dist3 = get_distribution(all_combos, pred3, e, "custom - te/tr split")
            dist4 = get_distribution(all_combos, pred4, e, "binary CE - te/tr split")

            results1 = get_stats(model_custom.predict_class(x_test), e, "custom")
            results2 = get_stats(model_bc.predict_class(x_test), e, "binary CE")
            results3 = get_stats(
                model_custom_batched.predict_class(x_test), e, "custom - te/tr split"
            )
            results4 = get_stats(
                model_bc_batched.predict_class(x_test), e, "binary CE - te/tr split"
            )

            df_new_dist = pd.DataFrame(dist1 + dist2 + dist3 + dist4)

            df_new_acc = pd.DataFrame([results1, results2, results3, results4])
            print(df_new_acc)
            print(df_new_dist)
            df_out_acc = pd.concat([df_out_acc, df_new_acc])
            df_out_dist = pd.concat([df_out_dist, df_new_dist])
        df_out_acc.columns = ["# Epochs", "Loss", "Grow Accuracy", "No Grow Accuracy"]
        df_out_dist.columns = ["Card", "# Epochs", "Loss", "Grow", "No Grow"]
        df_out_dist = df_out_dist.groupby(by=["Card", "# Epochs", "Loss"]).mean()
        print(df_out_dist)
        # col = list()
        # for x in range(2, 42):
        #     if x % 2 == 0:
        #         col.append(f"G ({x//2})")
        #     else:
        #         col.append(f"NG ({x//2})")

        # df_out_dist.columns = col
        df_out_acc.to_csv(f"df_out_results_acc_{name}.csv")
        df_out_dist.to_csv(f"df_out_results_dist_{name}.csv")

    # # for x, y in test_dataset:
    # model.evaluate(x_test, y_test)

    # results = model.predict_class(x_test)

    # generate_training_data()

    # scaler = MinMaxScaler()
    # max_n = 1000
    # data = np.genfromtxt('CDM_leave_out_training.csv', delimiter=',')
    # data_train = scaler.fit_transform(data[:max_n, :])
    # print(data_train)
    # x = data_train[:max_n, :-1]
    # y = np.array([[v] for v in data_train[:max_n, -1].tolist()])

    # print(y)

    # net = NeuralNetwork(x, y)
    # for _ in trange(100):
    #     net.feed_forward()
    #     net.back_propogation()

    # print(net.output)
