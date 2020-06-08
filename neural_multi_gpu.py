import csv
import datetime
import math
import os
import random
import statistics
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sp
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten
import tensorflow.keras.backend as K
from tensorflow.keras.mixed_precision import experimental

import model
import utils


class PredictNet:
    def __init__(
        self,
        exp_id,
        n_test,
        parent_logdir,
        n_layers=3,
        max_nodes=32,
        include_dropout=False,
        dropout_percent=0.0,
        lr=0.001,
        beta_1=0.90,
        beta_2=0.999,
        binary_train=False,
        n_epochs=20,
        train_batch_size=1024,
        kr_l1=0.0,
        kr_l2=0.0,
        ar_l1=0.0,
        ar_l2=0.0,
        br_l1=0.0,
        br_l2=0.0,
        leaky_relu=0.0,
        **kwargs,
    ):
        self.log_sparcity = True
        # tf.debugging.set_log_device_placement(True)
        # policy = experimental.Policy("mixed_float16")
        # experimental.set_policy(policy)
        # self.strategy = tf.distribute.MirroredStrategy()
        self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")
        # self.strategy = tf.distribute.OneDeviceStrategy(device="/device:CPU:0")
        with self.strategy.scope() as strategy:
            print("Number of devices: {}".format(self.strategy.num_replicas_in_sync))

            if n_layers == 3:
                if max_nodes == 32:
                    layer_order = [32, 8, 1]
                elif max_nodes == 256:
                    layer_order = [256, 64, 1]
                activation_types = ["relu", "relu", "sigmoid"]
            elif n_layers == 5:
                if max_nodes == 32:
                    layer_order = [32, 24, 16, 8, 1]
                elif max_nodes == 256:
                    layer_order = [256, 128, 64, 16, 1]
                activation_types = ["relu", "relu", "relu", "relu", "sigmoid"]

            kr = None
            if kr_l1 or kr_l2:
                kr = tf.keras.regularizers.l1(kr_l1)
                # kr = tf.keras.regularizers.l1_l2(l1=kr_l1, l2=kr_l2)
            ar = None
            if ar_l1 or ar_l2:
                ar = tf.keras.regularizers.l1(kr_l1)
                # ar = tf.keras.regularizers.l1_l2(l1=ar_l1, l2=ar_l2)
            br = None
            if br_l1 or br_l2:
                br = tf.keras.regularizers.l1(kr_l1)
                # br = tf.keras.regularizers.l1_l2(l1=br_l1, l2=br_l2)

            layers = []
            for idx, (n, a) in enumerate(zip(layer_order, activation_types)):
                if a == "leaky_relu":
                    a = None
                layers.append(
                    tf.keras.layers.Dense(
                        n,
                        activation=a,
                        kernel_regularizer=kr,
                        activity_regularizer=ar,
                        bias_regularizer=br,
                    )
                )
                if a == None:
                    layers.append(tf.keras.layers.LeakyReLU(alpha=leaky_relu))

                if include_dropout and idx != len(layer_order):
                    layers.append(
                        tf.keras.layers.Dropout(
                            dropout_percent, noise_shape=None, seed=None
                        )
                    )

            self.model = tf.keras.Sequential(layers)
            # if loss == "custom":
            #     loss = self.custom_loss
            self.loss_object = tf.keras.losses.BinaryCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            )
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=lr, beta_1=beta_1, beta_2=beta_2
            )

            self.model.compile(loss=self.loss_object, optimizer=self.optimizer)

            self.train_loss = tf.keras.metrics.Mean(name="train_loss")
            self.train_mse = tf.keras.metrics.MeanSquaredError(name="train_mse")
            self.train_accuracy = tf.keras.metrics.Accuracy(name="train_accuracy")
            self.test_loss = tf.keras.metrics.Mean(name="test_loss")
            self.test_mse = tf.keras.metrics.MeanSquaredError(name="test_mse")
            self.test_accuracy = tf.keras.metrics.Accuracy(name="test_accuracy")

            current_time = datetime.datetime.now().strftime(
                "%Y%m%d-%H%M%S" + f"-{exp_id}"
            )
            # parent_dir = "tensorboard_logs/fractional_factorial_results_100000/"
            current_run_dir = os.path.join(parent_logdir, current_time)
            print("Saving logs to " + current_run_dir)
            train_log_dir = os.path.join(current_run_dir, "train")
            test_log_dir = os.path.join(current_run_dir, "test")

            # self.profiler_log_dir = os.path.join(current_run_dir, "profile")
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
            # self.profile_writer = tf.summary.create_file_writer(self.profiler_log_dir)

            self.binary_train = binary_train
            self.n_epochs = n_epochs
            self.train_batch_size = train_batch_size
            self.n_test = n_test
            self.growth_cutoff = 0.25

    def get_mse(self, y_true, y_pred, **kwargs):
        return K.mean(K.square(y_true - y_pred))

    @tf.function
    def distributed_train_step(self, dist_inputs, global_batch_size):
        with self.strategy.scope() as strategy:

            def compute_loss(labels, predictions):
                per_example_loss = self.loss_object(labels, predictions)
                # per_example_loss = self.model.
                # return tf.nn.compute_average_loss(
                #     per_example_loss, global_batch_size=global_batch_size
                weighted_loss = tf.reduce_sum(per_example_loss) * (
                    1.0 / global_batch_size
                )
                return weighted_loss

            def train_step(inputs):
                inputs, labels = inputs
                with tf.GradientTape() as tape:
                    # training=True is only needed if there are layers with different
                    # behavior during training versus inference (e.g. Dropout).
                    predictions = self.model(inputs, training=True)
                    loss = compute_loss(labels, predictions)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )
                self.train_mse.update_state(labels, predictions)

                predictions = tf.cast(
                    K.greater_equal(predictions, self.growth_cutoff), tf.float32
                )
                labels = tf.cast(
                    K.greater_equal(labels, self.growth_cutoff), tf.float32
                )
                self.train_accuracy.update_state(labels, predictions)
                return loss

            per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
            return strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
            )

    # def train_step(self, inputs, labels):
    #     with tf.GradientTape() as tape:
    #         # training=True is only needed if there are layers with different
    #         # behavior during training versus inference (e.g. Dropout).
    #         predictions = self.model(inputs, training=True)
    #         loss = self.loss_object(labels, predictions)
    #     gradients = tape.gradient(loss, self.model.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    #     self.train_loss.update_state(loss)
    #     self.train_mse.update_state(labels, predictions)
    #     # self.train_loss.append(loss.numpy())
    #     # self.train_mse.append(self.get_mse(labels, predictions).numpy())
    @tf.function
    def distributed_test_step(self, dist_inputs):
        with self.strategy.scope() as strategy:

            def test_step(inputs):
                inputs, labels = inputs
                predictions = self.model(inputs, training=False)
                t_loss = self.loss_object(labels, predictions)

                self.test_mse.update_state(labels, predictions)
                self.test_loss.update_state(t_loss)

                predictions = tf.cast(
                    K.greater_equal(predictions, self.growth_cutoff), tf.float32
                )
                labels = tf.cast(
                    K.greater_equal(labels, self.growth_cutoff), tf.float32
                )

                self.test_accuracy.update_state(labels, predictions)
                # return t_loss

            # per_replica_losses = strategy.run(test_step, args=(dist_inputs,))
            # return strategy.reduce(
            #     tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
            # )
            return strategy.run(test_step, args=(dist_inputs,))

    # def test_step(self, inputs):
    #     # training=False is only needed if there are layers with different
    #     # behavior during training versus inference (e.g. Dropout).
    #     predictions = self.model(inputs, training=False)
    #     loss = self.loss_object(labels, predictions)

    #     self.test_loss.update_state(loss)
    #     self.test_mse.update_state(labels, predictions)

    def train(
        self, x_train, y_train, x_test=None, y_test=None,
    ):

        with self.strategy.scope() as strategy:
            n_replicas = strategy.num_replicas_in_sync
            train_buffer = y_train.size
            global_batch_size = self.train_batch_size * n_replicas
            if isinstance(y_test, np.ndarray):
                test_buffer = y_test.size

            y = np.copy(y_train)
            # Train the model here
            if self.binary_train:
                y[y >= self.growth_cutoff] = 1.0
                y[y < self.growth_cutoff] = 0

            train_ds = strategy.experimental_distribute_dataset(
                tf.data.Dataset.from_tensor_slices((x_train, y))
                .shuffle(train_buffer)
                .batch(global_batch_size)
                .prefetch(tf.data.experimental.AUTOTUNE)
                # .cache()
            )

            if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
                test_ds = strategy.experimental_distribute_dataset(
                    tf.data.Dataset.from_tensor_slices((x_test, y_test))
                    .shuffle(test_buffer)
                    .take(self.n_test)
                    .batch(self.n_test)
                    .prefetch(tf.data.experimental.AUTOTUNE)
                    # .cache()
                )

            for epoch in range(self.n_epochs):
                # Start profiler
                # with self.profile_writer.as_default():
                #     tf.summary.trace_on(graph=True, profiler=False)

                # Reset the metrics at the start of the next epoch
                self.train_loss.reset_states()
                self.train_mse.reset_states()
                self.train_accuracy.reset_states()
                self.test_loss.reset_states()
                self.test_mse.reset_states()
                self.test_accuracy.reset_states()

                total_loss = 0.0
                num_batches = 0
                for inputs in train_ds:
                    total_loss += self.distributed_train_step(inputs, global_batch_size)
                    num_batches += 1
                train_loss = total_loss / num_batches
                # self.train_loss.update_state(train_loss)

                if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
                    for inputs in test_ds:
                        self.distributed_test_step(inputs)

                with self.train_summary_writer.as_default():
                    tf.summary.scalar("loss", train_loss, step=epoch)
                    tf.summary.scalar("mse", self.train_mse.result(), step=epoch)
                    tf.summary.scalar(
                        "accuracy", self.train_accuracy.result(), step=epoch
                    )
                    if self.log_sparcity:
                        k_num = 0
                        k_den = 0
                        b_num = 0
                        b_den = 0
                        for layer in self.model.layers:
                            k, b = layer.get_weights()
                            k_num += k[np.absolute(k) < 1e-3].size
                            k_den += k.size
                            b_num += b[np.absolute(b) < 1e-3].size
                            b_den += b.size

                        k_sparcity = k_num / k_den
                        b_sparcity = b_num / b_den
                        tf.summary.scalar("k_sparcity", k_sparcity, step=epoch)
                        tf.summary.scalar("b_sparcity", b_sparcity, step=epoch)
                        tf.summary.scalar("a_sparcity", 0, step=epoch)

                if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
                    with self.test_summary_writer.as_default():
                        tf.summary.scalar("loss", self.test_loss.result(), step=epoch)
                        tf.summary.scalar("mse", self.test_mse.result(), step=epoch)
                        tf.summary.scalar(
                            "accuracy", self.test_accuracy.result(), step=epoch
                        )

                template = "Epoch {} - Loss: {:.3f}, MSE: {:.3f}, Acc: {:.3f}, Test Loss: {:.3f}, Test MSE: {:.3f}, Test Acc: {:.3f},"
                print(
                    template.format(
                        epoch + 1,
                        train_loss,
                        self.train_mse.result(),
                        self.train_accuracy.result(),
                        self.test_loss.result(),
                        self.test_mse.result(),
                        self.test_accuracy.result(),
                    )
                )
            return self.train_accuracy.result(), self.test_accuracy.result()

    def evaluate(self, data, data_labels):
        predictions = self.predict_class(data)
        data_labels = data_labels >= self.growth_cutoff
        data_labels = data_labels.astype(np.int)

        accuracy = metrics.accuracy_score(data_labels, predictions)
        precision = metrics.precision_score(data_labels, predictions)
        recall = metrics.recall_score(data_labels, predictions)
        return accuracy, precision, recall

    def predict_probability(self, data):
        return self.model.predict_proba(data)

    def predict_class(self, data):
        predictions = self.model.predict(data)
        predictions = predictions >= self.growth_cutoff
        predictions = predictions.astype(np.int)
        return predictions

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
        N = K.equal(y_true, 0)
        P = K.equal(y_true, 1)
        TN = tf.logical_and(K.equal(y_true, 0), K.less(y_pred, self.growth_cutoff))
        TP = tf.logical_and(
            K.equal(y_true, 1), K.greater_equal(y_pred, self.growth_cutoff)
        )

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


def load_data(filepath, starting_index=1, max_n=None):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: data and the corresponding labels
    """

    raw_data = np.genfromtxt(filepath, delimiter=",", dtype=np.float32)[1:, :]
    if max_n:
        raw_data = raw_data[
            np.random.choice(raw_data.shape[0], size=max_n, replace=False), :
        ]
    data = raw_data[starting_index:, :-1]
    data_labels = raw_data[starting_index:, -1]
    return data, data_labels


def get_stats(y_true, results, epoch, loss_name):
    y_true[y_true >= self.growth_cutoff] = 1
    y_true[y_true < self.growth_cutoff] = 0
    results[results >= self.growth_cutoff] = 1
    results[results < self.growth_cutoff] = 0

    TN, FP, FN, TP = metrics.confusion_matrix(y_true, results).ravel()

    grow_acc = TP / (TP + FP)
    no_grow_acc = TN / (TN + FN)
    return [epoch, loss_name, grow_acc, no_grow_acc, TN, TP, FN, FP]


def get_distribution(data, pred, epoch, loss_name):
    data = pd.DataFrame(data)
    data["card"] = data.sum(axis=1)
    data["pred"] = pred

    stats = []
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

    N_TEST = 100000

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    print(f"GPU: {tf.test.is_built_with_cuda()}")

    # experimental_design, design_low_high = utils.create_fractional_factorial_experiment(
    #     "files/fractional_design_k10n128.csv",
    #     "files/hyperparameters_regularization.csv",
    # )
    experimental_design = pd.read_csv(
        "data/neuralpy_optimization_expts/042320/experiments_sparcity_2.csv",
        index_col=0,
    )
    # design_low_high["train_accuracy"] = 0.0
    # design_low_high["test_accuracy"] = 0.0

    x, y = load_data(
        filepath=f"models/iSMU-test/data_20_extrapolated.csv", starting_index=0,
    )

    # sizes_to_test = list([x / 100 for x in range(0, 101, 5)])
    # interval = 10
    # sizes_to_test = (
    #     [0.0001] + list([x / 100 for x in range(interval, 100, interval)]) + [0.9999]
    # )
    sizes_to_test = [0.50]

    data_out_train_acc = pd.DataFrame(sizes_to_test, columns=["train_percentage"])
    data_out_test_acc = pd.DataFrame(sizes_to_test, columns=["train_percentage"])
    data_out_runtimes = pd.DataFrame(sizes_to_test, columns=["train_percentage"])

    for index, row in experimental_design.iterrows():
        train_accuracies = []
        test_accuracies = []
        runtimes = []
        for train_size in sizes_to_test:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=train_size
            )

            start = time.time()
            print(
                f"--------------------------- Starting Experiment {index} - {train_size*100}% split ----------------------------"
            )
            hyperparameters = row.to_dict()
            print(hyperparameters, "\n")

            # x_train, y_train = load_data(
            #     filepath=f"data/iSMU-test/initial_data/train_set_L1OL2OL1IL2I.csv",
            #     starting_index=0,
            # )

            model = PredictNet(
                n_test=N_TEST,
                exp_id=index,
                parent_logdir=f"tensorboard_logs/sparcity_eval/{train_size*100}-split-256",
                **hyperparameters,
            )

            train_acc, test_acc = model.train(x_train, y_train, x_test, y_test)
            # experimental_design.loc[index, "train_accuracy"] = train_acc.numpy()
            # experimental_design.loc[index, "test_accuracy"] = test_acc.numpy()
            runtime = round(time.time() - start, 2)
            train_accuracies.append(train_acc.numpy())
            test_accuracies.append(test_acc.numpy())
            runtimes.append(runtime)
            print(f"Total time to complete: {runtime} sec")

            predictions = pd.DataFrame(model.model.predict(x), columns=["predictions"])
            predictions.to_csv("predictions.csv")

        new_data_train_acc = pd.DataFrame(train_accuracies, columns=[f"{int(index)}"])
        new_data_test_acc = pd.DataFrame(test_accuracies, columns=[f"{int(index)}"])
        new_data_runtimes = pd.DataFrame(runtimes, columns=[f"{int(index)}"])

        data_out_train_acc = pd.concat([data_out_train_acc, new_data_train_acc], axis=1)
        data_out_test_acc = pd.concat([data_out_test_acc, new_data_test_acc], axis=1)
        data_out_runtimes = pd.concat([data_out_runtimes, new_data_runtimes], axis=1)
        # design_low_high["train_accuracy"] = experimental_design["train_accuracy"]
        # design_low_high["test_accuracy"] = experimental_design["test_accuracy"]
        data_out_train_acc.to_csv(
            "data/neuralpy_optimization_expts/042320/sparcity_eval_train_results.csv"
        )
        data_out_test_acc.to_csv(
            "data/neuralpy_optimization_expts/042320/sparcity_eval_test_results.csv"
        )
        data_out_runtimes.to_csv(
            "data/neuralpy_optimization_expts/042320/sparcity_eval_runtime_results.csv"
        )
    # experimental_design.to_csv(f"regularization_eval-{index}.csv")
    # design_low_high.to_csv(
    #     "fractional_factorial_results_low_high_100000_50split_regularization.csv"
    # )
