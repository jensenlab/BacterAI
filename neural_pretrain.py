import argparse
import copy
import csv
import datetime
import math
import os
import pickle
import pprint
import random
import statistics
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sp
from sklearn.model_selection import train_test_split
from sklearn import metrics
from termcolor import colored
import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
from tensorflow.keras.mixed_precision import experimental
from tensorflow.python.keras.engine import compile_utils

import model
import utils


class PredictNet:
    def __init__(
        self,
        exp_id,
        n_test,
        parent_logdir,
        save_model_path=None,
        n_layers=3,
        max_nodes=32,
        include_dropout=False,
        dropout_percent=0.0,
        lr=0.001,
        beta_1=0.90,
        beta_2=0.999,
        binary_train=False,
        n_epochs=100,
        n_retrain_epochs=200,
        train_batch_size=1024,
        kr_l1=0.001,
        ar_l1=0.001,
        br_l1=0.001,
        leaky_relu=0.0,
        k_init="glorot_uniform",
        log_sparcity=True,
        **kwargs,
    ):

        self.n_test = n_test
        self.exp_id = exp_id
        self.parent_logdir = parent_logdir
        self.save_model_path = save_model_path

        self.n_layers = n_layers
        self.max_nodes = max_nodes
        self.include_dropout = include_dropout
        self.dropout_percent = dropout_percent
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.binary_train = binary_train
        self.n_epochs = n_epochs
        self.n_retrain_epochs = n_retrain_epochs
        self.train_batch_size = train_batch_size
        self.kr_l1 = kr_l1
        self.ar_l1 = ar_l1
        self.br_l1 = br_l1
        self.leaky_relu = leaky_relu
        self.k_init = k_init
        self.log_sparcity = log_sparcity

        if n_layers == 3:
            if max_nodes == 32:
                layer_order = [32, 16, 1]
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
        if kr_l1:
            kr = tf.keras.regularizers.l1(kr_l1)
        ar = None
        if ar_l1:
            ar = tf.keras.regularizers.l1(ar_l1)
        br = None
        if br_l1:
            br = tf.keras.regularizers.l1(br_l1)

        layers = []
        for idx, (n, a) in enumerate(zip(layer_order, activation_types)):
            if a == "leaky_relu":
                a = None
            layers.append(
                tf.keras.layers.Dense(
                    n,
                    activation=a,
                    # kernel_initializer=k_init,
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
        self.loss_object = tf.keras.losses.BinaryCrossentropy()
        self.compiled_loss = compile_utils.LossesContainer(
            self.loss_object, output_names=self.model.output_names
        )
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=beta_1, beta_2=beta_2
        )

        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_mse = tf.keras.metrics.MeanSquaredError(name="train_mse")
        self.train_accuracy = tf.keras.metrics.Accuracy(name="train_accuracy")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_mse = tf.keras.metrics.MeanSquaredError(name="test_mse")
        self.test_accuracy = tf.keras.metrics.Accuracy(name="test_accuracy")

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S" + f"-{exp_id}")
        current_run_dir = os.path.join(parent_logdir, current_time)
        print(colored(f"Saving logs to {current_run_dir}", "cyan"))

        train_log_dir = os.path.join(current_run_dir, "train")
        test_log_dir = os.path.join(current_run_dir, "test")

        # self.profiler_log_dir = os.path.join(current_run_dir, "profile")
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        # self.profile_writer = tf.summary.create_file_writer(self.profiler_log_dir)

        self.growth_cutoff = 0.25

        self.accuracy_tracker = []
        self.k_sparcity_tracker = []
        self.b_sparcity_tracker = []
        self.n_grow_tracker = []
        self.n_grow_correct_tracker = []
        self.n_no_grow_tracker = []
        self.n_no_grow_correct_tracker = []

    def save(self):
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        save_model_path = os.path.join(self.save_model_path, "model.h5")
        self.model.save(save_model_path)

        params = {
            "exp_id": self.exp_id,
            "n_test": self.n_test,
            "parent_logdir": self.parent_logdir,
            "save_model_path": self.save_model_path,
            "n_layers": self.n_layers,
            "max_nodes": self.max_nodes,
            "include_dropout": self.include_dropout,
            "dropout_percent": self.dropout_percent,
            "lr": self.lr,
            "beta_1": self.beta_1,
            "beta_2": self.beta_2,
            "binary_train": self.binary_train,
            "n_epochs": self.n_epochs,
            "n_retrain_epochs": self.n_retrain_epochs,
            "train_batch_size": self.train_batch_size,
            "kr_l1": self.kr_l1,
            "ar_l1": self.ar_l1,
            "br_l1": self.br_l1,
            "leaky_relu": self.leaky_relu,
            "k_init": self.k_init,
        }
        save_path = os.path.join(self.save_model_path, "model_params.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(params, f)

    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path)
        self.model = model

    @classmethod
    def from_save(cls, save_dir):
        model_params_path = os.path.join(save_dir, "model_params.pkl")
        with open(model_params_path, "rb") as f:
            model_params = pickle.load(f)
        model_path = os.path.join(model_params["save_model_path"], "model.h5")
        obj = cls(**model_params)
        obj.load_model(model_path)
        return obj

    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(inputs, training=True)
            # loss = self.loss_object(labels, predictions)
            loss = self.compiled_loss(
                labels, predictions, regularization_losses=self.model.losses
            )

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # print(gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_mse.update_state(labels, predictions)

        predictions = tf.cast(
            K.greater_equal(predictions, self.growth_cutoff), tf.float32
        )
        labels = tf.cast(K.greater_equal(labels, self.growth_cutoff), tf.float32)
        self.train_accuracy.update_state(labels, predictions)
        # self.train_loss.append(loss.numpy())

    def test_step(self, inputs, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(inputs, training=False)
        loss = self.loss_object(labels, predictions)

        self.test_loss.update_state(loss)
        self.test_mse.update_state(labels, predictions)

        predictions = tf.cast(
            K.greater_equal(predictions, self.growth_cutoff), tf.float32
        )
        labels = tf.cast(K.greater_equal(labels, self.growth_cutoff), tf.float32)
        # print(predictions, labels)
        self.test_accuracy.update_state(labels, predictions)

    def train(
        self,
        x_train,
        y_train,
        x_test=None,
        y_test=None,
        is_retrain=False,
        evaluate_distribution=False,
    ):

        train_buffer = y_train.size
        if isinstance(y_test, np.ndarray):
            test_buffer = y_test.size

        y = np.copy(y_train)
        # Train the model here
        if self.binary_train:
            y[y >= self.growth_cutoff] = 1.0
            y[y < self.growth_cutoff] = 0

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y))
            .shuffle(train_buffer)
            .batch(self.train_batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        if evaluate_distribution:
            all_x = np.copy(x_train)
            all_y = np.copy(y_train)

        if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
            if evaluate_distribution:
                all_x = np.concatenate([all_x, x_test], axis=0)
                all_y = np.concatenate([all_y, y_test], axis=0)
            test_ds = (
                tf.data.Dataset.from_tensor_slices((x_test, y_test))
                .shuffle(test_buffer)
                .take(self.n_test)
                .batch(self.train_batch_size * 2)
                .prefetch(tf.data.experimental.AUTOTUNE)
            )

        epochs = self.n_retrain_epochs if is_retrain else self.n_epochs
        for epoch in range(epochs):
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

            for inputs, labels in train_ds:
                self.train_step(inputs, labels)
            # self.train_loss.update_state(train_loss)

            if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
                for inputs, labels in test_ds:
                    self.test_step(inputs, labels)

            if evaluate_distribution:
                (
                    num_grow,
                    grow_correct,
                    num_no_grow,
                    no_grow_correct,
                ) = self.evaluate_distribution(
                    all_x, all_y, cardinality=20, by_cardinality=False
                )
                self.n_grow_tracker.append(num_grow)
                self.n_grow_correct_tracker.append(grow_correct)
                self.n_no_grow_tracker.append(num_no_grow)
                self.n_no_grow_correct_tracker.append(no_grow_correct)

            with self.train_summary_writer.as_default():
                tf.summary.scalar("loss", self.train_loss.result(), step=epoch)
                tf.summary.scalar("mse", self.train_mse.result(), step=epoch)
                tf.summary.scalar("accuracy", self.train_accuracy.result(), step=epoch)
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
                else:
                    k_sparcity = "N/A"
                    b_sparcity = "N/A"

            if isinstance(x_test, np.ndarray) and isinstance(y_test, np.ndarray):
                with self.test_summary_writer.as_default():
                    tf.summary.scalar("loss", self.test_loss.result(), step=epoch)
                    tf.summary.scalar("mse", self.test_mse.result(), step=epoch)
                    tf.summary.scalar(
                        "accuracy", self.test_accuracy.result(), step=epoch
                    )

            results = "{} |\tTrain - Acc: {} MSE: {} |\tTest - Acc: {} MSE: {} |\tSparcity - K: {} B: {}"
            results = results.format(
                colored(epoch + 1, "yellow"),
                colored(f"{self.train_accuracy.result():.3f}", "green"),
                colored(f"{self.train_mse.result():.3f}", "green"),
                colored(f"{self.test_accuracy.result():.3f}", "green"),
                colored(f"{self.test_mse.result():.3f}", "green"),
                colored(f"{k_sparcity:.3f}", "green"),
                colored(f"{b_sparcity:.3f}", "green"),
            )

            print(results)

            if evaluate_distribution:
                self.accuracy_tracker.append(self.test_accuracy.result().numpy())
                self.k_sparcity_tracker.append(k_sparcity)
                self.b_sparcity_tracker.append(b_sparcity)

        if self.save_model_path:
            # if not os.path.exists(self.save_model_path):
            #     os.makedirs(self.save_model_path)
            # save_path = os.path.join(self.save_model_path, "model.h5")
            # self.model.save(save_path)
            self.save()

        return (
            self.train_accuracy.result(),
            self.test_accuracy.result(),
            k_sparcity,
            b_sparcity,
        )

    def pretrain_add_layer(self, x_train, y_train, x_test, y_test):
        # remember the current output layer
        output_layer = self.model.layers[-1]
        # remove the output layer
        self.model.pop()
        # mark all remaining layers as non-trainable
        for layer in self.model.layers:
            layer.trainable = False

        # add a new hidden layer
        kr = None
        if self.kr_l1:
            kr = tf.keras.regularizers.l1(self.kr_l1)
        ar = None
        if self.ar_l1:
            ar = tf.keras.regularizers.l1(self.ar_l1)
        br = None
        if self.br_l1:
            br = tf.keras.regularizers.l1(self.br_l1)

        new_layer = tf.keras.layers.Dense(
            16,
            activation="relu",
            kernel_regularizer=kr,
            activity_regularizer=ar,
            bias_regularizer=br,
        )

        self.model.add(new_layer)
        # re-add the output layer
        self.model.add(output_layer)
        # fit model
        train_acc, test_acc, k_sparcity, b_sparcity = self.train(
            x_train, y_train, x_test, y_test
        )
        return train_acc, test_acc, k_sparcity, b_sparcity

    def evaluate(self, data, data_labels):
        predictions = self.predict_class(data)
        data_labels = data_labels >= self.growth_cutoff
        data_labels = data_labels.astype(np.int)

        accuracy = metrics.accuracy_score(data_labels, predictions)
        precision = metrics.precision_score(data_labels, predictions)
        recall = metrics.recall_score(data_labels, predictions)
        return accuracy, precision, recall

    def evaluate_distribution(
        self, data, data_labels, cardinality, by_cardinality=True
    ):
        predictions = self.predict_class(data)
        data_labels = data_labels >= self.growth_cutoff
        data_labels = data_labels.astype(np.int)

        data = pd.DataFrame(data[:, :cardinality])
        data["cardinality"] = data.sum(axis=1)
        data["prediction"] = predictions
        data["grow"] = data_labels

        if by_cardinality:
            distribution = pd.DataFrame(
                {"cardinality": list(range(1, cardinality + 1))}
            )
            for c in range(1, cardinality + 1):
                results = data.loc[data["cardinality"] == c, ["prediction", "grow"]]
                grow_results = results.loc[results["grow"] == 1, "prediction"]
                no_grow_results = results.loc[results["grow"] == 0, "prediction"]

                num_grow = grow_results.shape[0]
                num_no_grow = no_grow_results.shape[0]

                grow_correct = sum(grow_results == 1)
                no_grow_correct = sum(no_grow_results == 0)

                distribution.loc[c, "total_grow"] = num_grow
                distribution.loc[c, "n_correct_grow"] = grow_correct
                distribution.loc[c, "p_correct_grow"] = grow_correct / num_grow

                distribution.loc[c, "total_no_grow"] = num_no_grow
                distribution.loc[c, "n_correct_no_grow"] = no_grow_correct
                distribution.loc[c, "p_correct_no_grow"] = no_grow_correct / num_no_grow
            return distribution
        else:
            results = data.loc[:, ["prediction", "grow"]]
            grow_results = results.loc[results["grow"] == 1, "prediction"]
            no_grow_results = results.loc[results["grow"] == 0, "prediction"]

            num_grow = grow_results.shape[0]
            num_no_grow = no_grow_results.shape[0]

            grow_correct = sum(grow_results == 1)
            no_grow_correct = sum(no_grow_results == 0)

            return num_grow, grow_correct, num_no_grow, no_grow_correct

    def predict_probability(self, data):
        return self.model.predict_proba(data)

    def predict_class(self, data):
        predictions = self.model.predict(data)
        predictions = predictions >= self.growth_cutoff
        predictions = predictions.astype(np.int)
        return predictions

    # # Hotfix function
    # def make_keras_picklable(self):
    #     def __reduce__(self):
    #         def unpack(model, training_config, weights):
    #             restored_model = tf.keras.layers.deserialize(model)
    #             if training_config is not None:
    #                 restored_model.compile(
    #                     **tf.python.keras.saving.saving_utils.compile_args_from_training_config(
    #                         training_config
    #                     )
    #                 )
    #             restored_model.set_weights(weights)
    #             return restored_model

    #         model_metadata = tf.python.keras.saving.saving_utils.model_metadata(self)
    #         training_config = model_metadata.get("training_config", None)
    #         model = tf.keras.layers.serialize(self)
    #         weights = self.get_weights()
    #         return (unpack, (model, training_config, weights))

    #     cls = tf.keras.models.Model
    #     cls.__reduce__ = __reduce__


def pretrain_scheme(
    experiment_dir,
    design_file_name,
    save_file_name,
    n_pretrain_layers,
    n_test,
    train_sizes,
    save_model_path=None,
    final_retrain=False,
    evaluate_distribution=False,
):
    experimental_design = pd.read_csv(
        os.path.join(experiment_dir, design_file_name), index_col=0,
    )
    x, y = load_data(
        filepath=f"models/iSMU-test/data_20_extrapolated.csv",
        # filepath=f"models/iSMU-test/data_20_extrapolated_with_features.csv",
        starting_index=0,
    )

    # data_out_train_acc = pd.DataFrame(
    #     list(range((n_layers + int(final_retrain)) + 1)), columns=["n_pretrain_layers"],
    # )
    # data_out_test_acc = pd.DataFrame(
    #     list(range((n_layers + int(final_retrain)) + 1)), columns=["n_pretrain_layers"],
    # )

    data_out_train_acc = pd.DataFrame(
        {"type": "train_accuracy", "train_percentage": train_sizes}
    )
    data_out_test_acc = pd.DataFrame(
        {"type": "test_accuracy", "train_percentage": train_sizes}
    )
    data_out_runtimes = pd.DataFrame(
        {"type": "runtime", "train_percentage": train_sizes}
    )
    data_out_k_sparcity = pd.DataFrame(
        {"type": "k_sparcity", "train_percentage": train_sizes}
    )
    data_out_b_sparcity = pd.DataFrame(
        {"type": "b_sparcity", "train_percentage": train_sizes}
    )

    for index, row in experimental_design.iterrows():
        train_accuracies = []
        test_accuracies = []
        runtimes = []
        k_sparcities = []
        b_sparcities = []
        for train_size in train_sizes:

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=train_size, random_state=12345
            )

            start = time.time()
            print(
                colored(
                    f"--------------------------- Starting Experiment {index} - {train_size*100}% split (Pretrain Scheme) ----------------------------",
                    "green",
                    attrs=["bold"],
                )
            )
            hyperparameters = row.to_dict()

            pprint.pprint(hyperparameters)

            model = PredictNet(
                n_test=n_test,
                exp_id=index,
                parent_logdir=f"tensorboard_logs/sparcity_eval/{train_size*100}_split-exp_{index}",
                save_model_path=save_model_path,
                **hyperparameters,
            )

            # Initial train
            # print(colored("\nInitial Train Started", "cyan"))
            # train_acc, test_acc, k_sparcity, b_sparcity = model.train(
            #     x_train, y_train, x_test, y_test
            # )
            # train_accuracies.append(train_acc.numpy())
            # test_accuracies.append(test_acc.numpy())

            # More layers
            for l in range(n_pretrain_layers):
                print(
                    colored(
                        f"\nPretrain - Layer {l+1}/{n_pretrain_layers} started", "cyan"
                    )
                )
                train_acc, test_acc, k_sparcity, b_sparcity = model.pretrain_add_layer(
                    x_train, y_train, x_test, y_test
                )
                # train_accuracies.append(train_acc.numpy())
                # test_accuracies.append(test_acc.numpy())

                # model_temp = tf.keras.models.clone_model(model.model)
                # model.model = tf.keras.models.clone_model(model.model)
                # for layer in model.model.layers:
                #     layer.trainable = True
                # train_acc, test_acc, k_sparcity, b_sparcity = model.train(x_train, y_train, x_test, y_test)
                # train_accuracies.append(train_acc.numpy())
                # test_accuracies.append(test_acc.numpy())
                # model.model = model_temp

            # Retrain all weights one last time
            if final_retrain:
                print(colored(f"\nFinal Retrain Started", "cyan"))
                for layer in model.model.layers:
                    layer.trainable = True
                train_acc, test_acc, k_sparcity, b_sparcity = model.train(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    is_retrain=True,
                    evaluate_distribution=evaluate_distribution,
                )
            runtime = round(time.time() - start, 2)

            train_accuracies.append(train_acc.numpy())
            test_accuracies.append(test_acc.numpy())
            runtimes.append(runtime)
            k_sparcities.append(k_sparcity)
            b_sparcities.append(b_sparcity)

            print(colored(f"Total time to complete: {runtime} sec", "magenta"))

            if evaluate_distribution:
                tracker_data = pd.DataFrame(
                    {
                        "accuracy": model.accuracy_tracker,
                        "k_sparcity": model.k_sparcity_tracker,
                        "b_sparcity": model.b_sparcity_tracker,
                        "n_grows": model.n_grow_tracker,
                        "n_grow_correct": model.n_grow_correct_tracker,
                        "n_no_grows": model.n_no_grow_tracker,
                        "n_no_grow_correct": model.n_no_grow_correct_tracker,
                    }
                )

                tracker_data.to_csv(
                    os.path.join(
                        experiment_dir,
                        "tracker_data",
                        f"EXP{index}_{train_size}_data.csv",
                    )
                )

        new_data_train_acc = pd.DataFrame(train_accuracies, columns=[f"{int(index)}"])
        new_data_test_acc = pd.DataFrame(test_accuracies, columns=[f"{int(index)}"])
        new_data_runtimes = pd.DataFrame(runtimes, columns=[f"{int(index)}"])
        new_data_k_sparcity = pd.DataFrame(k_sparcities, columns=[f"{int(index)}"])
        new_data_b_sparcity = pd.DataFrame(b_sparcities, columns=[f"{int(index)}"])

        data_out_train_acc = pd.concat([data_out_train_acc, new_data_train_acc], axis=1)
        data_out_test_acc = pd.concat([data_out_test_acc, new_data_test_acc], axis=1)
        data_out_runtimes = pd.concat([data_out_runtimes, new_data_runtimes], axis=1)
        data_out_k_sparcity = pd.concat(
            [data_out_k_sparcity, new_data_k_sparcity], axis=1
        )
        data_out_b_sparcity = pd.concat(
            [data_out_b_sparcity, new_data_b_sparcity], axis=1
        )

        data_out = pd.concat(
            [
                data_out_train_acc,
                data_out_test_acc,
                data_out_runtimes,
                data_out_k_sparcity,
                data_out_b_sparcity,
            ],
            axis=0,
        )
        data_out.to_csv(os.path.join(experiment_dir, save_file_name))
        print(
            "ZERO:",
            model.predict_class(
                np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
            ),
        )
        # data_out_train_acc.to_csv(
        #     os.path.join(
        #         experiment_dir, f"pretrain_eval_train_results_final_retrain.csv",
        #     )
        # )
        # data_out_test_acc.to_csv(
        #     os.path.join(
        #         experiment_dir, f"pretrain_eval_test_results_final_retrain.csv",
        #     )
        # )


def standard_train_scheme(experiment_dir, design_file_name, train_sizes, n_test):
    # experimental_design, design_low_high = utils.create_fractional_factorial_experiment(
    #     "files/fractional_design_k10n128.csv",
    #     "files/hyperparameters_regularization.csv",
    # )
    experimental_design = pd.read_csv(
        os.path.join(experiment_dir, design_file_name), index_col=0,
    )
    # design_low_high["train_accuracy"] = 0.0
    # design_low_high["test_accuracy"] = 0.0

    x, y = load_data(
        filepath=f"models/iSMU-test/data_20_extrapolated.csv",
        # filepath=f"models/iSMU-test/data_20_extrapolated_with_features.csv",
        starting_index=0,
    )

    for index, row in experimental_design.iterrows():
        train_accuracies = []
        test_accuracies = []
        runtimes = []
        for train_size in train_sizes:
            data_out_train_acc = pd.DataFrame(train_sizes, columns=["train_percentage"])
            data_out_test_acc = pd.DataFrame(train_sizes, columns=["train_percentage"])
            data_out_runtimes = pd.DataFrame(train_sizes, columns=["train_percentage"])

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, train_size=train_size, random_state=12345
            )

            start = time.time()
            print(
                colored(
                    f"--------------------------- Starting Experiment {index} - {train_size*100}% split (Std. Scheme) ----------------------------",
                    "green",
                    attrs=["bold"],
                )
            )
            hyperparameters = row.to_dict()
            print(hyperparameters, "\n")

            # x_train, y_train = load_data(
            #     filepath=f"data/iSMU-test/initial_data/train_set_L1OL2OL1IL2I.csv",
            #     starting_index=0,
            # )

            model = PredictNet(
                n_test=n_test,
                exp_id=index,
                parent_logdir=f"tensorboard_logs/init_eval/{train_size*100}-split",
                **hyperparameters,
            )

            train_acc, test_acc = model.train(x_train, y_train, x_test, y_test)

            runtime = round(time.time() - start, 2)
            train_accuracies.append(train_acc.numpy())
            test_accuracies.append(test_acc.numpy())
            runtimes.append(runtime)

            print(colored(f"Total time to complete: {runtime} sec", "magenta"))
            # predictions = pd.DataFrame(model.model.predict(x), columns=["predictions"])
            # predictions.to_csv("predictions.csv")

            new_data_train_acc = pd.DataFrame(
                train_accuracies, columns=[f"{int(index)}"]
            )
            new_data_test_acc = pd.DataFrame(test_accuracies, columns=[f"{int(index)}"])
            new_data_runtimes = pd.DataFrame(runtimes, columns=[f"{int(index)}"])

            data_out_train_acc = pd.concat(
                [data_out_train_acc, new_data_train_acc], axis=1
            )
            data_out_test_acc = pd.concat(
                [data_out_test_acc, new_data_test_acc], axis=1
            )
            data_out_runtimes = pd.concat(
                [data_out_runtimes, new_data_runtimes], axis=1
            )

            data_out = pd.concat(
                [data_out_train_acc, data_out_test_acc, data_out_runtimes], axis=0
            )

            data_out.to_csv(os.path.join(experiment_dir, "std_eval_results.csv"))

    # experimental_design.to_csv(f"regularization_eval-{index}.csv")
    # design_low_high.to_csv(
    #     "fractional_factorial_results_low_high_100000_50split_regularization.csv"
    # )


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
    parser = argparse.ArgumentParser(description="Run neural_pretrain.py")
    parser.add_argument(
        "-g", "--gpu", type=int, default=0, help="Choose GPU (0 or 1).",
    )
    args = parser.parse_args()

    with tf.device(f"/device:gpu:{args.gpu}"):
        N_TEST = 10000
        print(f"GPU: {tf.test.is_built_with_cuda()}")

        experiment_dir = "data/neuralpy_optimization_expts/052220-sparcity-3"
        design_file_name = "experiments_sparcity_10.csv"
        save_file_name = "experiments_sparcity_10_results_no_first_train_2500.csv"
        save_model_path = (
            "data/neuralpy_optimization_expts/052220-sparcity-3/working_model_2500"
        )

        # make_keras_picklable()
        # standard_train_scheme(
        #     experiment_dir,
        #     design_file_name,
        #     train_sizes=[0.001, 0.01, 0.05, 0.1],
        #     n_test=N_TEST,
        # )
        pretrain_scheme(
            experiment_dir,
            design_file_name,
            save_file_name,
            save_model_path=save_model_path,
            n_pretrain_layers=0,
            n_test=N_TEST,
            train_sizes=[0.01],
            final_retrain=True,
            evaluate_distribution=True,
        )
