import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
import tensorflow.keras.backend as K
import tqdm

import copy
import math
import model
import statistics
import utils


class COBRAnet:
    def __init__(self, media_names):

        # FBA model initialization
        self.fba_model = model.Model("models/iSMUv01_CDM.xml", 20)
        self.non_essential_genes = self.fba_model.get_non_essential_genes()
        self.n_genes = len(self.non_essential_genes)
        self.obj_value = self.fba_model.model.slim_optimize()
        self.media_names = media_names
        # (
        #     self.valid_reactions,
        #     self.original_bounds,
        # ) = self.fba_model.get_reactions_of_interest()
        # self.n_reactions = len(self.valid_reactions)

        # NN layer structure
        layers = [
            tf.keras.layers.Dense(
                int(0.50 * self.n_genes), input_shape=(21,), activation="relu",
            ),
            tf.keras.layers.Dense(int(0.75 * self.n_genes), activation="relu"),
            tf.keras.layers.Dense(self.n_genes, activation="sigmoid"),
        ]
        # NN initialization
        self.nn = tf.keras.Sequential(layers)
        self.nn.build((1, 21))

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # metrics
        self.train_loss = list()
        self.train_accuracy = list()
        self.test_loss = list()
        self.test_accuracy = list()

    def get_normalized_obj(self, genes, media):
        # Modify FBA model given gene activations and media
        # Return model fitness (objective value normalized to unmodified model's objective value)
        with self.fba_model.model as m:
            # Compute reactions to delete from boolean array, the knock them out
            is_aerobic = media[-1]
            media = media[:-1]

            rxns_to_delete = np.array(self.media_names)[np.invert(media)].tolist()
            if is_aerobic:
                rxns_to_delete += model.CDM_O2_RXN_IDS

            for rxn in rxns_to_delete:
                m.reactions.get_by_id(rxn).knock_out()
                # print(f"removed({rxn})")
            print(f"# removed RXNs({len(rxns_to_delete)})")
            obj_value = m.slim_optimize()
            # Compute genes to delete from boolean array, then knock them out
            # print(genes)
            genes_to_delete = np.array(self.non_essential_genes)[
                np.invert(genes)
            ].tolist()
            for gene in genes_to_delete:
                m.genes.get_by_id(gene).knock_out()
                # print(f"removed({gene})")
            print(f"# removed GENES({len(genes_to_delete)})")

            new_obj_value = m.slim_optimize()
            print(new_obj_value, obj_value)
            normalized = new_obj_value / obj_value
            # normalized = float(max(min(1, normalized), 0))  # clamp between [0,1]
        return normalized

    def get_predictions(self, genes, media, threshold=0.25):
        # Call FBA model helper function with boolean gene vector and boolean media vector
        # Return predicted growth
        genes = genes.numpy().astype(np.float32)
        # print(genes)
        genes[genes >= threshold] = 1
        genes[genes < threshold] = 0
        genes = genes.astype(np.bool)

        media = media.numpy().astype(np.bool)
        output = np.zeros((genes.shape[0],), dtype=np.float32)
        for x in range(genes.shape[0]):
            normalized = self.get_normalized_obj(genes[x, :], media[x])
            # normalized = np.ndarray([[normalized]])
            # output[x] = float((true_growth[x, 0] - normalized))
            output[x] = float(normalized)

        output = output.astype(np.float32)
        return output

    def fba_loss(self, true_growth, genes, media, train=True):
        # Custom loss function wrapper
        #
        # Use tf.py_function to call any python function, in this case, a function
        # with an external call to an FBA model
        #
        # Give it the a boolean vector of enabled genes and the input media
        # Return loss value mean(sum squared error)
        pred = tf.py_function(
            func=self.get_predictions, inp=[genes, media], Tout=tf.float32,
        )

        loss = K.mean(K.square(true_growth - pred))

        print(f"Loss({loss})", true_growth.numpy()[0, 0], "->", pred.numpy()[0], "\n")
        if train:
            self.train_loss.append(loss.numpy())
            self.train_accuracy.append([true_growth.numpy()[0, 0], pred.numpy()[0]])
        else:
            self.test_loss.append(loss.numpy())
            self.test_accuracy.append([true_growth.numpy()[0, 0], pred.numpy()[0]])
        return loss

    def compute_gradients(self, loss):
        weights = list()
        biases = list()
        for l in self.nn.layers:
            w, b = l.get_weights()
            weights.append(K.transpose(tf.convert_to_tensor(w)))
            biases.append(K.transpose(tf.convert_to_tensor([b])))

        def sigmoid_deriv(x):
            return K.exp(-x) * K.pow(1 + K.exp(-x), -2)

        def relu_deriv(x):
            return tf.cast(K.greater(x, 1.0), tf.float32)

        def mat_mul(x, y):
            return K.dot(x, K.transpose(y))

        L = 3

        # NN input values
        inputs = self.nn.layers[0].input
        # NN activation values
        output_1 = self.nn.layers[0].output
        output_2 = self.nn.layers[1].output
        output_3 = self.nn.layers[2].output

        # Place holder for cost gradients
        shadow_price = 1
        gradC_out = 2 * loss * shadow_price
        gradC_out = np.repeat(gradC_out, weights[L - 1].shape[0], axis=0).T

        # pre-calculate delta values
        delta_3 = sigmoid_deriv(output_3) * gradC_out
        delta_2 = relu_deriv(output_2) * K.transpose(
            mat_mul(K.transpose(weights[2]), delta_3)
        )
        delta_1 = relu_deriv(output_1) * K.transpose(
            mat_mul(K.transpose(weights[1]), delta_2)
        )

        # Compute weight gradients
        gradC_W3 = delta_3 * K.transpose(output_2)
        gradC_W2 = delta_2 * K.transpose(output_1)
        gradC_W1 = delta_1 * K.transpose(inputs)

        # Compute bias gradients
        gradC_b3 = tf.reshape(delta_3, (delta_3.shape[1],))
        gradC_b2 = tf.reshape(delta_2, (delta_2.shape[1],))
        gradC_b1 = tf.reshape(delta_1, (delta_1.shape[1],))

        gradients = [gradC_W1, gradC_b1, gradC_W2, gradC_b2, gradC_W3, gradC_b3]
        return gradients

    def train_step(self, inputs, true_growth):
        genes = self.nn(inputs)
        loss = self.fba_loss(true_growth, genes, inputs, train=True)
        gradients = self.compute_gradients(loss)

        self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_variables))

        # self.train_loss.append(loss.numpy())

        # pred = self.get_predictions(genes, inputs)[0]

        # if pred[0] >= 0.25:
        #     pred = 1
        # else:
        #     pred = 0
        # print(f"Loss({loss})", true_growth.numpy()[0, 0], "->", pred)
        # self.train_accuracy.append([true_growth.numpy()[0, 0], pred])

    def test_step(self, inputs, true_growth):
        genes = self.nn(inputs)
        loss = self.fba_loss(true_growth, genes, inputs, train=False)

        # self.test_loss.append(loss.numpy())
        # pred = self.get_predictions(genes, inputs)[0]
        # if pred[0] >= 0.25:
        #     pred = 1
        # else:
        #     pred = 0
        # print(f"Loss({loss})", true_growth.numpy()[0, 0], "->", pred)
        # self.test_accuracy.append([true_growth.numpy()[0, 0], pred])

    def train(self, epochs, x_train, x_test, y_train, y_test):
        for epoch in tqdm.trange(epochs):
            # Shuffle data between epochs
            train_ds = (
                tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(25)
                .batch(1)
            )
            test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)

            # Clear the metrics at the start of the next epoch
            self.train_loss = list()
            self.train_accuracy = list()
            self.test_loss = list()
            self.test_accuracy = list()

            for inputs, true_growth, in train_ds:
                self.train_step(inputs, true_growth)

            for inputs, true_growth in test_ds:
                self.test_step(inputs, true_growth)

            l = statistics.mean(self.train_loss)
            a = r2_score(self.train_accuracy[0], self.train_accuracy[1])
            t_l = statistics.mean(self.test_loss)
            t_a = r2_score(self.test_accuracy[0], self.test_accuracy[1])

            print(
                f"Epoch {epoch + 1}, Loss: {l}, Accuracy: {a}, Test Loss: {t_l}, Test Accuracy: {t_a}"
            )


if __name__ == "__main__":
    name_mappings_csv = "files/name_mappings_aa.csv"
    mapped_data_csv = "data/iSMU-012920/initial_data/mapped_data_SMU_combined.csv"
    components = [
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
    ]
    experiments, data_growth = utils.parse_data_map(
        name_mappings_csv, mapped_data_csv, components
    )

    media_names = experiments.columns.to_list()[:-1]  # exclude "aerobic" from names
    # data_growth[data_growth >= 0.25] = 1
    # data_growth[data_growth < 0.25] = 0

    aerobic_indexes = experiments["aerobic"] == "5% CO2 @ 37 C"
    experiments.loc[aerobic_indexes, "aerobic"] = 1
    experiments.loc[~aerobic_indexes, "aerobic"] = 0
    experiments.to_csv("experiments_df.csv")
    data_growth.to_csv("growth_data_df.csv")

    print(experiments)
    print(data_growth)

    experiments = experiments.to_numpy()
    data_growth = data_growth.to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        experiments, data_growth, test_size=0.2
    )

    x_train, x_test, y_train, y_test = (
        tf.convert_to_tensor(x_train, dtype=tf.float32),
        tf.convert_to_tensor(x_test, dtype=tf.float32),
        tf.convert_to_tensor(y_train, dtype=tf.float32),
        tf.convert_to_tensor(y_test, dtype=tf.float32),
    )

    nn = COBRAnet(media_names)
    epochs = 10
    nn.train(epochs, x_train, x_test, y_train, y_test)
