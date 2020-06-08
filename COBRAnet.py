import numpy as np
import pandas as pd
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


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units
        self.reference_fluxes = none

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="zero", trainable=True
        )
        self.reference_fluxes = self.add_weight(
            shape=(self.units,), initializer="zero", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class COBRAnet:
    def __init__(self, model_filepath, media_names):

        # FBA model initialization
        self.fba_model = model.Model(model_filepath, components=media_names)
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
        mean = 0
        stddev = 0.1
        layers = [
            tf.keras.layers.Dense(
                int(0.50 * self.n_genes),
                input_shape=(21,),
                activation="relu",
                kernel_initializer=tf.random_normal_initializer(
                    mean=mean, stddev=stddev
                ),
            ),
            tf.keras.layers.Dense(
                int(0.75 * self.n_genes),
                activation="relu",
                kernel_initializer=tf.random_normal_initializer(
                    mean=mean, stddev=stddev
                ),
            ),
            tf.keras.layers.Dense(
                self.n_genes,
                activation="sigmoid",
                kernel_initializer=tf.random_normal_initializer(
                    mean=mean, stddev=stddev
                ),
            ),
        ]
        # NN initialization
        self.nn = tf.keras.Sequential(layers)
        # self.nn.build((1, 21))

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()
        self.standard_loss = tf.keras.losses.BinaryCrossentropy()
        self.growth_cutoff = 0.25
        # metrics
        # self.train_loss = []
        # self.train_accuracy = []
        # self.test_loss = []
        # self.test_accuracy = []
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.Accuracy(name="train_accuracy")
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.Accuracy(name="test_accuracy")

    def get_normalized_obj(self, genes, media):
        """Modify FBA model given gene activations and media
        Return model fitness (objective value normalized to unmodified model's objective value)
        """
        # Compute reactions to delete from boolean array
        is_aerobic = media[-1]
        media = media[:-1]
        rxns_to_delete = np.array(self.media_names)[np.invert(media)].tolist()
        if not is_aerobic:
            rxns_to_delete += model.CDM_O2_RXN_IDS

        with self.fba_model.model as m:
            # Knock out the reactions for the media configuration
            for rxn in rxns_to_delete:
                m.reactions.get_by_id(rxn).knock_out()
            # print(f"# removed RXNs({len(rxns_to_delete)})")
            reference_obj_value = m.slim_optimize()

            with self.fba_model.model as m:
                # Compute genes to delete from boolean array
                genes_to_delete = np.array(self.non_essential_genes)[
                    np.invert(genes)
                ].tolist()
                # Knock out the genes
                for gene in genes_to_delete:
                    m.genes.get_by_id(gene).knock_out()
                # print(f"# removed GENES({len(genes_to_delete)})")

                obj_value = m.slim_optimize()
                normalized = obj_value / reference_obj_value
                if np.isnan(normalized):
                    normalized = round(normalized, 5)

                # print(new_obj_value, obj_value)

            # Use finite difference method to estimate dLoss/dGenes, also knows as gradient of cost
            # For each gene, if it's on, turn it off, and the inverse and to measure dLoss/dG
            # where the flipped gene, G, is notated G'
            # approximation for the derivative, where dLoss/dG = (Loss(G') - Loss(G))/ deltaG
            # where deltaG = 1, since the gene vector is only changing by 1 gene at a time.
            delta_growth = np.zeros((genes.shape[0],), dtype=np.float32)
            for idx in range(len(self.non_essential_genes)):
                binary_genes = copy.copy(genes)
                binary_genes[idx] = not binary_genes[idx]
                with self.fba_model.model as m:
                    # Compute genes to delete from boolean array
                    genes_to_delete = np.array(self.non_essential_genes)[
                        np.invert(binary_genes)
                    ].tolist()

                    # Knock out the genes
                    for gene in genes_to_delete:
                        m.genes.get_by_id(gene).knock_out()

                    new_obj_value = m.slim_optimize()
                    new_normalized = new_obj_value / reference_obj_value
                    if not np.isnan(new_normalized):
                        new_normalized = round(new_normalized, 5)
                    # print("delta_g", normalized - new_normalized)
                    # delta_growth[idx] = normalized - new_normalized
                    delta_growth[idx] = obj_value - new_obj_value
            # grad_C = np.ones((genes.shape[0],), dtype=np.float32)

        delta_growth = np.nan_to_num(delta_growth)
        normalized = np.nan_to_num(normalized)
        print("Delta growth\n", delta_growth)
        print("normalized\n", normalized)
        return normalized, delta_growth

    def get_predictions(self, genes, media):
        """Call FBA model helper function with boolean gene vector and boolean media vector
        Return predicted growth"""
        genes = genes.numpy().astype(np.float32)
        genes[genes >= self.growth_cutoff] = 1
        genes[genes < self.growth_cutoff] = 0

        genes = genes.astype(np.bool)
        media = media.numpy().astype(np.bool)

        output = np.zeros((genes.shape[0],), dtype=np.float32)
        delta_growths = np.zeros((genes.shape[0], genes.shape[1]), dtype=np.float32)
        for x in range(genes.shape[0]):
            normalized, delta_growth = self.get_normalized_obj(genes[x, :], media[x])
            output[x] = float(normalized)
            delta_growths[x, :] = delta_growth

        output = output.astype(np.float32)
        return output, delta_growths

    def fba_loss(
        self, true_growth, genes, media, train=True, normalize_sum_square=True
    ):
        """Custom loss function wrapper
        
        Use tf.py_function to call any python function, in this case, a function
        with an external call to an FBA model
        
        Give it the a boolean vector of enabled genes and the input media
        Return loss value mean(sum squared error)"""

        pred, delta_growths = tf.py_function(
            func=self.get_predictions,
            inp=[genes, media],
            Tout=[tf.float32, tf.float32],
        )
        # Calculate loss
        total = 0
        # if normalize_sum_square:
        #     total = K.sum(K.square(true_growth - pred))
        # if total.numpy() > 0:
        #     loss = (K.square(true_growth - pred)) / (total * 2)
        # else:
        # loss = (K.square(true_growth - pred)) / 2
        with tf.GradientTape() as tape:
            loss_bc = self.standard_loss(genes, genes)
            print(loss_bc)
        gradients_bc = tape.gradient(loss_bc, self.nn.trainable_variables)
        print("REAL GRADIENTS")
        print(gradients_bc)

        loss = K.mean((K.square(true_growth - pred))) / 2
        cost_gradients = tf.math.divide_no_nan(loss_bc, delta_growths)
        # print("Loss\n", loss)
        # print("Cost gradients\n", cost_gradients)
        # cost_gradients = loss / delta_growths
        # cost_gradients[tf.where(np.isnan(cost_gradients))] = 0
        # print(loss, cost_gradients)
        # mean_loss = K.mean(loss)

        # Calculate performance
        # print(f"Loss({loss})", true_growth.numpy()[0, 0], "->", pred.numpy()[0], "\n")
        if train:
            self.train_loss.update_state(loss)
            self.train_accuracy.update_state(true_growth, pred)
        else:
            self.test_loss.update_state(loss)
            pred = tf.cast(K.greater_equal(pred, self.growth_cutoff), tf.float32)
            true_growth = tf.cast(
                K.greater_equal(true_growth, self.growth_cutoff), tf.float32
            )
            self.test_accuracy.update_state(true_growth, pred)
        return loss, cost_gradients, gradients_bc

    # def compute_gradients(self, loss, cost_gradients):
    #     # gradC_W1, gradC_b1, gradC_W2, gradC_b2, gradC_W3, gradC_b3 = tf.py_function(
    #     #     func=self.py_compute_gradients,
    #     #     inp=[loss, cost_gradients],
    #     #     Tout=[
    #     #         tf.float32,
    #     #         tf.float32,
    #     #         tf.float32,
    #     #         tf.float32,
    #     #         tf.float32,
    #     #         tf.float32,
    #     #     ],
    #     # )

    #     out = tf.py_function(
    #         func=self.py_compute_gradients,
    #         inp=[loss, cost_gradients],
    #         Tout=[tf.float32],
    #     )
    #     # gradients = [gradC_W1, gradC_b1, gradC_W2, gradC_b2, gradC_W3, gradC_b3]
    #     # return gradC_W1

    def sigmoid_deriv(self, x):
        return K.exp(-x) * K.pow(1 + K.exp(-x), -2)

    def relu_deriv(self, x):
        return tf.cast(K.greater(x, 1.0), tf.float32)

    def mat_mul(self, x, y):
        return K.dot(x, K.transpose(y))

    def compute_gradients(self, loss, cost_gradients):
        weights = []
        biases = []
        for l in self.nn.layers:
            w, b = l.get_weights()
            weights.append(K.transpose(tf.convert_to_tensor(w)))
            biases.append(K.transpose(tf.convert_to_tensor([b])))

        # def sigmoid_deriv(x):
        #     return K.exp(-x) * K.pow(1 + K.exp(-x), -2)

        # def relu_deriv(x):
        #     return tf.cast(K.greater(x, 1.0), tf.float32)

        # def mat_mul(x, y):
        #     return K.dot(x, K.transpose(y))

        # NN input values
        inputs = self.nn.layers[0].input
        # NN activation values
        output_1 = self.nn.layers[0].output
        output_2 = self.nn.layers[1].output
        output_3 = self.nn.layers[2].output

        # Place holder for cost gradients
        shadow_price = 1
        gradC_out = loss * cost_gradients
        # print(cost_gradients)
        # gradC_out = np.repeat(gradC_out, weights[2].shape[0], axis=0).T

        # pre-calculate delta values
        delta_3 = self.sigmoid_deriv(output_3) * gradC_out
        delta_2 = self.relu_deriv(output_2) * K.transpose(
            self.mat_mul(K.transpose(weights[2]), delta_3)
        )
        delta_1 = self.relu_deriv(output_1) * K.transpose(
            self.mat_mul(K.transpose(weights[1]), delta_2)
        )
        delta_1 = K.print_tensor(delta_1, message="delta_1")

        # Compute weight gradients
        gradC_W3 = delta_3 * K.transpose(output_2)
        gradC_W2 = delta_2 * K.transpose(output_1)
        gradC_W1 = delta_1 * K.transpose(inputs)

        # Compute bias gradients
        gradC_b3 = tf.reshape(delta_3, (delta_3.shape[1],))
        gradC_b2 = tf.reshape(delta_2, (delta_2.shape[1],))
        gradC_b1 = tf.reshape(delta_1, (delta_1.shape[1],))
        gradC_W1 = K.print_tensor(gradC_W1)
        gradC_b1 = K.print_tensor(gradC_b1)
        gradC_W2 = K.print_tensor(gradC_W2)
        gradC_b2 = K.print_tensor(gradC_b2)
        gradC_W3 = K.print_tensor(gradC_W3)
        gradC_b3 = K.print_tensor(gradC_b3)
        # gradC_W1 = tf.convert_to_tensor(gradC_W1)
        # gradC_b1 = tf.convert_to_tensor(gradC_b1)
        # gradC_W2 = tf.convert_to_tensor(gradC_W2)
        # gradC_b2 = tf.convert_to_tensor(gradC_b2)
        # gradC_W3 = tf.convert_to_tensor(gradC_W3)
        # gradC_b3 = tf.convert_to_tensor(gradC_b3)
        gradients = [gradC_W1, gradC_b1, gradC_W2, gradC_b2, gradC_W3, gradC_b3]
        # print("MY GRADIENTS")
        print("MY GRADIENTS", gradients)

        # return gradC_W1
        # return gradC_W1, gradC_b1, gradC_W2, gradC_b2, gradC_W3, gradC_b3
        # print(cost_gradients.numpy())
        # return gradients
        return gradients

    def train_step(self, inputs, true_growth):
        genes = self.nn(inputs)
        loss, cost_gradients, gradients_bc = self.fba_loss(
            true_growth, genes, inputs, train=True
        )
        gradients = self.compute_gradients(loss, cost_gradients)

        self.optimizer.apply_gradients(zip(gradients_bc, self.nn.trainable_variables))
        # self.optimizer.apply_gradients(zip(gradients, self.nn.trainable_variables))

    def test_step(self, inputs, true_growth):
        genes = self.nn(inputs)
        loss = self.fba_loss(true_growth, genes, inputs, train=False)

    def train(self, epochs, x_train, x_test, y_train, y_test):
        for epoch in tqdm.trange(epochs):
            # Shuffle data between epochs
            train_ds = (
                tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(x_train.shape[0])
                .batch(1)
            )
            test_ds = (
                tf.data.Dataset.from_tensor_slices((x_test, y_test))
                .shuffle(x_test.shape[0] // 10)  # 10% buffer size for shuffle
                .batch(x_test.shape[0])  # max batch size
            )

            # Clear the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()
            train_ds = list(train_ds.as_numpy_iterator())
            for idx in tqdm.trange(x_train.shape[0]):
                inputs, true_growth = train_ds[idx]
                self.train_step(inputs, true_growth)
                # weights = []
                # biases = []
                for idx, l in enumerate(self.nn.layers):
                    w, b = l.get_weights()
                    print(f"Layer {idx}", w, b)
                    # weights.append(K.transpose(tf.convert_to_tensor(w)))
                    # biases.append(K.transpose(tf.convert_to_tensor([b])))

            for inputs, true_growth in test_ds:
                self.test_step(inputs, true_growth)

            # l = statistics.mean(self.train_loss)
            # a = r2_score(self.train_accuracy[0], self.train_accuracy[1])
            # t_l = statistics.mean(self.test_loss)
            # t_a = r2_score(self.test_accuracy[0], self.test_accuracy[1])

            print(
                f"Epoch {epoch + 1}, Loss: {self.train_loss.result()}, Accuracy: {self.train_accuracy.result()}, Test Loss: {self.test_loss.result()}, Test Accuracy: {self.test_accuracy.result()}"
            )


if __name__ == "__main__":
    # name_mappings_csv = "files/name_mappings_aa.csv"
    # mapped_data_csv = "data/iSMU-012920/initial_data/mapped_data_SMU_combined.csv"
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
    # experiments, data_growth = utils.parse_data_map(
    #     name_mappings_csv, mapped_data_csv, components
    # )

    # data_growth[data_growth >= 0.25] = 1
    # data_growth[data_growth < 0.25] = 0
    # aerobic_indexes = experiments["aerobic"] == "5% CO2 @ 37 C"
    # experiments.loc[aerobic_indexes, "aerobic"] = 1
    # experiments.loc[~aerobic_indexes, "aerobic"] = 0
    # experiments.to_csv("experiments_df.csv")
    # data_growth.to_csv("growth_data_df.csv")
    # media_names = experiments.columns.to_list()[:-1]  # exclude "aerobic" from names

    # print(experiments)
    # print(data_growth)

    # Use second GPU
    physical_devices = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(physical_devices[1:], device_type="GPU")
    # print("\n\n", tf.config.list_logical_devices("GPU"))

    experiments = pd.read_csv("experiments_df.csv", index_col=0)
    data_growth = pd.read_csv("growth_data_df.csv", index_col=0)
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

    nn = COBRAnet("models/iSMUv01_CDM.xml", components)
    epochs = 10
    nn.train(epochs, x_train, x_test, y_train, y_test)
