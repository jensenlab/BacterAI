import argparse
import ast
import copy
import csv
import itertools
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sp
import sklearn.metrics
from matplotlib import colors
from scipy.spatial import distance

import model
import neural
import utils


parser = argparse.ArgumentParser(description="Run agent.py")
parser.add_argument(
    "-c",
    "--num_components",
    required=True,
    type=int,
    help="Number of components in media.",
)
parser.add_argument(
    "-n",
    "--generate_new_data",
    action="store_true",
    default=False,
    help="Make new data rather than reading from file.",
)
parser.add_argument(
    "-nm", "--new_model", action="store_true", default=False, help="Generate new model."
)
parser.add_argument(
    "-s",
    "--save_models",
    action="store_true",
    default=False,
    help="Save NN model every cycle.",
)
args = parser.parse_args()


class Agent:
    def __init__(
        self,
        model,
        predictor,
        data,
        data_labels,
        minimum_cardinality,
        cycle=0,
        current_solution=None,
        data_history=None,
        data_labels_history=None,
    ):
        self.model = model
        self.predictor = predictor
        self.data = pd.DataFrame(data)
        self.data_labels = pd.DataFrame(data_labels)
        self.minimum_cardinality = (
            minimum_cardinality if minimum_cardinality else self.model.num_components
        )

        self.cycle = 0
        self.current_solution = None
        self.data_history = None
        self.data_labels_history = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["predictor"] = type(state["predictor"])
        return state

    def __setstate__(self, state):
        state["predictor"] = state["predictor"]()
        self.__dict__.update(state)
        return state

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            agent = pickle.load(f)
        return agent

    def initialize_learning(self, batch_size, save_location):
        starting_data, starting_labels = self.get_starting_data()

        self.data_history = starting_data.copy()
        self.data_labels_history = starting_labels.copy()

        print("DATA HISTORY:\n", self.data_history)

        self._perform_cycle(
            batch_train_data=starting_data,
            batch_train_data_labels=starting_labels,
            batch_size=batch_size,
            save_location=save_location,
        )

    def continue_learning(
        self, name_mappings_path, batch_data_path, batch_size, save_location
    ):
        batch, batch_labels = utils.parse_data_map(
            name_mappings_path, batch_data_path, self.model.media_ids
        )
        batch, batch_labels = utils.match_original_data(self.data, batch, batch_labels)
        self._perform_cycle(
            batch_train_data=batch,
            batch_train_data_labels=batch_labels,
            batch_size=batch_size,
            save_location=save_location,
        )

    def new_batch(self, K, threshold=0.5, use_neural_net=True):
        """Predicts a new batch of size `K` from the growth media data set 
        `self.data`. Filters batch based on a growth threshold of a neural 
        net's predictions in addition to the cardinality of the growth medias.
        The new batches represent the Agent's predictions for minimal growth
        medias.
    
        Inputs
        ------
        K: int
            Batch size to return.
        threshold: float, default=0.5
            Threshold to determine if prediction from `self.predictor` is 'grow'
            or 'no grow'.
        use_neural_net: Boolean, default=True
            If 'True' use neural net to predict growth, otherwise use Naive 
            Bayes predictor.
            
        Returns
        -------
        min_K_indexes: np.ndarray()
            Indexes of `self.data` that correspond to the new batch.
        batch_min_cardinality: int
            The batch minimum cardinality.
        
        """

        def _get_random(data, n):
            """Gets random media combinations from `data`.
        
            Inputs
            ------
            data: pd.DataFrame()
                The data from which to sample from.
            n: int
                The number of random data to find.
                
            Returns
            -------
            data: pd.DataFrame
                The original data minus the samples chosen.
            data_random: pd.DataFrame
                The samples chosen.
            
            """
            random_indexes = random.sample(
                data.index.to_list(), min(n, data.index.shape[0])
            )
            data_random = data.loc[random_indexes, :]
            data = data.drop(random_indexes)
            print("Random Data\n", data_random)
            return data, data_random

        def _get_min_K(
            n_needed, explore_cardinality=None, add_random=False, used_indexes=None
        ):
            """Filters `self.data` to return `n_needed` samples.
        
            Inputs
            ------
            explore_cardinality: int, default=None
                Exploration variable.
            add_random: float, default=False
                The proportion of batch which will be random samples.
                
            Returns
            -------
            min_K_indexes: np.ndarray()
                Indexes of `self.data` that correspond to the new batch.
            batch_min_cardinality: int
                The batch minimum cardinality.
            
            """
            # Make copy of data to manipulate
            data = self.data.copy()

            # Drop used indexes
            if used_indexes is not None:
                data = data.drop(index=used_indexes)

            # Calculate cardinality
            data["cardinality"] = data.sum(axis=1)
            print("Added cardinality\n", data)

            if use_neural_net:
                print("USING NEURAL NET")
                # Get probability of grow using NN
                result = self.predictor.predict_probability(
                    data.drop(columns=["cardinality"]).to_numpy()
                )
            else:
                print("USING NAIVE BAYES")
                # Get probability of grow using Naive Bayes
                result = self.predictor.predict_bayes(
                    data.drop(columns=["cardinality"]).to_numpy()
                )

            # Append neural net/naive Bayes predictions
            data["prediction"] = result

            # Add random samples to for exploration
            if add_random > 0:
                data, data_random = _get_random(data, int(n_needed * add_random))

            # Take only data that have a prediction > threshold
            data = data[data.prediction >= threshold]

            print("results\n", result)
            print(
                f"Num <= 0.50: {result[result <= 0.5].shape[0]}, Num > 0.50: {result[result >= 0.5].shape[0]}"
            )
            print("Threshold filter\n", data)

            if explore_cardinality is not None:
                # Exploration of search space, helpful for getting out of local
                # minima
                target_cardinality = self.minimum_cardinality + explore_cardinality
                data = data[data.cardinality == target_cardinality]
                print("Target card filter\n", data)

                data_tested = self.data_history.copy()
                data_tested["cardinality"] = data_tested.sum(axis=1)
                data_tested = data_tested[data_tested.cardinality == target_cardinality]
                print("Target card filter (history)\n", data_tested)

                data_tested = data_tested.drop(columns=["cardinality"]).to_numpy()

                def _avg_hamming_dist(x):
                    """Compute average Hamming distance of input `x` against
                    all of the samples in `data_tested`. Used as an exploration
                    heuristic to choose new samples which are more unlike 
                    previously tested data.
                    """

                    if data_tested.shape[0] == 0:
                        return 0
                    distances = [np.count_nonzero(t != x) for t in data_tested]
                    hamming_total = sum(distances)
                    hamming_avg = hamming_total / len(distances)
                    return hamming_avg

                if data.shape[0] > 0:
                    hamming_distances = np.apply_along_axis(
                        _avg_hamming_dist,
                        axis=1,
                        arr=data.to_numpy()[:, : self.model.num_components],
                    )
                    data["hamming dist"] = hamming_distances
                    data = data.sort_values(
                        by=["hamming dist", "cardinality", "prediction"],
                        ascending=[False, True, False],
                    )

            else:
                # Default condition:
                # Take only data that have cardinality < current minimum
                data = data[data.cardinality < self.minimum_cardinality]
                print("Card filter\n", data)

                # Sort by cardinality then prediction confidence
                data = data.sort_values(
                    by=["cardinality", "prediction"], ascending=[True, False]
                )

            # Only add on random samples if add_random is specified
            if isinstance(add_random, float) and add_random < 1.0 and add_random > 0.0:
                data = pd.concat([data_random, data])
                print("CONCAT RANDOM", data.shape, data)
            elif add_random >= 1.0:
                data = data_random

            data = data.drop(data.index[n_needed:])

            current_min_cardinality = data.cardinality.min()
            print("BATCH MIN:", current_min_cardinality)

            print("DATA DataFrame:\n", data)

            # Take only the first n_needed indexes
            min_K_indexes = data.index.to_numpy()

            return min_K_indexes, current_min_cardinality

        # Get new batch
        min_K_indexes, batch_min_cardinality = _get_min_K(
            n_needed=K, add_random=0.10
        )  # , explore_cardinality=0)

        # Start exploration if there are not enough samples in the batch
        # `explore_var` will be incremented until explore_var + current
        # minimum cardinality > the number of media components. When this
        # occurs, it will fill the remaining spaces in the batch with random
        # data.
        n_needed = K - min_K_indexes.shape[0]
        explore_var = 0
        add_random = 0.0
        stop = False
        while n_needed > 0 and not stop:
            print(f"Exploring ({explore_var})...", f"Number needed: {n_needed}")
            min_K_indexes_new, batch_min_cardinality_new = _get_min_K(
                n_needed=n_needed,
                explore_cardinality=explore_var,
                add_random=add_random,
                used_indexes=min_K_indexes,
            )

            min_K_indexes = np.concatenate(
                [min_K_indexes, min_K_indexes_new], axis=None
            )
            if batch_min_cardinality_new <= batch_min_cardinality or math.isnan(
                batch_min_cardinality
            ):
                batch_min_cardinality = batch_min_cardinality_new

            n_needed = K - min_K_indexes.shape[0]
            explore_var += 1
            if self.minimum_cardinality + explore_var >= self.model.num_components:
                print("Cannot explore any further...choosing random")
                add_random = 1.0
                stop = True

        print("indexes:", min_K_indexes)
        return min_K_indexes, batch_min_cardinality

    def get_metrics(self, data, data_labels, use_bayes=False):
        if data.size == 0 or data_labels.size == 0:
            print("Cannot calculate metrics... empty data.")
            return None, None, None
        if use_bayes:
            predictions = self.predictor.predict_bayes(data)
            predictions[predictions >= 0.5] = 1
            predictions[predictions < 0.5] = 0
        else:
            predictions = self.predictor.predict_class(data)

        precision = sklearn.metrics.precision_score(data_labels, predictions)
        accuracy = sklearn.metrics.accuracy_score(data_labels, predictions)
        recall = sklearn.metrics.recall_score(data_labels, predictions)

        print(f"Precision: {precision}, Accuracy: {accuracy}, Recall: {recall}")
        return precision, accuracy, recall

    def get_starting_data(self):
        # Initialize with L1O and L2O data
        L1O_exp = utils.get_LXO(self.model.num_components, 1)
        L2O_exp = utils.get_LXO(self.model.num_components, 2)
        L3O_exp = utils.get_LXO(self.model.num_components, 3)

        n = 300
        subset_indexes = random.sample(
            range(L3O_exp.shape[0]), min(n, L3O_exp.shape[0])
        )
        additional_exp = L3O_exp[subset_indexes]
        # Semi random inputs based on L3Os
        # for x in additional_exp:
        #     n = sp.poisson.rvs(1)
        #     candidates_indexes = np.where(x == 1)[0]
        #     set_to_zero = np.random.choice(candidates_indexes, size=n)
        #     x[set_to_zero] = 0

        # random_inputs = np.random.choice([0, 1], size=(1000, self.model.num_components))

        data = np.concatenate([L1O_exp, L2O_exp, additional_exp])
        data_labels = np.array(
            self.model.evaluate(data, use_bool=False, use_multiprocessing=False)
        )
        data = pd.DataFrame(data)
        data_labels = pd.DataFrame(data_labels)

        data, data_labels = utils.match_original_data(self.data, data, data_labels)

        return data, data_labels

    def _perform_cycle(
        self, batch_train_data, batch_train_data_labels, batch_size, save_location,
    ):
        self.cycle += 1
        answer = self.model.minimal_components

        print(f"Batch train data: \n{batch_train_data}")
        print(f"Batch train data labels: \n{batch_train_data_labels}")

        # Add batch to data history
        self.data_history = pd.concat([self.data_history, batch_train_data])
        print("DATA HISTORY", self.data_history.shape, self.data_history)
        batch_train_data_labels.columns = self.data_labels_history.columns
        self.data_labels_history = pd.concat(
            [self.data_labels_history, batch_train_data_labels]
        )

        # Filter new batch results
        # Only include cardinalities < current min that are known to grow
        cardinality = batch_train_data.sum(axis=1)
        batch = pd.concat([cardinality, batch_train_data_labels], axis=1)
        batch.columns = ["cardinality", "labels"]
        batch = batch[
            (batch["cardinality"] < self.minimum_cardinality) & (batch["labels"] == 1)
        ]
        print("BATCH\n", batch)

        # Evaluate new batch to check for growth & new min solutions
        if batch.size > 0:
            self.minimum_cardinality = batch.cardinality.min()
            minimum_index = batch[
                batch["cardinality"] == self.minimum_cardinality
            ].index[0]
            self.current_solution = self.data.loc[minimum_index, :]
            print(
                f"INPUT FOR NEW MIN ({self.minimum_cardinality}): {self.current_solution}"
            )

        # Remove used experiments from data set
        self.data.drop(batch_train_data.index, inplace=True)
        self.data_labels.drop(batch_train_data.index, inplace=True)

        # Reset predictor to train a new model every time
        self.predictor = type(self.predictor)()

        self.predictor.train(
            self.data_history.to_numpy(), self.data_labels_history.to_numpy(), epochs=5,
        )
        self.predictor.train_bayes(
            self.data_history.to_numpy(), self.data_labels_history.to_numpy()
        )

        # Export models every cycle

        model_name = f"NN_{self.model.num_components}_C{self.cycle}.h5"
        model_output_folder = os.path.join(save_location, "neural_nets")
        model_output_path = os.path.join(model_output_folder, model_name)
        if not os.path.exists(model_output_folder):
            os.makedirs(model_output_folder)
        self.predictor.model.save(model_output_path)

        # Get next batch
        new_batch_indexes, batch_min_cardinality = self.new_batch(
            K=batch_size, threshold=0.3
        )

        batch_name = f"batch_C{self.cycle}.csv"
        batch_output_folder = os.path.join(save_location, "batches")
        batch_output_path = os.path.join(batch_output_folder, batch_name)
        if not os.path.exists(batch_output_folder):
            os.makedirs(batch_output_folder)

        batch_output_df = self.data.loc[new_batch_indexes, :]
        batch_output_df.to_csv(batch_output_path)

        agent_name = f"agent_state_C{self.cycle}.pkl"
        agent_output_folder = os.path.join(save_location, "agents")
        agent_output_path = os.path.join(agent_output_folder, agent_name)
        if not os.path.exists(agent_output_folder):
            os.makedirs(agent_output_folder)
        self.save(agent_output_path)

        # Get a subset of data to speed up stats
        test_indexes = np.random.choice(
            self.data.index, size=int(0.1 * self.data.shape[0]), replace=False
        )
        test_data_x = self.data.loc[test_indexes, :]
        test_data_y = self.data_labels.loc[test_indexes]

        # Get NN prediction stats for data set
        precision, accuracy, recall = self.get_metrics(
            test_data_x.to_numpy(), test_data_y.to_numpy()
        )

        # Output findings and real solution
        print(f"\n\n################ CYCLE {self.cycle} COMPLETE! #################")
        found_solution = set()
        for idx, c in enumerate(self.current_solution):
            if c == 1:
                found_solution.add(self.model.media_ids[idx])

        print(
            f"ACCURACY: {accuracy}, BATCH MIN CARDINALITY: {batch_min_cardinality}, OVERALL MIN CARDINALITY: {self.minimum_cardinality}"
        )
        print("CORRECT MINIMUM:", len(answer), "\tCORRECT SET:", answer)
        print(
            "FOUND MINIMUM:",
            int(self.minimum_cardinality),
            "\tFOUND SET:",
            found_solution,
        )

        utils.batch_to_deep_phenotyping_protocol(
            f"BacterAI-SMU-C{self.cycle}",
            batch_output_df,
            self.model.media_ids,
            "files/name_mappings_aa.csv",
        )


if __name__ == "__main__":
    # Silence some Tensorflow outputs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    model_folder = "iSMUv01_CDM_2020-01-17T153130538/12"

    if args.new_model:
        m = model.Model("models/iSMUv01_CDM.xml", 16)
        model_path = m.make_minimal_media_models(max_n=1)
        data_path = os.path.join(
            "/".join(model_path.split("/")[:-1]), f"data_{args.num_components}.csv"
        )
    else:
        folder_path = os.path.join("models", model_folder)
        model_filename = "iSMUv01_CDM_12.xml"
        model_path = os.path.join(folder_path, model_filename)
        data_path = os.path.join(folder_path, f"data_{args.num_components}.csv")

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
    m = model.Model(
        model_path,
        num_components=args.num_components,
        new_data=args.generate_new_data,
        components=components[: args.num_components],
    )

    if args.generate_new_data:
        m.generate_data_csv(use_multiprocessing=True)

    print("Model:", model_path)
    print(f"MIN ({len(m.minimal_components)}): {m.minimal_components}")

    data = np.genfromtxt(data_path, delimiter=",")
    predictor = neural.PredictNet()
    agent = Agent(m, predictor, data[1:, :-1], data[1:, -1], args.num_components)

    # if args.save_models:
    #     agent.find_minimal_media_2(batch_size=300, save_models=model_folder)
    # else:
    #     agent.find_minimal_media_2(batch_size=300)

    # agent.initialize_learning(batch_size=378, save_location="intialize_learning_test/")
    agent_cont = Agent.load("data/testing_cycles/agents/agent_state_C1.pkl")
    agent_cont.continue_learning(
        name_mappings_path="files/name_mappings_aa.csv",
        batch_data_path="data/testing_cycles/mapped_data_dp_test.csv",
        batch_size=378,
        save_location="data/testing_cycles/",
    )

    # agent_d = agent.__dict__
    # for k, v in agent_d.items():
    #     print("\nKEY:\n", k)
    #     print("\nVAL:\n", v)

    # # del agent_d["model"]
    # del agent_d["predictor"]
    # with open("test_agent.pkl", "wb") as f:
    #     pickle.dump(agent_d, f)

    # with open("intialize_learning_test/agents/agent_state_C1.pkl", "rb") as f:
    #     agent = pickle.load(f)
    #     agent_d = agent.__dict__
    #     for k, v in agent_d.items():
    #         print("\nKEY:\n", k)
    #         print("\nVAL:\n", v)
    #         # if k=="predictor":
    #         #     p = v()
    #         #     print(p)
