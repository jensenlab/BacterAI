import argparse
import ast
import collections
import copy
import csv
import itertools
import math
import os
import pickle
import random

import numpy as np
import pandas as pd
from scipy.spatial import distance
import scipy.stats as sp
import sklearn.metrics
from matplotlib import colors
import tensorflow as tf

import model
import neural_multi_gpu as neural
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

parser.add_argument(
    "-p",
    "--production",
    action="store_true",
    default=False,
    help="Output to production server.",
)

parser.add_argument(
    "-o",
    "--output_batch",
    action="store_true",
    default=False,
    help="Generate DeepPhenotyping experiment.",
)

parser.add_argument(
    "-g", "--gpu", type=int, default=0, help="Choose GPU (0 or 1).",
)

args = parser.parse_args()


class Agent:
    def __init__(
        self,
        model,
        predictor,
        data_path,
        minimum_cardinality,
        name_mappings_path,
        cycle=0,
        current_solution=None,
        data_history=None,
        data_labels_history=None,
        growth_threshold=0.25,
    ):
        self.model = model
        self.predictor = predictor
        self.growth_threshold = growth_threshold

        data = pd.read_csv(data_path)
        self.data_labels = data.iloc[:, [-1]]
        self.data = data.iloc[:, :-1]

        self.minimum_cardinality = (
            minimum_cardinality if minimum_cardinality else self.model.num_components
        )

        self.name_mappings_path = name_mappings_path
        self.cycle = 0
        self.current_solution = None
        self.data_history = pd.DataFrame(columns=self.data.columns, dtype=np.float32)
        self.data_labels_history = pd.DataFrame(
            columns=self.data_labels.columns, dtype=np.float32
        )

        self.model_answer = self.model.minimal_components
        # answer = data.copy()
        data["cardinality"] = self.data.sum(axis=1)
        data = data[data["grow"] >= self.growth_threshold]
        data = data[data["cardinality"] == data["cardinality"].min()]
        data = data.iloc[0, :-2]
        self.answer = set(data.index[(data == 1)])

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
    def load(cls, agent_path, predictor_path):
        with open(agent_path, "rb") as f:
            agent = pickle.load(f)
        agent.predictor.model = tf.keras.models.load_model(predictor_path)
        return agent

    def learn(
        self,
        batch_size,
        save_location,
        data_path,
        exp_name,
        output_batch=False,
        parse_required=True,
        make_binary=False,
    ):

        batch_data, batch_labels = self.get_matching_data(
            batch_data_path=data_path, parse_required=parse_required
        )
        if make_binary:
            batch_labels = batch_labels >= self.growth_threshold
            batch_labels = batch_labels.astype(np.int)

        new_batch_path = self._perform_cycle(
            batch_train_data=batch_data,
            batch_train_data_labels=batch_labels,
            batch_size=batch_size,
            save_location=save_location,
            development=not args.production,
            exp_name=exp_name,
            output_batch=output_batch,
        )
        return new_batch_path

    def simulate(self, n_sample_exploration_cycles=None, **kwargs):
        new_batch_path = True
        while new_batch_path:
            # if n_sample_exploration_cycles:
            #     if self.cycle < n_sample_exploration_cycles:

            print(kwargs)
            new_batch_path = self.learn(**kwargs)

            print("\n\nRetrieving simulated batch results for:")
            print("\t", new_batch_path)

            batch_data, _ = self.get_matching_data(
                batch_data_path=new_batch_path,
                parse_required=False,
                included_labels=False,
                has_index=True,
            )

            split_path = os.path.split(new_batch_path)
            batch_file_name = os.path.splitext(split_path[1])

            simulation_save_location = os.path.join(
                split_path[0], batch_file_name[0] + "_data" + batch_file_name[1]
            )
            batch_indexes = batch_data.index
            batch_labels = self.data_labels.loc[batch_indexes]
            batch_data["grow"] = batch_labels
            batch_data.to_csv(simulation_save_location, index=False)

            kwargs["data_path"] = simulation_save_location
            kwargs["parse_required"] = False

    def get_matching_data(
        self,
        batch_data_path,
        parse_required=True,
        ignore_aerobic=True,
        included_labels=True,
        has_index=False,
    ):
        if parse_required:
            batch, batch_labels = utils.parse_data_map(
                self.name_mappings_path,
                batch_data_path,
                self.model.media_ids,
                self.growth_threshold,
            )
        else:
            if has_index:
                index_col = 0
            else:
                index_col = None
            data = pd.read_csv(batch_data_path, index_col=index_col)
            print(data)
            if included_labels:
                batch = data.iloc[:, :-1]
                batch_labels = data.iloc[:, [-1]]
            else:
                batch = data
                batch_labels = None
            print(batch)
            batch.columns = self.model.media_ids

        if ignore_aerobic and ("aerobic" in list(batch.columns)):
            batch = batch.drop(columns=["aerobic"])
        batch, batch_labels = utils.match_original_data(self.data, batch, batch_labels)

        return batch, batch_labels

    def assemble_initial_data(
        self, data_path, p, supplemental_data_path=None, even_grow_split=True
    ):
        data = pd.read_csv(data_path, index_col=None)
        data["cardinality"] = data.drop(columns=["grow"]).sum(axis=1)
        cardinalities = pd.unique(data["cardinality"])

        indexes_of_interest = []
        for card in cardinalities:
            grow_indexes = data.loc[
                (data["cardinality"] == card) & (data["grow"] >= self.growth_threshold),
                :,
            ].index.to_list()
            no_grow_indexes = data.loc[
                (data["cardinality"] == card) & (data["grow"] >= self.growth_threshold),
                :,
            ].index.to_list()

            total_length = len(no_grow_indexes) + len(grow_indexes)
            num_to_take = max(int(total_length * p), min(10, total_length))

            if even_grow_split:
                num_to_take //= 2
                grow_indexes = np.random.choice(
                    grow_indexes, size=num_to_take, replace=False,
                ).tolist()
                no_grow_indexes = np.random.choice(
                    no_grow_indexes, size=num_to_take, replace=False,
                ).tolist()
                indexes = grow_indexes + no_grow_indexes
            else:
                indexes = grow_indexes + no_grow_indexes
                indexes = np.random.choice(
                    indexes, size=num_to_take, replace=False,
                ).tolist()
            indexes_of_interest += indexes

        print(len(indexes_of_interest), "chosen")
        data = data.drop(columns=["cardinality"]).loc[indexes_of_interest, :]

        if supplemental_data_path:
            supp_data = pd.read_csv(supplemental_data_path, index_col=None)
            data = pd.concat([data, supp_data])

        file_name = "initial_data_assembled.csv"
        save_path = os.path.join(os.path.dirname(data_path), file_name)
        data.to_csv(save_path, index=False)
        return save_path

    def new_batch(self, K, use_neural_net=True):
        """Predicts a new batch of size `K` from the growth media data set 
        `self.data`. Filters batch based on a growth threshold of a neural 
        net's predictions in addition to the cardinality of the growth medias.
        The new batches represent the Agent's predictions for minimal growth
        medias.
    
        Inputs
        ------
        K: int
            Batch size to return.
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
            # print("Random Data\n", data_random)
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
            # print("Added cardinality\n", data)

            if use_neural_net:
                # print("USING NEURAL NET")
                # Get probability of grow using NN
                result = self.predictor.model.predict(
                    data.drop(columns=["cardinality"]).to_numpy()
                )
                # print("RESULT", result)
            else:
                # print("USING NAIVE BAYES")
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
            data = data[data.prediction >= self.growth_threshold]
            # print("results\n", result)
            print(
                f"Num <= {self.growth_threshold}: {result[result < self.growth_threshold].shape[0]}, Num >= {self.growth_threshold}: {result[result >= self.growth_threshold].shape[0]}"
            )
            # print("Threshold filter\n", data)

            if explore_cardinality is not None:
                # Exploration of search space, helpful for getting out of local
                # minima
                target_cardinality = self.minimum_cardinality + explore_cardinality
                data = data[data.cardinality == target_cardinality]
                # print("Target card filter\n", data)

                data_tested = self.data_history.copy()
                data_tested["cardinality"] = data_tested.sum(axis=1)
                data_tested = data_tested[data_tested.cardinality == target_cardinality]
                # print("Target card filter (history)\n", data_tested)

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
                # print("Card filter\n", data)

                # Sort by cardinality then prediction confidence
                data = data.sort_values(
                    by=["cardinality", "prediction"], ascending=[True, False]
                )

            # Only add on random samples if add_random is specified
            if isinstance(add_random, float) and add_random < 1.0 and add_random > 0.0:
                data = pd.concat([data_random, data])
                # print("CONCAT RANDOM", data.shape, data)
            elif add_random >= 1.0:
                data = data_random

            data = data.drop(data.index[n_needed:])

            current_min_cardinality = data.cardinality.min()
            print("BATCH MIN:", current_min_cardinality)

            # print("DATA DataFrame:\n", data)

            # Take only the first n_needed indexes
            min_K_indexes = data.index.to_numpy()

            return min_K_indexes, current_min_cardinality

        def _calc_num_needed(n, current_accuracy):
            distribution = sp.poisson.rvs(0.1 + current_accuracy, size=n)
            counts = dict(collections.Counter(distribution))
            return counts

        ##### Get new batch
        # Evaluate accuracy at current cardinality
        data = self.data.copy()
        data["cardinality"] = data.sum(axis=1)

        x = self.data_history.copy(deep=True)
        y = self.data_labels_history.copy(deep=True)
        x["cardinality"] = x.sum(axis=1)
        current_indexes_at_card = x["cardinality"] == self.minimum_cardinality
        x = x[current_indexes_at_card]
        y = y[current_indexes_at_card]
        x = x.drop(columns=["cardinality"])

        acc, _, _ = self.predictor.evaluate(x.to_numpy(), y.to_numpy())
        print("ACCURACY AT CURRENT CARD:", acc)
        counts = _calc_num_needed(n=K, current_accuracy=acc)
        print("CALC NUM NEEDED COUNTS", counts)
        ### NEW
        min_K_indexes, batch_min_cardinality = np.array([]), float("+inf")
        for explore_var, n_needed in sorted(counts.items()):
            if self.minimum_cardinality + explore_var >= self.model.num_components:
                print("Cannot explore any further...choosing random")
                break
            else:
                add_random = 0.0
                print(f"Adding Card({explore_var})...", f"Number needed: {n_needed}")
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

        ## END NEW

        # min_K_indexes, batch_min_cardinality = _get_min_K(n_needed=K, add_random=0.10)

        # # Start exploration if there are not enough samples in the batch
        # # `explore_var` will be incremented until explore_var + current
        # # minimum cardinality > the number of media components. When this
        # # occurs, it will fill the remaining spaces in the batch with random
        # # data.
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

    def _perform_cycle(
        self,
        batch_train_data,
        batch_train_data_labels,
        batch_size,
        save_location,
        development,
        exp_name,
        output_batch,
    ):
        self.cycle += 1

        print(f"Batch train data: \n{batch_train_data}")
        per_grow = (
            batch_train_data_labels[batch_train_data_labels >= self.growth_threshold]
            .sum()
            .to_numpy()[0]
            * 100
            / batch_train_data_labels.shape[0]
        )
        print(
            f"Batch train data labels: \n{batch_train_data_labels}",
            f"GROW: ({per_grow}%)",
        )

        # Add batch to data history
        self.data_history = pd.concat([self.data_history, batch_train_data])
        batch_train_data_labels.columns = self.data_labels_history.columns
        self.data_labels_history = pd.concat(
            [self.data_labels_history, batch_train_data_labels]
        )

        # Filter new batch results
        # Only include cardinalities < current min that are known to grow
        cardinality = batch_train_data.sum(axis=1)
        batch_results = pd.concat([cardinality, batch_train_data_labels], axis=1)
        batch_results.columns = ["cardinality", "labels"]
        batch_grows = batch_results[
            (batch_results["cardinality"] < self.minimum_cardinality)
            & (batch_results["labels"] >= self.growth_threshold)
        ]

        # Evaluate new batch to check for growth & new min solutions
        if batch_grows.size > 0:
            self.minimum_cardinality = batch_grows.cardinality.min()
            minimum_index = batch_grows[
                batch_grows["cardinality"] == self.minimum_cardinality
            ].index[0]
            self.current_solution = self.data_history.loc[minimum_index, :]
            print(
                f"INPUT FOR NEW MIN ({self.minimum_cardinality}): {self.current_solution}"
            )

        # Remove used experiments from data set
        self.data.drop(batch_train_data.index, inplace=True)
        self.data_labels.drop(batch_train_data.index, inplace=True)

        # Reset predictor to train a new model every time
        # TODO: make this not manually passed in
        self.predictor = type(self.predictor)(
            exp_id="1", n_test=10000, parent_logdir=parent_logdir
        )

        train_acc, test_acc = self.predictor.train(
            self.data_history.to_numpy(),
            self.data_labels_history.to_numpy(),
            self.data.to_numpy(),
            self.data_labels.to_numpy(),
        )
        # self.predictor.train_bayes(
        #     self.data_history.to_numpy(), self.data_labels_history.to_numpy()
        # )

        # Export models every cycle

        model_name = f"NN_{self.model.num_components}_C{self.cycle}.h5"
        model_output_folder = os.path.join(save_location, "neural_nets")
        model_output_path = os.path.join(model_output_folder, model_name)
        if not os.path.exists(model_output_folder):
            os.makedirs(model_output_folder)
        self.predictor.model.save(model_output_path)

        # Get next batch
        new_batch_indexes, batch_min_cardinality = self.new_batch(K=batch_size)
        if new_batch_indexes is None and batch_min_cardinality is None:
            # Terminate agent exploration, DONE!
            return None

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

        # # Get a subset of data to speed up stats
        # test_indexes = np.random.choice(
        #     self.data.index, size=int(0.1 * self.data.shape[0]), replace=False
        # )

        # test_data_x = self.data.loc[test_indexes, :]
        # test_data_y = self.data_labels.loc[test_indexes]

        # # Get NN prediction stats for data set
        # accuracy, precision, recall = self.predictor.evaluate(
        #     test_data_x.to_numpy(), test_data_y.to_numpy()
        # )

        # Output findings and real solution
        print(f"\n\n################ CYCLE {self.cycle} COMPLETE! #################")
        found_solution = set()
        for idx, c in enumerate(self.current_solution):
            if c == 1:
                found_solution.add(self.model.media_ids[idx])

        print(
            f"ACCURACY REMAINING DATA: {train_acc}, BATCH MIN CARDINALITY: {batch_min_cardinality}, OVERALL MIN CARDINALITY: {self.minimum_cardinality}"
        )
        print(
            "MODEL MINIMUM:", len(self.model_answer), "\tMODEL SET:", self.model_answer
        )
        print("DATASET MINIMUM:", len(self.answer), "\tDATASET SET:", self.answer)
        print(
            "CURRENT MINIMUM:",
            int(self.minimum_cardinality),
            "\tCURRENT SOLUTION:",
            found_solution,
        )

        if output_batch:
            utils.batch_to_deep_phenotyping_protocol(
                f"{exp_name}-C{self.cycle}",
                batch_output_df,
                self.model.media_ids,
                self.name_mappings_path,
                development=development,
            )
        return batch_output_path


if __name__ == "__main__":
    # Set GPU
    with tf.device(f"/device:gpu:{args.gpu}"):
        # Silence some Tensorflow outputs
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.keras.backend.set_floatx("float64")

        # RENAME THESE FOR EXPERIMENT
        model_folder = "iSMU-test"
        model_name = "iSMUv01_CDM.xml"
        data_folder = "tweaked_agent_learning_policy"
        # initial_data_filename = "data_20_extrapolated_10.csv"
        initial_data_filename = "train_set_out_real.csv"
        entire_set_filename = "data_20_extrapolated.csv"
        logdir_folder = "logs"
        # data_filename = "L1L2_inout_SMU_NH4.csv"
        # data_filename = "mapped_data_SMU_(-)rescale_(+)NH4.csv"
        exp_name = "BacterAI-SMU"

        if args.new_model:
            m = model.Model(f"models/{model_name}", 20)
            model_path = m.make_minimal_media_models(max_n=1)
            entire_set_path = os.path.join(
                "/".join(model_path.split("/")[:-1]), f"data_{args.num_components}.csv"
            )
        else:
            model_folder_path = os.path.join("models", model_folder)
            model_path = os.path.join(model_folder_path, model_name)
            entire_set_path = os.path.join(model_folder_path, entire_set_filename)

        save_location = os.path.join("data", data_folder)
        initial_data_location = os.path.join(
            save_location, "initial_data", initial_data_filename
        )
        parent_logdir = os.path.join(save_location, logdir_folder)

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

        data = np.genfromtxt(entire_set_path, delimiter=",")
        predictor = neural.PredictNet(
            exp_id="1", n_test=10000, parent_logdir=parent_logdir
        )
        agent = Agent(
            model=m,
            predictor=predictor,
            data_path=entire_set_path,
            minimum_cardinality=args.num_components,
            name_mappings_path="files/name_mappings_aa.csv",
        )

        # Jump start learning with sample of sim data
        assembled_data_path = agent.assemble_initial_data(
            entire_set_path,
            p=0.1,
            supplemental_data_path=initial_data_location,
            even_grow_split=True,
        )

        # Start simulation
        agent.simulate(
            batch_size=378,
            save_location=save_location,
            data_path=assembled_data_path,
            exp_name=exp_name,
            output_batch=args.output_batch,
            parse_required=False,
            make_binary=False,
        )

        # print(
        #     "FULL DATASET ACCURACY:",
        #     agent.predictor.evaluate(data[1:, :-1], data[1:, -1])[0],
        # )
        # predictions = agent.predictor.predict_class(data[1:, :-1])
        # np.savetxt("predictions.csv", predictions, delimiter=",")
        # np.savetxt("media_combos.csv", data[1:, :-1], delimiter=",")

        # from analyze_cycle import analyze

        # analyze("tweaked_agent_learning")

        # print(agent_cont.current_solution)
        # print(agent_cont.model.media_ids)
        # agent_cont.learn(
        #     batch_data_path=os.path.join(save_location, "batches/mapped_data_C1.csv"),
        #     batch_size=378,
        #     save_location=save_location,
        #     exp_name=exp_name,
        #     growth_threshold=0.25,
        #     output_batch=False
        # )
