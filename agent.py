import argparse
import copy
import csv
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import search
import neural_pretrain as neural
import utils


parser = argparse.ArgumentParser(description="Run agent.py")

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


class Controller:
    def __init__(
        self,
        ingredients,
        value_model_dir,
        value_model_weights_dir,
        agents=[],
        simulation_data_path=None,
    ):
        self.agents = agents  # list of Agent()
        self.ingredients = ingredients  # list of str for each possible ingredient
        self.value_model = neural.PredictNet.from_save(value_model_dir)
        self.value_model_weights_dir = value_model_weights_dir
        # self.value_model = neural.PredictNet(exp_id,
        # n_test,
        # parent_logdir
        # )
        self.shared_history = pd.DataFrame()
        self.growth_threshold = 0.25

        self.simulation_data_dict = None
        self.simulation_data = None
        if simulation_data_path:
            # str pointing to csv data
            self.simulation_data_dict = self.create_simulation_dict(
                simulation_data_path
            )
            self.simulation_data = pd.read_csv(simulation_data_path, index_col=None)
            # print(self.simulation_data_dict)

    def add_agents(self, agents):
        self.agents.extend(agents)

    def update_history(self, new_data):
        # TODO: check if new data has any collisions
        self.shared_history = pd.concat([self.shared_history, new_data])

    def simulate_random_initial_data(self, n, supplemental_data_path=None):
        print("Setting initial data")
        supplemental_data = pd.read_csv(supplemental_data_path, index_col=None)

        random_indexes = np.random.choice(
            self.simulation_data.index, size=n, replace=False
        )
        initial_data = self.simulation_data.loc[random_indexes, :]

        # Look for duplicates in supplemental data?
        initial_data = pd.concat([initial_data, supplemental_data])

        self.shared_history = initial_data

    def create_simulation_dict(self, simulation_data_path):
        print("Creating simulation dict")

        data_dict = {}
        with open(simulation_data_path, newline="") as f:
            f.readline()
            reader = csv.reader(f, delimiter=",")
            for row in tqdm(reader):
                key = tuple([int(i) for i in row[:-1]])
                data_dict[key] = int(float(row[-1]) >= self.growth_threshold)

        # data = pd.read_csv(simulation_data_path, index_col=None)
        # data_dict = {}
        # columns = [c for c in data.columns if not "grow"]
        # for idx, row in tqdm(data.iterrows()):
        #     data_dict[tuple(row[columns].values)] = int(
        #         row["grow"] >= self.growth_threshold
        #     )
        return data_dict

    def get_simulation_growth(self, state):
        input_data = tuple(state.values())
        return self.simulation_data_dict.get(input_data, None)

    def simulate(self, n_agents, starting_state):
        if self.shared_history is None:
            raise Exception("ERROR: You need to add input date to data shared_history")

        input_data = self.shared_history.values
        x, y = input_data[:, :-1], input_data[:, -1]
        self.value_model.train(x, y)

        if self.simulation_data_dict is None:
            raise Exception("ERROR: You need to set the simulation data")

        # Initialze Agents
        for i in range(n_agents):
            print(f"Creating agent {i}")
            a = Agent(self, self.value_model_weights_dir, starting_state=starting_state)
            self.agents.append(a)

        state_memory = None
        round_n = 0
        while len(self.agents):
            print(f"\n\n################# STARTING ROUND {round_n} #################")
            print(f"Current History: {self.shared_history}")
            for i, agent in enumerate(self.agents):
                print(f"Simulating agent {i}")
                # Run simulation on Agent's next choice
                next_media, state_memory = agent.get_next_media(state_memory)
                if next_media is None:
                    print(f"Couldn't find a next media for Agent {i}!")
                    self.agents = [
                        self.agents[j] for j in range(len(self.agents)) if j is not i
                    ]
                    continue
                agent.set_state(next_media)
                simulation_result = self.get_simulation_growth(next_media)
                print("Next media:", next_media, f"Grow Result: {simulation_result}")
                # Update history
                new_row = pd.DataFrame(next_media, index=[0])
                new_row["grow"] = simulation_result
                print(f"ADDED DATA: {new_row}")
                self.update_history(new_row)
            x, y = self.shared_history.values[:, :-1], self.shared_history.values[:, -1]
            self.value_model.train(x, y)
            round_n += 1

        # import pathos.multiprocessing as mp
        # pool = mp.ProcessingPool(mp.cpu_count() - 1)
        # rewards = pool.map(run_agent, self.agents)

        # Retrain value_net using updated history
        # self.value_model = self.value_model
        # print("updating model with", self.shared_history)


# def run_agent(agent):
#     next_media, _ = agent.get_next_media()
#     agent.set_state(next_media)
#     simulation_result = self.get_simulation_growth(next_media)
#     print("Next media:", next_media, f"Grow Result: {simulation_result}")
#     # Update history
#     new_row = pd.DataFrame(next_media, index=[0])
#     new_row["grow"] = simulation_result
#     return new_row


class Agent:
    def __init__(self, commander, value_model_weights_dir, starting_state=None):

        self.commander = commander
        self.value_model_weights_dir = value_model_weights_dir
        self.experiment_history = pd.DataFrame()

        self.current_state = None
        if starting_state:
            self.current_state = starting_state

    def set_state(self, new_state):
        self.current_state = new_state

    def get_next_media(self, state_memory=None):
        # results is a dict() from ingredient_removed -> growth result

        # Perform L10 on current state
        # First, generate dict of media states to test
        l1o_experiments = {}
        still_present = [i for i, present in self.current_state.items() if present]
        for ingredient in still_present:
            new_expt = copy.copy(self.current_state)
            new_expt[ingredient] = 0
            l1o_experiments[ingredient] = new_expt
        # Second, get growth values from simulation
        l1o_results = {
            i: self.commander.get_simulation_growth(expt)
            for i, expt in l1o_experiments.items()
        }

        # Available actions are only L1Os that grow
        available_actions = [i for i, does_grow in l1o_results.items() if does_grow]

        # Initialize most updated MCTS
        mcts = search.MCTS(
            self.value_model_weights_dir, self.current_state, state_memory
        )

        # Perform rollout to determine next best media
        best_action = mcts.perform_rollout(
            limit=1000, available_actions=available_actions, log_graph=False
        )
        state_memory = mcts.get_state_memory()
        # print(len(state_memory))
        print("Chosen ingredient to remove:", best_action)

        # Generate new media dict and return
        if best_action is None:
            return None, state_memory
        next_media = copy.copy(self.current_state)
        next_media[best_action] = 0
        return next_media, state_memory


if __name__ == "__main__":
    # Set GPU
    with tf.device(f"/device:gpu:{args.gpu}"):
        # Silence some Tensorflow outputs
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.keras.backend.set_floatx("float64")

        # RENAME THESE FOR EXPERIMENT
        data_folder = "tweaked_agent_learning_policy"
        # initial_data_filename = "data_20_extrapolated_10.csv"
        initial_data_filename = "train_set_out_real.csv"
        entire_set_filename = "data_20_extrapolated.csv"
        logdir_folder = "logs"
        # data_filename = "L1L2_inout_SMU_NH4.csv"
        # data_filename = "mapped_data_SMU_(-)rescale_(+)NH4.csv"
        exp_name = "BacterAI-SMU"

        save_location = os.path.join("data", data_folder)
        initial_data_location = os.path.join(
            save_location, "initial_data", initial_data_filename
        )
        parent_logdir = os.path.join(save_location, logdir_folder)

        ingredients = [
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

        # Starting state initialzation, everything is in the media
        # starting_state = {i: 1 for i in ingredients}
        starting_state = {
            "ala_exch": 0,
            "gly_exch": 0,
            "arg_exch": 1,
            "asn_exch": 0,
            "asp_exch": 0,
            "cys_exch": 1,
            "glu_exch": 0,
            "gln_exch": 1,
            "his_exch": 0,
            "ile_exch": 1,
            "leu_exch": 1,
            "lys_exch": 0,
            "met_exch": 1,
            "phe_exch": 1,
            "ser_exch": 0,
            "thr_exch": 1,
            "trp_exch": 1,
            "tyr_exch": 1,
            "val_exch": 0,
            "pro_exch": 1,
        }
        # starting_state = {i: np.random.randint(0, 2) for i in ingredients}

        controller = Controller(
            ingredients=ingredients,
            value_model_dir="data/neuralpy_optimization_expts/052220-sparcity-3/working_model",
            value_model_weights_dir="data/neuralpy_optimization_expts/052220-sparcity-3/working_model/weights.npz",
            simulation_data_path="models/iSMU-test/data_20_extrapolated.csv",
        )
        controller.simulate_random_initial_data(
            n=2500,
            supplemental_data_path="data/iSMU-test/initial_data/train_set_L1OL2O.csv",
        )
        controller.simulate(2, starting_state)
