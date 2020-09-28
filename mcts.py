import argparse
import collections
import copy
import logging
import operator
import os
import random
import sys

# Suppress Tensorflow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocess as mp
import tensorflow as tf
from tqdm import tqdm, trange

import neural_pretrain as neural
import utils

# Logging set up
logger = logging.getLogger(__name__)
LOGLEVELS = (logging.DEBUG, logging.INFO, logging.ERROR)
# Levels, descending
LOGTYPE = collections.namedtuple("LOGTYPE", "debug info error")
LOG = LOGTYPE(logger.debug, logger.info, logger.error)

INDENT = "  "

# CLI argument set up
parser = argparse.ArgumentParser(description="Run mcts.py")

parser.add_argument(
    "-g", "--gpu", type=int, default=0, choices=[0, 1], help="Choose GPU (0 or 1).",
)

parser.add_argument(
    "-l",
    "--log",
    type=int,
    default=3,
    choices=range(1, 4),
    help="Choose a minimum log level: 1 for debug, 2 for info, 3 for error.",
)

args = parser.parse_args()

logging.basicConfig(format="%(levelname)s: %(message)s")
logger.setLevel(LOGLEVELS[args.log - 1])


def get_indent(n):
    return "".join([INDENT for _ in range(n)])


class MCTS(object):
    def __init__(
        self,
        growth_model_weights_dir,
        current_state,
        state_memory=None,
        growth_cutoff=0.25,
        seed=None,
    ):
        self.growth_model_weights_dir = growth_model_weights_dir
        self.growth_model_weights = None
        self.all_ingredients = list(current_state.keys())
        self.current_state = self.dict_to_ingredients(current_state)
        self.state_memory = {}
        if state_memory:
            self.state_memory.update(state_memory)

        self.growth_cutoff = growth_cutoff
        self.np_state = utils.seed_numpy_state(seed)

    def get_value_weights(self):
        """
        Helper function for setting growth_model_weights when instantiating the class.
        """
        if not self.growth_model_weights:
            self.growth_model_weights = np.load(self.growth_model_weights_dir)
        return self.growth_model_weights

    def evaluate_growth_model(self, inputs, return_bool=False):
        """
        Computes the inference value using the stored neural network weights from self.growth_model_weights
        of the inputs. This corresponds to the predicted probability of growth of a media. Using 
        these cached weights allows for multiprocessing and reduced overhead compared to 
        an equivalent call to Tensorflow on the trained model, while provinding the same value.

        Inputs
        ------
        inputs: np.array(int)
            A boolean array representing a media, 1 = present, 0 = not present.
        return_bool: bool, default=False
            A flag to determine if the output should be converted to a bool.

        Return
        -------
        answer: float or bool
            The neural network output prediction.
        """

        n = self.get_value_weights()["num_layers"]
        answer = inputs
        for i in range(n):
            answer = np.matmul(answer, self.get_value_weights()[f"W{i}"])
            answer += self.get_value_weights()[f"b{i}"]
            if i < n - 1:
                answer[answer <= 0] = 0  # ReLU activation function
            else:
                answer = 1 / (1 + np.exp(-1 * answer))  # Sigmoid activation function
        answer = answer[0, 0]
        if return_bool:
            answer = 1 if answer >= self.growth_cutoff else 0
        return answer

    def get_value_cache(self, state):
        """
        Gets the cached predicted value from self.state_memory if the state has been evaluated
        already, otherwise it computes the prediction and caches it in self.state_memory.

        Inputs
        ------
        state: list(str)
            The present ingredients in the solution.
        
        Return
        -------
        result: float
            The cached value.
        """

        state = self.ingredients_to_input(state)

        key = tuple(state.tolist()[0])
        if key in self.state_memory.keys():
            result = self.state_memory[key]
        else:
            result = self.evaluate_growth_model(state)
            self.state_memory[key] = result
        return result

    def find_candidates(self, state):
        """
        Finds candidate ingredients to remove. Without these inputs, the `self.growth_model`
        still predicts `grow.` State results are cached in `self.state_memory` for improved
        performance.

        Inputs
        ------
        state: list(str)
            The present ingredients in the solution.
        
        Return
        -------
        candidates: list(str)
            The candidate ingredients.
        """

        does_grow = {}
        for ingredient in state:
            test_state = self.remove_from_list(state, ingredient)
            does_grow[ingredient] = self.get_value_cache(test_state)

        candidates = [k for k, v in does_grow.items() if v >= self.growth_cutoff]
        return candidates

    def get_state_memory(self):
        """
        Return state_memory
        """

        return self.state_memory

    def __getstate__(self):
        """
        Automatically called when pickling this class, to avoid pickling errors 
        when saving model weights .npz object.
        """
        state = dict(self.__dict__)
        state["growth_model_weights"] = None
        return state

    def trajectory(self, state, horizon, numpy_state, grow_advantage=None):
        """
        Calculates the trajectory of a given state. A random ingredient is chosen from remaining 
        ingredients until the solution results in a 'no grow' prediction from the `self.growth_model`.

        Inputs
        ------
        state: list(str)
            The present ingredients in the solution.
        horizon: int
            The depth of the rollout.
        grow_advantage: int or None
            Parameter used to set the relative probability of choosing an ingredient that is
            predicted to grow. Values <1 will skew the choice towards an ingredient that 
            results in no growth, and the opposite for values >1. If 1 or None is passed in,
            the probabilites for each ingredient are set to 1/N where N=len(state). Otherwise,
            ingredients' values will be set to `grow_advantage`/N or 1/N, if they grow or don't grow,
            respectively.
        
        Return
        -------
        length: int
            The minimum size of media that still results in growth in this random trajectory. 
        """
        trajectory_state = copy.copy(state)
        length = len(trajectory_state)

        # Random walk to remove 'horizon' ingredients
        for step in range(horizon):
            if length <= 0:
                break
            # Set choice probabilities
            p = np.ones(length)
            if grow_advantage != None and grow_advantage != 1:
                candidates = self.find_candidates(trajectory_state)
                for idx, i in enumerate(trajectory_state):
                    if i in candidates:
                        p[idx] = grow_advantage
            p /= p.sum()

            # Pick a random ingredient from the current state
            ingredient = numpy_state.choice(trajectory_state, p=p)
            trajectory_state = self.remove_from_list(trajectory_state, ingredient)
            length = len(trajectory_state)

            # Cache calculated state values if we haven't seen the state yet,
            # otherwise ask the value model for the prediction
            grow_result = self.get_value_cache(trajectory_state)
            if grow_result <= self.growth_cutoff:
                cardinality_reward = grow_result * (len(state) + step)
                return

        cardinality_reward = grow_result * (len(state) + horizon)
        return cardinality_reward

    def spsa(self):
        """Not Implemented Yet"""
        pass

    def dict_to_ingredients(self, state):
        """
        Converts a dictionary of ingredients to a list of present ingredients.

        Inputs
        ------
        state: dict(str -> int)
            A dictionary mapping the ingredient to a 0 or 1, corresponding to not present or
            present in the solution, respectively.

        Return
        -------
        state: list(str)
            The present ingredients in a solution.
        """

        ingredients = [k for k, v in state.items() if v == 1]
        return ingredients

    def remove_from_list(self, state, to_remove):
        """
        Removes the ingredient `to_remove` from state if it is present.
        
        Inputs
        ------
        state: list(str)
            The present ingredients in the solution.
        to_remove: str or list(str)
            The ingredient(s) to remove
        
        Return
        -------
        state: list(str)
            The remaining ingredients in the solution.
        """
        if isinstance(to_remove, list):
            state = [i for i in state if i in to_remove]
        else:
            state = [i for i in state if i != to_remove]
        return state

    def ingredients_to_input(self, state):
        """
        Converts to numpy a state into the input form for `self.growth_model`.
        
        Inputs
        ------
        state: list(str)
            The present ingredients in the solution.
        
        Return
        -------
        inputs: np.array(int)
            A boolean array representation of the state.
        """

        inputs = np.array(self.all_ingredients)
        inputs = np.isin(inputs, state).astype(float).reshape((1, -1))
        return inputs

    def run(self, starting_state, n=2, **kwargs):

        if n > 1:
            results = self.run_n_stage(starting_state, n, **kwargs)
        else:
            results = self.perform_rollout(starting_state, **kwargs)

        print(results)
        return results

    def run_n_stage(self, state, n, **kwargs):
        print(f"stage: {n}")
        if n == 1:
            print("performing rollout")
            return self.perform_rollout(self.current_state, **kwargs)
        else:

            def _run_helper(_state, _n, args):
                print(f"\tremoving: {set(state) - set(_state)}")
                return self.run_n_stage(_state, _n, **args)

            n_items = len(state)
            if n - 1 == 1:
                kwargs["use_multiprocessing"] = False
                threads = mp.cpu_count() - 1
                with mp.Pool(threads) as pool:
                    all_results = pool.starmap(
                        _run_helper,
                        zip(
                            [self.remove_from_list(state, c) for c in state],
                            [n - 1] * n_items,
                            [kwargs] * n_items,
                        ),
                    )
                mapped_results = dict(zip(state, all_results))
            else:
                all_results = map(
                    _run_helper,
                    [self.remove_from_list(state, c) for c in state],
                    [n - 1] * n_items,
                    [kwargs] * n_items,
                )
                mapped_results = dict(zip(state, all_results))
            return mapped_results

    def perform_rollout(
        self,
        state,
        limit,
        horizon,
        available_actions=None,
        grow_advantage=None,
        log_graph=True,
        use_multiprocessing=True,
    ):
        """
        Performs an Monte Carlo Rollout Simulation for the solution `self.current_state`. The 
        trajectories for each remaining ingredient are averaged over `limit` times. The
        ingredient with the lowest predicted score (equivalent to the cardinality of the 
        solution) is returned.
        
        Inputs
        ------
        limit: int
            The number of times a trajectory will be calculated for each ingredient.
        horizon: int
            The depth of the rollout.
        grow_advantage: int or None
            Parameter used to set the relative probability of choosing an ingredient that is
            predicted to grow.
        log_graph: boolean
            Flag to enable the graphical output.
        use_multiprocessing: boolean or int
            Flag to enable the multiprocessing. If enabled it defaults to using n-1 threads, or
            the number of threads passed in.
        
        Return
        -------
        rewards: dict(str -> float)
            The predict growth values for each ingredient after `limit` iterations of rollout.

        Outputs
        ------
        'rollout_result.png': PNG image
            Graph of each ingredient's average trajectories over time when `log_graph` 
            is set to `True`.
        """

        if available_actions is None:
            available_actions = self.find_candidates(state)
            # available_actions = self.current_state
        rewards = {}
        LOG.debug(f"Available actions: {available_actions}")

        if log_graph:
            all_results = np.empty((len(available_actions), limit))

        if use_multiprocessing is False:
            # t1 = tqdm(available_actions, desc="Exploring Actions", leave=True)
            # t2 = tqdm(total=limit, desc="Calculating Trajectory", leave=True)
            for i, action in enumerate(available_actions):
                # t1.set_description(f"Exploring ({action})")
                # t1.refresh()  # to show immediately the update
                test_state = self.remove_from_list(state, action)

                results = np.empty(limit)
                results.fill(np.nan)
                # t2.reset()
                for j in range(limit):
                    results[j] = self.trajectory(
                        test_state,
                        horizon,
                        self.np_state,
                        grow_advantage=grow_advantage,
                    )

                    intermediate_result = np.nanmean(results) if j is not 0 else 0
                    # t2.set_description(
                    #     f"Calculating Trajectory ({round(intermediate_result, 3)})"
                    # )
                    # t2.update()  # to show immediately the update
                    if log_graph:
                        all_results[i, j] = intermediate_result
                rewards[action] = (np.min(results), np.max(results), np.mean(results))

            # t2.close()
            # t1.close()
        else:
            # Set up multiprocessing helper function
            def _rollout_multi_helper(action, limit, grow_advantage, seed=None):
                numpy_state = utils.seed_numpy_state(seed)
                test_state = self.remove_from_list(state, action)
                results = np.empty(limit)
                results.fill(np.nan)

                # Use TQDM if logging level is INFO or below
                if args.log <= 2:
                    t_range = trange(limit, desc="Calculating Trajectory", leave=True)
                else:
                    t_range = limit

                for j in range(limit):
                    if args.log <= 2:
                        intermediate_result = np.nanmean(results) if j is not 0 else 0
                        t_range.set_description(
                            f"Calculating Trajectory ({round(intermediate_result, 3)})"
                        )
                        t_range.update()  # to show immediately the update

                    results[j] = self.trajectory(
                        test_state, horizon, numpy_state, grow_advantage=grow_advantage,
                    )

                if args.log <= 2:
                    t_range.close()

                LOG.debug(f"Results: {results}")
                results = (np.min(results), np.max(results), np.mean(results))
                return results

            # Set up multiprocessing
            if isinstance(use_multiprocessing, bool):
                threads = mp.cpu_count() - 1
            elif isinstance(use_multiprocessing, int):
                threads = use_multiprocessing

            # with mp.get_context("spawn").Pool(threads) as pool:
            with mp.Pool(threads) as pool:
                # pool = mp.Pool(threads)
                n_actions = len(available_actions)
                rewards = pool.starmap(
                    _rollout_multi_helper,
                    zip(
                        available_actions,
                        [limit] * n_actions,
                        [grow_advantage] * n_actions,
                        self.np_state.randint(2 * 32 - 1, size=n_actions).tolist(),
                    ),
                )
                # pool.close()
            rewards = dict(zip(available_actions, rewards))

        if log_graph:
            for y in all_results:
                plt.plot(range(limit), y)
            plt.legend(available_actions, bbox_to_anchor=(1.05, 1.0), loc="upper left")
            plt.xlabel("after N trajectories")
            plt.ylabel("average value")
            plt.title("average trajectory over time with ingredient removed")
            plt.tight_layout()
            plt.savefig("rollout_result.png")

        if not rewards:
            return None

        # Sort rewards based on value, descending
        rewards = {
            k: v
            for k, v in sorted(rewards.items(), key=lambda x: x[1][2], reverse=True)
        }
        LOG.info("Calculated Rewards:")
        for k, v in rewards.items():
            LOG.info(f"{k} ->\t{v}")

        return rewards


if __name__ == "__main__":

    with tf.device(f"/device:gpu:{args.gpu}"):
        # data_path = "models/iSMU-test/data_20_extrapolated.csv"
        # data = pd.read_csv(data_path)
        # data = data.drop(columns=["grow"])
        # starting_state = data.sample(1).to_dict("records")[0]
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

        available_actions = [k for k, v in starting_state.items() if v == 1]
        # starting_state = data.iloc[112123, :]
        # starting_state = starting_state.to_dict()
        search = MCTS(
            # growth_model_dir="data/neuralpy_optimization_expts/052220-sparcity-3/working_model",
            growth_model_weights_dir="data/neuralpy_optimization_expts/052220-sparcity-3/working_model/weights.npz",
            current_state=starting_state,
            seed=10,
        )
        # dill.detect.trace(True)
        # dill.detect.errors(mcts)
        # import pprint

        # pprint.pprint(dill.detect.errors(mcts, depth=1))
        # print("\nStarting State:", starting_state)
        # rollout.simulate(1000)
        # best_action = search.perform_rollout(
        #     available_actions=None,
        #     state=search.current_state,
        #     horizon=4,
        #     limit=10,
        #     grow_advantage=1,
        #     use_multiprocessing=True,
        # )

        best_action = search.run(
            starting_state=search.current_state,
            n=2,
            available_actions=None,
            horizon=4,
            limit=10,
            grow_advantage=1,
            use_multiprocessing=True,
        )
