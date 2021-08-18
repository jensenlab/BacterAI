import argparse
import collections
import copy
import datetime
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

import multiprocessing as mp
from multiprocessing import shared_memory

import neural
import utils
from utils import decoratortimer

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
    "-g",
    "--gpu",
    type=int,
    default=0,
    choices=[0, 1],
    help="Choose GPU (0 or 1).",
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
        ingredient_names,
        state_memory=None,
        growth_cutoff=0.25,
        seed=None,
    ):
        self.growth_model_weights_dir = growth_model_weights_dir
        self.growth_model_weights = None
        self.ingredient_names = ingredient_names
        # if use_multiprocessing:
        #     self.mp_manager = mp.Manager()
        #     self.state_memory = self.mp_manager.dict()
        # else:
        self.state_memory = {}

        if state_memory is not None:
            self.state_memory.update(state_memory)

        self.growth_cutoff = growth_cutoff
        self.np_state = utils.seed_numpy_state(seed)
        self.load_value_weights()

    def get_value_weights(self):
        """
        Helper function for setting growth_model_weights when instantiating the class.
        """
        return self.growth_model_weights

    def load_value_weights(self):
        """
        Helper function for loading growth_model_weights.
        """
        self.reset_state_memory()
        self.growth_model_weights = np.load(self.growth_model_weights_dir)

        self.n_layers = self.growth_model_weights["num_layers"].tolist()
        self.weights = []
        self.biases = []
        self.w_shapes = []
        self.b_shapes = []
        self.w_split_sizes = []
        self.b_split_sizes = []
        for layer in range(self.n_layers):
            w = self.growth_model_weights[f"W{layer}"].astype(np.float64)
            self.w_shapes.append(w.shape)
            w = w.flatten()
            if layer == 0:
                idx = w.size
            else:
                idx = self.w_split_sizes[-1] + w.size
            self.w_split_sizes.append(idx)
            self.weights.append(w)

            b = self.growth_model_weights[f"b{layer}"].astype(np.float64)
            self.b_shapes.append(b.shape)
            b = b.flatten()
            if layer == 0:
                idx = b.size
            else:
                idx = self.b_split_sizes[-1] + b.size
            self.b_split_sizes.append(idx)
            self.biases.append(b)

        self.weights = np.concatenate(self.weights)
        self.biases = np.concatenate(self.biases)

    def evaluate_growth_model(
        self,
        inputs,
        n_layers,
        weights,
        biases,
        w_shapes,
        b_shapes,
        w_split_sizes,
        b_split_sizes,
        return_bool=False,
        **kwargs,
    ):
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
        weights = np.split(weights, w_split_sizes)
        biases = np.split(biases, b_split_sizes)
        ans = inputs
        for layer in range(n_layers):
            w = weights[layer].reshape(w_shapes[layer])
            b = biases[layer].reshape(b_shapes[layer])
            ans = np.matmul(ans, w) + b
            if layer < n_layers - 1:
                ans = np.maximum(0, ans)  # ReLU activation function
            else:
                ans = 1 / (1 + np.exp(-1 * ans))  # Sigmoid activation function

        if return_bool:
            ans = ans >= self.growth_cutoff

        return ans.ravel()

    # def evaluate_growth_model_cached(self, inputs, cache):
    #     """
    #     Gets the cached predicted value from self.state_memory if the state has been evaluated
    #     already, otherwise it computes the prediction and caches it in self.state_memory.
    #     """

    #     l = inputs.shape[0]
    #     unused_indexes = np.arange(l)
    #     used_indexes = []
    #     answers = np.zeros((l,), dtype=np.float64)
    #     for idx, row in enumerate(inputs):
    #         cached_ans = cache.get(row.tobytes(), None)
    #         if cached_ans != None:
    #             # print("CACHED:", answers)
    #             answers[idx] = cached_ans
    #             used_indexes.append(idx)
    #     if len(used_indexes) > 0:
    #         inputs = np.delete(inputs, used_indexes, axis=0)
    #         unused_indexes = np.delete(unused_indexes, used_indexes)

    #     ans = inputs.copy()
    #     n = self.get_value_weights()["num_layers"]
    #     for layer in range(n + 1):
    #         ans = (
    #             np.matmul(ans, self.get_value_weights()[f"W{layer}"])
    #             + self.get_value_weights()[f"b{layer}"]
    #         )
    #         if layer < n - 1:
    #             ans = np.maximum(0, ans)  # ReLU activation function
    #         else:
    #             ans = 1 / (1 + np.exp(-1 * ans))  # Sigmoid activation function

    #     if inputs.size > 0:
    #         for k, v in zip(inputs, ans):
    #             # print(f"{k}: {v}")
    #             cache[k.tobytes()] = v[0]

    #     np.put(answers, unused_indexes, ans.ravel())
    #     return answers

    @decoratortimer(2)
    def find_candidates(self, state, **kwargs):
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

        size = state.size
        does_grow = np.empty((size, size))
        for i in range(state.size):
            does_grow[i] = self.remove_ingredient(state, i)

        does_grow = self.evaluate_growth_model(does_grow, **kwargs)
        candidates = np.argwhere(does_grow >= self.growth_cutoff).ravel()
        return candidates

    def get_state_memory(self):
        """
        Return state_memory
        """

        return self.state_memory

    def reset_state_memory(self):
        """
        Reset state_memory
        """

        self.state_memory = {}

    def save_state_memory(self):
        mem = pd.DataFrame([list(k) + [v] for k, v in self.state_memory.items()])
        mem.to_csv(
            f"state_memory_{datetime.datetime.now().isoformat()}.csv", index=None
        )

    def __getstate__(self):
        """
        Automatically called when pickling this class, to avoid pickling errors
        when saving model weights .npz object.
        """
        state = dict(self.__dict__)
        state["growth_model_weights"] = None
        return state

    def trajectory(
        self,
        state,
        numpy_state,
        horizon,
        limit,
        grow_advantage=None,
        **kwargs,
    ):
        """
        Calculates the trajectory reward of a given state. A random ingredient is chosen from remaining
        ingredients until the solution results in a 'no grow' prediction from the `self.growth_model`.

        Inputs
        ------
        state: list(str)
            The present ingredients in the solution.
        horizon: int
            The depth of the rollout.
        length: int
            The length the full media
        grow_advantage: int or None
            Parameter used to set the relative probability of choosing an ingredient that is
            predicted to grow. Values <1 will skew the choice towards an ingredient that
            results in no growth, and the opposite for values >1. If 1 or None is passed in,
            the probabilites for each ingredient are set to 1/N where N=len(state). Otherwise,
            ingredients' values will be set to `grow_advantage`/N or 1/N, if they grow or don't grow,
            respectively.

        Return
        -------
        reward: float
            Reward = # of ingredients removed (len full media + current step) * growth_result (of
            most recent simulation above the growth cutoff).
        """

        trajectory_states = np.tile(state, (limit, 1))
        length = trajectory_states[0].sum()
        rewards = np.zeros((limit,))
        order = list(range(limit))

        # Random walk to remove 'horizon' ingredients
        for step in range(1, horizon + 1):
            if trajectory_states.size <= 0:
                break

            choices = np.argwhere(trajectory_states == 1)
            if choices.size == 0:
                break

            s0 = np.r_[
                0,
                np.flatnonzero(choices[1:, 0] > choices[:-1, 0]) + 1,
                choices.shape[0],
            ]

            for i in range(s0.shape[0] - 1):
                row = choices[s0[i], 0]
                trajectory_states[
                    row, numpy_state.choice(choices[s0[i] : s0[i + 1], 1], 1, False)
                ] = 0

            length = trajectory_states[0].sum()
            # grow_results = self.evaluate_growth_model_cached(
            #     trajectory_states, state_memory
            # )
            grow_results = self.evaluate_growth_model(
                trajectory_states,
                **kwargs,
            )
            cardinality_rewards = grow_results * (length + step)

            idx_dels = list()
            for i, r in enumerate(grow_results):
                if r <= self.growth_cutoff:
                    idx_dels.append(i)
                    continue
                rewards[order[i]] = cardinality_rewards[i]

            trajectory_states = np.delete(trajectory_states, idx_dels, axis=0)
            order = [o for idx, o in enumerate(order) if idx not in idx_dels]

        # # Set choice probabilities
        # p = np.ones(length)
        # if grow_advantage != None and grow_advantage != 1:
        #     candidates = self.find_candidates(trajectory_state, state_memory)
        #     for idx, i in enumerate(trajectory_state):
        #         if i in candidates:
        #             p[idx] = grow_advantage
        # p /= p.sum()

        return rewards

    # @decoratortimer(2)
    def remove_ingredient(self, state, idxs):
        """
        Removes the ingredient `to_remove` from state if it is present.

        Inputs
        ------
        state: np.array(int)
            The present ingredients in the solution.
        idx: np.array(int)
            The indexes to remove

        Return
        -------
        state: np.array(int)
            The remaining ingredients in the solution.
        """

        s = state.copy()
        np.put(s, idxs, 0)
        return s

    def partition(self, n, k):
        """Split an integer n into k groups as evenly as possible.

        Example
        -------
        >>> partition(13, 5)
        [3, 3, 3, 2, 2]
        """

        sizes = [n // k for _ in range(k)]
        for i in range(n % k):
            sizes[i] += 1
        return sizes

    def _rollout_helper(
        self,
        state,
        limit,
        reward_shape,
        starting_idx,
        action,
        action_idx,
        seed=None,
        **kwargs,
    ):
        """
        Helper function for multiprocessing
        """

        kwargs["limit"] = limit
        existing_shm = shared_memory.SharedMemory(name="rw_shm")
        rewards = np.ndarray(reward_shape, dtype=np.float64, buffer=existing_shm.buf)
        numpy_state = utils.seed_numpy_state(seed)
        test_state = self.remove_ingredient(state, action)
        results = self.trajectory(
            test_state,
            numpy_state,
            **kwargs,
        )
        rewards[action_idx, starting_idx : starting_idx + limit] = results
        existing_shm.close()

    @decoratortimer(2)
    def perform_rollout(
        self,
        state,
        limit,
        horizon,
        available_actions=None,
        log_graph=True,
        threads=None,
    ):
        """
        Performs an Monte Carlo Rollout Simulation for the solution `self`. The
        trajectories for each remaining ingredient are averaged over `limit` times. The
        ingredient with the lowest predicted score (equivalent to the cardinality of the
        solution) is returned.

        Inputs
        ------
        limit: int
            The number of times a trajectory will be calculated for each ingredient.
        horizon: int
            The depth of the rollout.
        log_graph: boolean
            Flag to enable the graphical output.

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

        if isinstance(state, list):
            ing = np.array(self.ingredient_names)
            state = np.isin(ing, state).astype(int).reshape((1, -1))
        elif not isinstance(state, np.ndarray):
            raise Error("Invalid state, must be type list or np.ndarray")

        size = len(self.ingredient_names)
        if available_actions is None:
            available_actions = self.find_candidates(
                state,
                self.n_layers,
                self.weights,
                self.biases,
                self.w_shapes,
                self.b_shapes,
                self.w_split_sizes,
                self.b_split_sizes,
            )
        else:
            ing = np.array(self.ingredient_names)
            available_actions = (
                np.isin(ing, available_actions).astype(int).reshape((1, -1))
            )

        LOG.debug(f"Available actions: {available_actions}")

        if log_graph:
            all_results = np.empty((size, limit))

        actions = np.argwhere(available_actions.ravel() == 1).ravel()

        # Create a NumPy array backed by shared memory
        rewards = np.zeros((len(actions), limit), dtype=np.float64)
        shm = shared_memory.SharedMemory(
            name="rw_shm", create=True, size=rewards.nbytes
        )
        rewards_shared = np.ndarray(rewards.shape, dtype=np.float64, buffer=shm.buf)
        rewards_shared[:] = rewards[:]

        # Set up multiprocessing groups
        if threads == None:
            threads = mp.cpu_count()

        if threads > len(actions):
            process_groups = self.partition(threads, len(actions))
        else:
            process_groups = [1] * len(actions)

        n_per_group = [
            [i for i in self.partition(limit, n) if i > 0] for n in process_groups
        ]

        keywords = {
            "horizon": horizon,
            "n_layers": self.n_layers,
            "weights": self.weights,
            "biases": self.biases,
            "w_shapes": self.w_shapes,
            "b_shapes": self.b_shapes,
            "w_split_sizes": self.w_split_sizes,
            "b_split_sizes": self.b_split_sizes,
            "state_memory": self.state_memory,
        }
        processes = []
        for action_idx, (action, limits) in enumerate(zip(actions, n_per_group)):
            reward_idx = 0
            for limit in limits:
                p = mp.Process(
                    target=self._rollout_helper,
                    args=(
                        state,
                        limit,
                        rewards_shared.shape,
                        reward_idx,
                        action,
                        action_idx,
                        self.np_state.randint(2 * 32 - 1, size=1),
                    ),
                    kwargs=keywords,
                )
                processes.append(p)
                p.start()
                reward_idx += limit

        for p in processes:
            p.join()

        if log_graph:
            for y in all_results:
                plt.plot(range(limit), y)
            plt.legend(available_actions, bbox_to_anchor=(1.05, 1.0), loc="upper left")
            plt.xlabel("after N trajectories")
            plt.ylabel("average value")
            plt.title("average trajectory over time with ingredient removed")
            plt.tight_layout()
            plt.savefig("rollout_result.png")

        # if not rewards:
        #     return None

        # Sort rewards based on value, descending
        rewards = {
            a: (r.min(), r.max(), r.mean()) for a, r in zip(actions, rewards_shared)
        }
        rewards = {
            self.ingredient_names[k]: v
            for k, v in sorted(rewards.items(), key=lambda x: x[1][2], reverse=True)
        }

        # Closed shared memory
        shm.close()
        shm.unlink()

        LOG.info("Calculated Rewards:")
        for k, v in rewards.items():
            LOG.info(f"{k} ->\t{v}")

        return rewards


if __name__ == "__main__":

    # data_path = "models/iSMU-test/data_20_extrapolated.csv"
    # data = pd.read_csv(data_path)
    # data = data.drop(columns=["grow"])
    # starting_state = data.sample(1).to_dict("records")[0]
    # starting_state = {
    #     "ala": 0,
    #     "gly": 0,
    #     "arg": 1,
    #     "asn": 0,
    #     "asp": 0,
    #     "cys": 1,
    #     "glu": 0,
    #     "gln": 1,
    #     "his": 0,
    #     "ile": 1,
    #     "leu": 1,
    #     "lys": 0,
    #     "met": 1,
    #     "phe": 1,
    #     "ser": 0,
    #     "thr": 1,
    #     "trp": 1,
    #     "tyr": 1,
    #     "val": 0,
    #     "pro": 1,
    # }

    ingredients = [
        "ala",
        "gly",
        "arg",
        "asn",
        "asp",
        "cys",
        "glu",
        "gln",
        "his",
        "ile",
        "leu",
        "lys",
        "met",
        "phe",
        "ser",
        "thr",
        "trp",
        "tyr",
        "val",
        "pro",
    ]
    starting_state = np.ones((1, 20))
    available_actions = ingredients.copy()

    # starting_state[0, 3:8] = 0
    # del available_actions[3:8]
    # print(available_actions)
    # print(starting_state)

    search = MCTS(
        growth_model_weights_dir="models/SMU_NN_oracle/weights.npz",
        ingredient_names=ingredients,
        # seed=0,
    )
    # import pprint
    # dill.detect.trace(True)
    # pprint.pprint(dill.detect.errors(search, depth=1))
    # dill.detect.errors(search)
    # dill.pickles(search)

    # print("\nStarting State:", starting_state)

    search.perform_rollout(
        state=starting_state,
        limit=200,
        horizon=5,
        available_actions=available_actions,
        log_graph=False,
    )

    # print(len(search.get_state_memory()))  # , search.get_state_memory())

    # from timeit import Timer

    # t1 = Timer(lambda: _f1(trajectory_states))
    # t2 = Timer(lambda: _f2(trajectory_states))
    # print()
    # print(f"1 avg: {round(t1.timeit(number=100),3) * 10}ms",)
    # print(f"2 avg: {round(t2.timeit(number=100),3) * 10}ms",)
