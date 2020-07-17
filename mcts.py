import argparse
import copy
import operator
import random


import dill
import pathos.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tqdm import tqdm, trange

import neural_pretrain as neural


class MCTS:
    def __init__(self, value_model_weights_dir, current_state, state_memory=None):
        self.value_model_weights_dir = value_model_weights_dir
        self.value_model_weights = None
        # self.value_model = neural.PredictNet.from_save(value_model_dir)
        self.all_ingredients = list(current_state.keys())
        self.current_state = self.dict_to_ingredients(current_state)
        self.state_memory = {}
        if state_memory:
            self.state_memory.update(state_memory)
            # print(self.state_memory)
        self.growth_cutoff = 0.25

    def get_value_weights(self):
        if not self.value_model_weights:
            self.value_model_weights = np.load(self.value_model_weights_dir)
        return self.value_model_weights

    def evaluate_value_model(self, inputs, return_bool=False):
        n = self.get_value_weights()["num_layers"]
        answer = inputs
        for i in range(n):
            answer = np.matmul(answer, self.get_value_weights()[f"W{i}"])
            answer += self.get_value_weights()[f"b{i}"]
            if i < n - 1:
                answer[answer <= 0] = 0
            else:
                answer = 1 / (1 + np.exp(-1 * answer))
        answer = answer[0, 0]
        if return_bool:
            answer = 1 if answer >= 0.25 else 0
        return answer

    def find_candidates(self, state):
        """
        Finds candidate ingredients to remove. Without these inputs, the `self.value_model`
        still predicts `grow.` State results are cached in `self.state_memory` for improved
        performance.

        Inputs
        ------
        state: set(str)
            A set of the present ingredients in the solution.
        
        Return
        -------
        candidates: list(str)
            The candidate ingredients.
        """

        does_grow = {}
        for ingredient in state:
            test_state = self.remove_from_list(state, ingredient)
            test_state = self.ingredients_to_input(test_state)

            key = tuple(test_state.tolist()[0])
            if key in self.state_memory.keys():
                does_grow[ingredient] = self.state_memory[key]
            else:
                # states_to_test.append(test_state)
                # test_order.append(ingredient)
                # keys[ingredient] = key
                result = self.evaluate_value_model(test_state)
                self.state_memory[key] = result
                does_grow[ingredient] = result

        # if states_to_test:
        #     states_to_test = np.concatenate(states_to_test)
        #     growth_results = self.value_model.predict_class(states_to_test).reshape(
        #         (1, -1)
        #     )[0]
        #     for idx, ingredient in enumerate(test_order):
        #         result = growth_results[idx]
        #         does_grow[ingredient] = result
        #         key = keys[ingredient]
        #         self.state_memory[key] = result
        print("find_candidates results", does_grow)
        candidates = [k for k, v in does_grow.items() if v >= self.growth_cutoff]
        return candidates

    # def informed_trajectory(self, state):
    #     """
    #     Calculates the trajectory of a given state.

    #     State is a List(ingredients)
    #     """
    #     trajectory_state = copy.copy(state)
    #     while len(trajectory_state) > 0:
    #         print("\nSTARTING TRAJECTORY FOR:", trajectory_state)
    #         candidates = self.find_candidates(trajectory_state)
    #         if len(candidates) == 0:
    #             break
    #         choice = random.choice(candidates)
    #         trajectory_state.remove(choice)
    #         print("CHOICE TO REM:", choice, len(trajectory_state))

    #         # print("TS", len(trajectory_state), trajectory_state)
    #         # print(f"removed {choice} from trajectory state", trajectory_state)
    #     # print("Found end state", len(trajectory_state))
    #     return len(trajectory_state)

    def get_state_memory(self):
        return self.state_memory

    def __getstate__(self):
        state = dict(self.__dict__)
        state["value_model_weights"] = None
        return state

    def trajectory(self, state, grow_advantage=None):
        """
        Calculates the trajectory of a given state. A random ingredient is chosen from remaining 
        ingredients until the solution results in a 'no grow' prediction from the `self.value_model`.

        Inputs
        ------
        state: set(str)
            A set of the present ingredients in the solution.
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
        print("")
        trajectory_state = copy.copy(state)
        # trajectory_state = set(self.find_candidates(state))
        length = len(trajectory_state)
        # print(state)
        # print(grow_result)
        prev_grow_result = 0
        while length >= 1:
            # trajectory_state = set(self.find_candidates(trajectory_state))
            # if not len(trajectory_state):
            #     break
            # Set choice probabilities
            temp_state = list(trajectory_state)
            p = np.ones(length)
            if grow_advantage != None and grow_advantage != 1:
                candidates = self.find_candidates(trajectory_state)
                for idx, i in enumerate(temp_state):
                    if i in candidates:
                        p[idx] = grow_advantage
            p /= p.sum()

            # Pick a random ingredient from the current state
            print(len(temp_state), len(p))
            ingredient = np.random.choice(temp_state, p=p)
            test_state = self.remove_from_list(state, ingredient)
            test_state = self.ingredients_to_input(test_state)

            # Cache calculated state values if we haven't seen the state yet,
            # otherwise ask the value model for the prediction
            key = tuple(test_state.tolist()[0])
            if key in self.state_memory.keys():
                grow_result = self.state_memory[key]
            else:
                # print("THIS SHOULD NEVER HAPPEN")
                # grow = self.value_model.model.predict(test_state).reshape((1, -1))[0]

                grow_result = self.evaluate_value_model(test_state)
                # grow = self.value_model.predict_class(test_state)
                # print("GROW SELF", grow, grow2)
                # grow = 1 if grow >= 0.25 else 0
                self.state_memory[key] = grow_result

            # Stop if the solution results in no growth, or if all ingredients
            # have been removed.
            # if length == 1:
            #     return prev_grow_result
            print(grow_result, length, ingredient)
            trajectory_state.remove(ingredient)
            # print(trajectory_state)
            length = len(trajectory_state)
            prev_grow_result = grow_advantage

        return prev_grow_result

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
        state: set(str)
            A set of the present ingredients in a solution.
        """

        ingredients = [k for k, v in state.items() if v == 1]
        return set(ingredients)

    def remove_from_list(self, state, to_remove):
        """
        Removes the ingredient `to_remove` from state if it is present.
        
        Inputs
        ------
        state: set(str)
            A set of the present ingredients in the solution.
        
        Return
        -------
        state: set(str)
            A set of the remaining ingredients in the solution.
        """

        state = state - set([to_remove])
        return state

    def ingredients_to_input(self, state):
        """
        Converts to numpy a state into the input form for `self.value_model`.
        
        Inputs
        ------
        state: set(str)
            A set of the present ingredients in the solution.
        
        Return
        -------
        inputs: np.array(int)
            A boolean array representation of the state.
        """

        inputs = np.array(self.all_ingredients)
        inputs = np.isin(inputs, list(state)).astype(float).reshape((1, -1))
        return inputs

    def perform_rollout(
        self,
        limit,
        available_actions=None,
        grow_advantage=None,
        log_graph=True,
        use_multiprocessing=False,
    ):
        """
        Performs an MCTS Rollout Simulation for the solution `self.current_state`. The 
        trajectories for each remaining ingredient are averaged over `limit` times. The
        ingredient with the lowest predicted score (equivalent to the cardinality of the 
        solution) is returned.
        
        Inputs
        ------
        limit: int
            The number of times a trajectory will be calculated for each ingredient.
        grow_advantage: int or None
            Parameter used to set the relative probability of choosing an ingredient that is
            predicted to grow.
        log_graph: boolean
            Flag to enable the graphical output.
        
        Return
        -------
        best_action: str
            The action that minimizes the predicted cardinality of the final solution.

        Outputs
        ------
        'rollout_result.png': PNG image
            Graph of each ingredient's average trajectories over time when `log_graph` 
            is set to `True`.
        """

        if not available_actions:
            available_actions = self.find_candidates(self.current_state)
        rewards = {}
        print("Available actions", available_actions)

        if log_graph:
            all_results = np.empty((len(available_actions), limit))

        if not use_multiprocessing:
            t1 = tqdm(available_actions, desc="Exploring Actions", leave=True)
            # t2 = tqdm(total=limit, desc="Calculating Trajectory", leave=True)
            for i, action in enumerate(t1):
                t1.set_description(f"Exploring ({action})")
                t1.refresh()  # to show immediately the update
                test_state = self.remove_from_list(self.current_state, action)

                results = np.empty(limit)
                results.fill(np.nan)
                # t2.reset()
                for j in range(limit):
                    results[j] = self.trajectory(
                        test_state, grow_advantage=grow_advantage
                    )

                    intermediate_result = np.nanmean(results) if j is not 0 else 0
                    # t2.set_description(
                    #     f"Calculating Trajectory ({round(intermediate_result, 3)})"
                    # )
                    # t2.update()  # to show immediately the update
                    if log_graph:
                        all_results[i, j] = intermediate_result
                rewards[action] = np.mean(results)

            # t2.close()
            t1.close()
        else:

            def _rollout_multi_helper(action, limit, grow_advantage):
                np.random.seed()
                test_state = self.remove_from_list(self.current_state, action)
                results = np.empty(limit)
                results.fill(np.nan)
                with trange(limit, desc="Calculating Trajectory", leave=True) as t2:
                    for j in t2:
                        intermediate_result = np.nanmean(results) if j is not 0 else 0
                        t2.set_description(
                            f"Calculating Trajectory ({round(intermediate_result, 3)})"
                        )
                        t2.update()  # to show immediately the update
                        results[j] = self.trajectory(
                            test_state, grow_advantage=grow_advantage
                        )
                results = np.mean(results)
                # print(len(self.get_state_memory()))
                return results

            pool = mp.Pool(mp.cpu_count() - 1)
            rewards = pool.starmap(
                _rollout_multi_helper,
                zip(
                    available_actions,
                    [limit] * len(available_actions),
                    [grow_advantage] * len(available_actions),
                ),
            )
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

        rewards = {k: v for k, v in sorted(rewards.items(), key=lambda x: x[1])}
        print("\nCalculated Rewards:")
        for k, v in rewards.items():
            print(f"{k} ->\t{v}")

        best_action = max(rewards.items(), key=operator.itemgetter(1))[0]
        return best_action

    # def simulate(self, limit):
    #     best_action = rollout.get_best_action(limit=limit, grow_advantage=1.5)
    #     print("\Best Action:", best_action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run neural_pretrain.py")
    parser.add_argument(
        "-g", "--gpu", type=int, default=0, help="Choose GPU (0 or 1).",
    )
    args = parser.parse_args()

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
        mcts = MCTS(
            # value_model_dir="data/neuralpy_optimization_expts/052220-sparcity-3/working_model",
            value_model_weights_dir="data/neuralpy_optimization_expts/052220-sparcity-3/working_model/weights.npz",
            current_state=starting_state,
        )
        # dill.detect.trace(True)
        # dill.detect.errors(mcts)
        # import pprint

        # pprint.pprint(dill.detect.errors(mcts, depth=1))
        # print("\nStarting State:", starting_state)
        # rollout.simulate(1000)
        best_action = mcts.perform_rollout(
            available_actions=None, limit=1000, grow_advantage=1
        )
