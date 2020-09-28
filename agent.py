import argparse
import collections
import copy
import csv
import logging
import math
import operator
import os

import multiprocess as mp
import numpy as np
import pandas as pd

# Suppress Tensorflow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from termcolor import colored
from tqdm import tqdm

import neural_pretrain as neural
import mcts
import spsa
import utils

# Logging set up
logger = logging.getLogger(__name__)
LOGLEVELS = (logging.DEBUG, logging.INFO, logging.ERROR)
# Levels, descending
LOGTYPE = collections.namedtuple("LOGTYPE", "debug info error")
LOG = LOGTYPE(logger.debug, logger.info, logger.error)

INDENT = "  "


def get_indent(n):
    return "".join([INDENT for _ in range(n)])


# CLI argument set up
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
    "--no_reinforce",
    action="store_true",
    default=False,
    help="Don't use the REINFORCE algorithm when updating policy.",
)
parser.add_argument(
    "-mp",
    "--no_multiprocessing",
    action="store_true",
    default=False,
    help="Don't use multiprocessing.",
)

parser.add_argument(
    "-g",
    "--gpu",
    type=int,
    default=0,
    choices=range(0, 2),
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


class AgentController(object):
    def __init__(
        self,
        ingredients,
        growth_model_dir,
        agents=[],
        simulation_data_path=None,
        simulation_rule=None,
        shared_history=None,
        seed=None,
    ):
        self.agents = agents  # list of Agent()
        self.ingredients = ingredients  # list of str for each possible ingredient
        self.growth_model = neural.PredictNet.from_save(growth_model_dir)
        self.growth_model_weights_dir = os.path.join(growth_model_dir, "weights.npz")

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

        self.simulation_rule = None
        if simulation_rule:
            self.simulation_rule = simulation_rule

        self.np_state = utils.seed_numpy_state(seed)

    def add_agents(self, agents):
        self.agents.extend(agents)

    def update_history(self, new_data):
        # TODO: check if new data has any collisions
        if isinstance(new_data, np.ndarray):
            new_data = pd.DataFrame(new_data, columns=self.shared_history.columns)

        self.shared_history = pd.concat([self.shared_history, new_data])

    def __getstate__(self):
        """
        Automatically called when pickling this class, to avoid pickling errors 
        when saving growth model object.
        """
        state = dict(self.__dict__)
        state["growth_model"] = None
        return state

    def retrain_growth_model(self):
        """
        Trains new value model using self.shared_history data
        """
        x, y = (self.shared_history.values[:, :-1], self.shared_history.values[:, -1])
        # TODO: retrain from scratch
        self.growth_model = self.growth_model.get_reset_clone()
        self.growth_model.train(x, y)

    def simulate_random_initial_data(self, n=None, supplemental_data_path=None):
        LOG.info("Setting initial data")

        try:
            if n is None and supplemental_data_path is None:
                raise Exception("Need one input.")
        except Exception as e:
            LOG.error("simulate_random_initial_data:" + str(e))
        else:
            accum = []
            if n is not None:
                random_indexes = self.np_state.choice(
                    self.simulation_data.index, size=n, replace=False
                )
                simulation_data = self.simulation_data.loc[random_indexes, :]
                accum.append(simulation_data)
            if supplemental_data_path is not None:
                supplemental_data = pd.read_csv(supplemental_data_path, index_col=None)
                accum.append(supplemental_data)

            # TODO: check for duplicates when adding supplemental data (can use np.unique on np.ndarray)
            assembled_data = pd.concat(accum)

            self.shared_history = assembled_data

    def create_simulation_dict(self, simulation_data_path):
        LOG.info("Creating simulation dict")

        data_dict = {}
        with open(simulation_data_path, newline="") as f:
            f.readline()
            reader = csv.reader(f, delimiter=",")
            for row in tqdm(reader):
                key = tuple([int(i) for i in row[:-1]])
                data_dict[key] = float(row[-1])
        return data_dict

    def get_simulation_growth(self, state):
        """
        Get growth result from simuation dict by passing in state dict
        """
        try:
            if isinstance(state, dict):
                if set(state.keys()) != set(self.ingredients.keys()):
                    raise Exception("State does not include all ingredients.")
                input_data = tuple(state.values())
            elif isinstance(state, tuple):
                if len(state) != len(self.ingredients.keys()):
                    raise Exception(
                        "State length must match number of all ingredients."
                    )
                input_data = state
            else:
                raise Exception("Must pass in tuple or dict.")

            if self.simulation_rule is None and self.simulation_data_dict is None:
                raise Exception("Need either simulation data or simulation rule.")
            elif self.simulation_data_dict is None:
                result = float(self.simulation_rule.evaluate(np.array(input_data)))
            elif input_data not in self.simulation_data_dict:
                # growth_model
                result = float(
                    self.growth_model.predict_probability(np.array([input_data]))
                )
                self.simulation_data_dict[input_data] = result
            elif self.simulation_rule is None:
                result = self.simulation_data_dict.get(input_data, None)
            else:
                raise Exception("Unknown: could not get simulation growth result.")
        except Exception as e:
            LOG.error("get_simulation_growth:" + str(e))
        else:
            return result

    def get_l1o_at_state(self, state):
        """
        Calculate L10 experiments on current state, returns a dict of 
        the removed ingredient str -> experiment state tuple
        """
        # First, generate dict of media states to test
        l1o_experiments = {}
        still_present = [i for i, present in state.items() if present]
        for ingredient in still_present:
            new_expt = copy.copy(state)
            new_expt[ingredient] = 0
            l1o_experiments[ingredient] = tuple(new_expt.values())
        return l1o_experiments

    def update_policy(self, current_policy, results, scores, learning_rate=0.001):
        LOG.info("Updating Policy")
        LOG.debug(f"Current Policy: {current_policy}")

        # all_keys = set()
        # for i in scores.keys()
        #     all_keys.add(i)

        gradients = {i: [] for i in current_policy.keys()}
        for idx, target_score in enumerate(results):
            agent_scores = zip(*scores[idx].values())
            for s in agent_scores:
                total = sum(s)
                diff = total - target_score
                for key, val in current_policy.items():
                    grad = diff * val
                    gradients[key].append(grad)

        LOG.debug(f"Gradients: {gradients}")
        new_policy = {}
        for key, grads in gradients.items():
            sum_grad = sum(grads)
            LOG.debug(f"Grad({key}) sum: {sum_grad}")
            new_policy[key] = current_policy[key] - learning_rate * sum_grad
        LOG.info(f"New Policy: {new_policy}")
        return new_policy

    def update_policy_reinforce(
        self,
        current_policy,
        results,
        episodes,
        learning_rate=0.05,
        discount_factor=1,
        episilon=1e-5,
    ):
        LOG.info("Updating Policy - REINFORCE")
        LOG.debug(f"Current Policy: {current_policy}")

        policy_updates = {theta_name: [] for theta_name in current_policy.keys()}
        for e, episode in enumerate(episodes.values()):
            T = len(episode)
            for t in range(T - 1):
                s, a, r = episode[t]
                G = results[e]  # final cardinality

                policy_score = sum(
                    [
                        theta * s[theta_name]
                        for theta_name, theta in current_policy.items()
                    ]
                )
                for theta_name in current_policy.keys():
                    theta_deriv = (1 / policy_score) * s[theta_name]
                    policy_updates[theta_name].append(
                        learning_rate * (discount_factor ** t) * G * theta_deriv
                    )

        LOG.debug("Policy deltas")
        new_policy = {}
        all_updates = []
        for theta_name in current_policy.keys():
            avg_update = np.mean(policy_updates[theta_name])
            all_updates.append(avg_update)
            LOG.debug(get_indent(1) + str(avg_update))
            new_policy[theta_name] = current_policy[theta_name] + avg_update

        LOG.info(f"New Policy: {new_policy}")

        below_episilon = False
        if np.mean(all_updates) <= episilon:
            below_episilon = True
        return new_policy, below_episilon

    def generate_SPSA_experiment(self, states, n_grads=2):
        """returns experiments dict {removed_ingredient: ([(p_plus, p_minus)], perturbation vector)...}"""
        experiments = {}
        for removed_ingredient, state in states.items():
            # Remove any ingredients not available
            # still_present = [i for i, present in state.items() if present == 1]
            n_remaining = sum(state)
            s = spsa.SPSA(W=np.ones(n_remaining))
            expts, perturbation = s.gen_spsa_experiments(n_grads)

            # need experiments including zeros for non-present ingredients
            full_expts = []
            for expt_plus, expt_minus in expts:
                expt_plus = iter(expt_plus)
                expt_minus = iter(expt_minus)
                full_expt_plus = []
                full_expt_minus = []
                for is_present in state:
                    if is_present == 1:
                        full_expt_plus.append(next(expt_plus))
                        full_expt_minus.append(next(expt_minus))
                    else:
                        full_expt_plus.append(0)
                        full_expt_minus.append(0)
                print(
                    "full_expt_plus, full_expt_minus", full_expt_plus, full_expt_minus
                )
            full_expts.append((full_expt_plus, full_expt_minus))
            # print(f"Perturbed weights {(removed_ingredient)}", expts)
            experiments[removed_ingredient] = list(zip(full_expts, perturbation))
        return experiments

    def get_spsa_results(self, experiments, is_simulation=True):
        """experiments dict is {removed_ingredient: ([(p_plus, p_minus)], perturbation vector)...}"""
        # print("exp", experiments)
        all_results = {}
        for removed_ingredient, ingredient_experiments in experiments.items():
            perturb_results = []
            print(ingredient_experiments)
            for (p_plus, p_minus), perturb_vector in ingredient_experiments:

                print(p_plus, p_minus)
                print(perturb_vector)

                # p_plus, p_minus are the inputs perturbed in the +/- direction
                if is_simulation:
                    result_plus = self.get_simulation_growth(tuple(p_plus))
                    result_minus = self.get_simulation_growth(tuple(p_minus))
                else:
                    # TODO: read in/format experimental results from CSV
                    continue
                print(result_plus, result_minus)
                perturb_results.append(((result_plus, result_minus), perturb_vector))
            all_results[removed_ingredient] = perturb_results

        return all_results

    def compute_spsa_gradients(self, spsa_results):
        all_gradients = {}
        for removed_ingredient, ingredient_results in spsa_results.items():
            ingredient_grads = []
            for (result_plus, result_minus), perturb_vector in ingredient_results:
                gradient = spsa.compute_spsa_gradient(
                    result_plus, result_minus, perturb_vector
                )
                ingredient_grads.append(gradient)
            if len(ingredient_grads) == 0:
                print("ZERO")
            if len(ingredient_grads) == 1:
                mean_grad = ingredient_grads
            else:
                mean_grad = np.mean(np.vstack(ingredient_grads), axis=0)

            # TODO: Ensure dimensionality
            ingredient_grads.append(mean_grad)
            all_gradients[removed_ingredient] = ingredient_grads
        return all_gradients

    def initialize_agents(self, n_agents, starting_state):
        self.agents = []
        # Initialze Agents
        for i in range(n_agents):
            LOG.debug(f"Creating agent {i}")
            random_seed = utils.numpy_state_int(self.np_state)
            a = Agent(
                self,
                self.growth_model_weights_dir,
                starting_state=starting_state,
                seed=random_seed,
            )
            self.agents.append(a)

    def run_experiment(
        self, n_agents, output_files_dir, experimental_results=None, continuing=False
    ):

        self.perform_round()
        pass

    def run_simulation():

        self.perform_round()

    def perform_round():
        pass

    def simulate(
        self,
        n_agents,
        n_policy_iterations,
        starting_state,
        online_growth_training=False,
    ):

        try:
            if self.shared_history is None:
                raise Exception("You need to add input date to data shared_history")
            if self.simulation_data_dict is None and self.simulation_rule is None:
                raise Exception("You need to set the simulation data or rule")
        except Exception as e:
            LOG.error("simulate:" + str(e))
        else:

            # n_policy_iterations = 20
            policy = {
                "min_rollout": self.np_state.uniform(0, 1),
                "max_rollout": self.np_state.uniform(0, 1),
                "mean_rollout": self.np_state.uniform(0, 1),
            }
            policies = [policy]
            policy_round = 0

            # for policy_round in range(n_policy_iterations):
            final_media_states = {}
            for _ in range(n_policy_iterations):
                self.initialize_agents(n_agents, starting_state)

                LOG.info(
                    colored(
                        f"################# STARTING POLICY ROUND {policy_round} #################",
                        "white",
                        "on_blue",
                        attrs=["bold"],
                    )
                )

                evaluation_round_n = 0
                agent_media_states = []
                agent_cardinalities = []
                final_scores = {i: {} for i in range(len(self.agents))}
                episodes = {i: [] for i in range(len(self.agents))}
                while len(self.agents):
                    LOG.info(
                        colored(
                            f"################# STARTING EVALUATION ROUND {evaluation_round_n} #################",
                            "white",
                            "on_magenta",
                            attrs=["bold"],
                        )
                    )
                    LOG.debug(f"Current History: {self.shared_history}")

                    # Determine all L1O expertiments that need to be run based on each Agent's current state
                    expt_to_removed_ingredient = []
                    l1o_experiments = set()
                    spsa_experiments = []
                    for agent in self.agents:
                        l1o = self.get_l1o_at_state(agent.current_state)
                        l1o_experiments.update(l1o.values())
                        expt_to_removed_ingredient.append(l1o)

                        # Generate SPSA experiments from each l1o
                        spsa_expts = self.generate_SPSA_experiment(l1o)
                        spsa_experiments.append(spsa_expts)

                    # Retrive results from simulation dict
                    l1o_results = {
                        expt: self.get_simulation_growth(expt)
                        for expt in l1o_experiments
                    }

                    spsa_results = [
                        self.get_spsa_results(agent_expt)
                        for agent_expt in spsa_experiments
                    ]
                    # print("spsa_results", spsa_results)
                    spsa_gradients = [
                        self.compute_spsa_gradients(agent_results)
                        for agent_results in spsa_results
                    ]
                    # print("spsa_gradients", spsa_gradients)

                    # Get a list of available actions for each agent based on L1O results
                    agent_available_actions = []
                    for agent, experiment_dict in zip(
                        self.agents, expt_to_removed_ingredient
                    ):
                        # Available actions are only L1Os that grow
                        actions = []
                        for ingredient, expt in experiment_dict.items():
                            if l1o_results[expt] >= self.growth_threshold:
                                actions.append(ingredient)
                        agent_available_actions.append(actions)

                    # Train value net based on these L1Os
                    media_columns = [
                        c for c in self.shared_history.columns if c != "grow"
                    ]
                    l1o_df = pd.DataFrame(l1o_results.keys(), columns=media_columns)
                    l1o_df["grow"] = l1o_results.values()

                    LOG.debug(f"Added L1O data: {l1o_df}")
                    self.update_history(l1o_df)
                    # if online_growth_training:
                    #     self.retrain_growth_model()

                    # Get next media prediction
                    agents_to_stop = set()

                    for i, (agent, actions) in enumerate(
                        zip(self.agents, agent_available_actions)
                    ):
                        LOG.info(
                            colored(f"Simulating Agent #{i}", "cyan", attrs=["bold"])
                        )
                        LOG.debug(
                            f"Current State: {agent.get_current_state(as_binary=True)}"
                        )
                        # Set current agent to terminate if there are no available actions
                        if not actions:
                            LOG.info(
                                colored(
                                    f"Terminating Agent #{i}! Final media state:",
                                    "red",
                                    attrs=["bold"],
                                )
                                + str(agent.current_state)
                            )
                            agent_media_states.append(
                                agent.get_current_state(as_binary=True)
                            )
                            agent_cardinalities.append(agent.get_cardinality())
                            agents_to_stop.add(i)
                            continue

                        # Run simulation on Agent's next choice
                        next_media, state, action, reward = agent.get_next_media(
                            available_actions=actions,
                            spsa_gradients=spsa_gradients,
                            policy=policy,
                            n_rollout=100,
                            agent_i=i,
                        )
                        episodes[i].append((state, action, reward))

                        for name, value in reward.items():
                            if name not in final_scores[i]:
                                final_scores[i][name] = []
                            final_scores[i][name].append(value)

                        # Move to best state. Don't need to update value net, since this next
                        # state is chosen from available_actions, which are a subset of L1O at
                        # the current state, which have already been tested.
                        agent.current_state = next_media

                    # Remove any agents that need to be terminated
                    if agents_to_stop:
                        self.agents = [
                            self.agents[j]
                            for j in range(len(self.agents))
                            if j not in agents_to_stop
                        ]

                    evaluation_round_n += 1

                LOG.info(colored(f"Final Media States:", "green"))
                for m in agent_media_states:
                    LOG.info(get_indent(1) + str(m))
                LOG.info(
                    colored(f"Final Media Cardinalities:", "green")
                    + str(agent_cardinalities)
                )

                if args.no_reinforce:
                    policy = self.update_policy(
                        policy, agent_cardinalities, final_scores
                    )
                    below_episilon = False
                else:
                    policy, below_episilon = self.update_policy_reinforce(
                        policy, agent_cardinalities, episodes
                    )

                policies.append(policy)

                if below_episilon:
                    # Stop iterating policy if policy deltas are below threshold
                    break

                final_media_states[policy_round] = agent_media_states

                #### REFACTOR FOR AGENT POLICY ITERATION ####
                self.agents = copy.copy(self.agents)
                # self.simulate_random_initial_data(
                #     n=2500,
                #     supplemental_data_path="data/iSMU-test/initial_data/train_set_L1OL2O.csv",
                # )
                self.retrain_growth_model()
                policy_round += 1

            LOG.info(f"All Policies: {policies}")
            return final_media_states


class Agent:
    def __init__(
        self, controller, growth_model_weights_dir, starting_state=None, seed=None
    ):

        self.controller = controller
        self.growth_model_weights_dir = growth_model_weights_dir
        self.experiment_history = pd.DataFrame()

        self._current_state = None
        if starting_state:
            self._current_state = starting_state

        self.np_state = utils.seed_numpy_state(seed)

    def get_current_state(self, as_binary=False):
        if as_binary:
            return [v for _, v in self._current_state.items()]
        return self._current_state

    def set_current_state(self, new_state):
        self._current_state = new_state

    current_state = property(get_current_state, set_current_state)

    def get_cardinality(self):
        return sum(self.get_current_state(as_binary=True))

    def softmax_with_tiebreak(self, scores):
        total_scores = {ingredient: s["total"] for ingredient, s in scores.items()}

        exps = np.exp(list(total_scores.values()))
        softmax_scores = exps / exps.sum()

        max_actions = []
        current_max_score = 0
        for a, score in zip(total_scores.keys(), softmax_scores):
            if score > current_max_score:
                max_actions = [a]
                current_max_score = score
            elif score == current_max_score:
                max_actions.append(a)

        if len(max_actions) is 1:
            return max_actions[0]
        else:
            return self.np_state.choice(max_actions)

    def pick_best_action(
        self,
        policy=None,
        rollout_predictions=None,
        growth_ods=None,
        spsa_gradients=None,
    ):
        """
        Picks and action based on the provided sources of information and 
        optionally weights them unevenly when making the final choice.

        Inputs must each be a dict(str -> x) where the str is the ingredient name and x 
        is the respective information.

        Implement the REGRADE algorithm.
        """

        ### Finishes calculating the value of our state and chooses a best action
        # based on a policy

        has_rollout = rollout_predictions is not None
        has_od = growth_ods is not None
        has_gradient = spsa_gradients is not None
        n_choices = 3 * int(has_rollout) + int(has_od) + int(has_gradient)

        keys = []
        if rollout_predictions:
            keys.extend(["min_rollout", "max_rollout", "mean_rollout"])
        if growth_ods:
            keys.append("growth_ods")
        if spsa_gradients:
            keys.append("spsa_gradients")

        if policy is None:
            policy = {k: 1 / n_choices for k in keys}
        elif len(policy) != n_choices:
            raise Exception(
                f"ERROR: len(policy) must match number of choices passed in. ({len(policy)}, {n_choices})"
            )
        # elif round(sum(weights), 5) != 1:
        #     raise "ERROR: sum(weights) must equal 1."

        scores = {}
        if has_rollout:
            for ingredient, score in rollout_predictions.items():
                if ingredient not in scores.keys():
                    scores[ingredient] = {"total": 0}
                min_score = policy["min_rollout"] * rollout_predictions[ingredient][0]
                max_score = policy["max_rollout"] * rollout_predictions[ingredient][1]
                mean_score = policy["mean_rollout"] * rollout_predictions[ingredient][2]

                scores[ingredient]["min_rollout"] = min_score
                scores[ingredient]["max_rollout"] = max_score
                scores[ingredient]["mean_rollout"] = mean_score
                scores[ingredient]["total"] += min_score + max_score + mean_score
        if has_od:
            for ingredient, score in growth_ods.items():
                if ingredient not in scores.keys():
                    scores[ingredient] = {"total": 0}
                growth_score = policy["growth_ods"] * growth_ods[ingredient]
                scores[ingredient]["growth_ods"] = growth_score
                scores[ingredient]["total"] += growth_score
        if has_gradient:
            # TODO: process SPSA gradients

            # for gradient in
            # scores[ingredient]["spsa_gradients"] = policy["spsa_gradients"] * self.process_gradients(spsa_gradients[ingredient])
            pass

        # logit_scores = {
        #     ingredient: self.logit_eval(score["total"])
        #     for ingredient, score in scores.items()
        # }

        action = self.softmax_with_tiebreak(scores)

        reward = scores[action]
        reward.pop("total")

        state = {k: None for k in keys}
        if rollout_predictions:
            state["min_rollout"] = rollout_predictions[action][0]
            state["max_rollout"] = rollout_predictions[action][1]
            state["mean_rollout"] = rollout_predictions[action][2]

        LOG.info(f"State: {state}")
        LOG.info(f"Action: {action}")
        LOG.info(f"Reward: {reward}")
        return state, action, reward

    def get_next_media(
        self,
        available_actions,
        spsa_gradients,
        policy,
        n_rollout=100,
        agent_i=0,
        state_memory=None,
        log=False,
    ):
        ### This is part of the value function for our agent

        # Perform rollout
        rollout_scores = self.run_rollout(available_actions, n_rollout)

        min_scores = []
        max_scores = []
        avg_scores = []
        for s in rollout_scores.values():
            min_scores.append(s[0])
            max_scores.append(s[1])
            avg_scores.append(s[2])

        mean_min_rollout_score = np.mean(min_scores)
        mean_max_rollout_score = np.mean(max_scores)
        mean_avg_rollout_score = np.mean(avg_scores)

        if log:
            filename = (
                f"avg_rollout_score_agent{agent_i}.csv"
                if args.no_reinforce
                else f"avg_rollout_score_reinforce_agent{agent_i}.csv"
            )
            with open(filename, "a") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        mean_min_rollout_score,
                        mean_max_rollout_score,
                        mean_avg_rollout_score,
                    ]
                )
        # TODO: Get OD growth results
        growth_ods = None

        # Choose best actions
        state, action, reward = self.pick_best_action(
            policy, rollout_scores, growth_ods, spsa_gradients
        )
        LOG.info(f"Chosen ingredient to remove: {action}")

        # Generate new media dict and return if valid action
        next_media = copy.copy(self.current_state)
        next_media[action] = 0
        return next_media, state, action, reward

    def run_rollout(self, available_actions, limit, state_memory=None):
        # Initialize most updated MCTS
        rand_seed = utils.numpy_state_int(self.np_state)
        search = mcts.MCTS(
            self.growth_model_weights_dir,
            self.current_state,
            state_memory,
            seed=rand_seed,
        )
        # Perform rollout to determine next best media
        rollout_scores = search.perform_rollout(
            state=self.current_state,
            limit=limit,
            horizon=5,
            available_actions=available_actions,
            log_graph=False,
            use_multiprocessing=not args.no_multiprocessing,
        )

        # Use results of state_memory to speed up subsequent rollouts
        # state_memory = search.get_state_memory()
        return rollout_scores


if __name__ == "__main__":
    # import pickle

    # a = pickle.load(
    #     open(
    #         "/home/lab/Documents/github/BacterAI/data/neuralpy_optimization_expts/052220-sparcity-3/no_training/model_params.pkl",
    #         "rb",
    #     )
    # )
    # a["lr"] = 0.004
    # a["n_epochs"] = 10
    # a["n_retrain_epochs"] = 10

    # pickle.dump(
    #     a,
    #     open(
    #         "/home/lab/Documents/github/BacterAI/data/neuralpy_optimization_expts/052220-sparcity-3/no_training/model_params.pkl",
    #         "wb",
    #     ),
    # )

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

        ingredients = {
            "ala_exch": 0.1,
            "gly_exch": 0.1,
            "arg_exch": 0.1,
            "asn_exch": 0.1,
            "asp_exch": 0.1,
            "cys_exch": 0.65,
            "glu_exch": 0.1,
            "gln_exch": 0.2,
            "his_exch": 0.1,
            "ile_exch": 0.1,
            "leu_exch": 0.1,
            "lys_exch": 0.1,
            "met_exch": 0.1,
            "phe_exch": 0.1,
            "ser_exch": 0.1,
            "thr_exch": 0.2,
            "trp_exch": 0.1,
            "tyr_exch": 0.1,
            "val_exch": 0.1,
            "pro_exch": 0.2,
        }

        # ingredients = {
        #     "ala_exch": "0.1 g/mol",
        #     "gly_exch": "0.1 g/mol",
        #     "arg_exch": "0.1 g/mol",
        #     "asn_exch": "0.1 g/mol",
        #     "asp_exch": "0.1 g/mol",
        #     "cys_exch": "0.65 g/mol",
        #     "glu_exch": "0.1 g/mol",
        #     "gln_exch": "0.2 g/mol",
        #     "his_exch": "0.1 g/mol",
        #     "ile_exch": "0.1 g/mol",
        #     "leu_exch": "0.1 g/mol",
        #     "lys_exch": "0.1 g/mol",
        #     "met_exch": "0.1 g/mol",
        #     "phe_exch": "0.1 g/mol",
        #     "ser_exch": "0.1 g/mol",
        #     "thr_exch": "0.2 g/mol",
        #     "trp_exch": "0.1 g/mol",
        #     "tyr_exch": "0.1 g/mol",
        #     "val_exch": "0.1 g/mol",
        #     "pro_exch": "0.2 g/mol",
        # }

        # Starting state initialzation, everything is in the media
        starting_state = {i: 1 for i in ingredients}
        # starting_state = {
        #     "ala_exch": 0,
        #     "gly_exch": 0,
        #     "arg_exch": 1,
        #     "asn_exch": 0,
        #     "asp_exch": 0,
        #     "cys_exch": 1,
        #     "glu_exch": 0,
        #     "gln_exch": 1,
        #     "his_exch": 0,
        #     "ile_exch": 1,
        #     "leu_exch": 1,
        #     "lys_exch": 0,
        #     "met_exch": 1,
        #     "phe_exch": 1,
        #     "ser_exch": 0,
        #     "thr_exch": 1,
        #     "trp_exch": 1,
        #     "tyr_exch": 1,
        #     "val_exch": 0,
        #     "pro_exch": 1,
        # }
        # starting_state = {i: np.random.randint(0, 2) for i in ingredients}

        controller = AgentController(
            ingredients=ingredients,
            growth_model_dir="data/neuralpy_optimization_expts/052220-sparcity-3/no_training",
            simulation_data_path="models/iSMU-test/data_20_extrapolated.csv",
            seed=0,
        )
        controller.simulate_random_initial_data(
            n=2500,
            supplemental_data_path="data/iSMU-test/initial_data/train_set_L1OL2O.csv",
        )
        controller.retrain_growth_model()
        controller.simulate(
            n_agents=1, n_policy_iterations=10, starting_state=starting_state
        )
