import argparse
import collections
import copy
import csv
from dataclasses import dataclass
import datetime
import json
import logging
import math
import operator
import os
import pickle
import pprint
import uuid

import multiprocess as mp
import numpy as np
import pandas as pd
import scipy.stats

# Suppress Tensorflow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from termcolor import colored
from tqdm import tqdm

import dnf
import neural_pretrain as neural
import mcts
import spsa
import utils
from utils import decoratortimer

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
    choices=(0, 1),
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

pp = pprint.PrettyPrinter(indent=2)


class AgentController(object):
    def __init__(
        self,
        experiment_path,
        ingredients,
        growth_model_dir,
        agents=[],
        oracle_model_dir=None,
        simulation_data_path=None,
        simulation_rule=None,
        shared_history=None,
        reinforce_params=None,
        growth_threshold=0.25,
        seed=None,
    ):
        self.experiment_path = experiment_path
        self.experiment_cycle = 0
        self.policy_iteration = 0
        self.agents = agents  # list of Agent()
        self.finished_agents = []
        self.ingredients = ingredients  # list of str for each possible ingredient
        self.growth_model = neural.PredictNet.from_save(growth_model_dir)
        self.growth_model_dir = growth_model_dir
        self.shared_history = pd.DataFrame()
        self.growth_threshold = growth_threshold
        self.reinforce_params = reinforce_params

        self.oracle_model = None
        self.oracle_model_dir = oracle_model_dir
        if oracle_model_dir:
            self.oracle_model = neural.PredictNet.from_save(oracle_model_dir)

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
            print("\nUsing simulation:", simulation_rule)
            self.simulation_rule = simulation_rule

        self.episode_history = {}

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
        state["oracle_model"] = None
        return state

    def save_state(self):
        folder_path = os.path.join(
            self.experiment_path,
            f"policy_iteration_{self.policy_iteration}",
            f"cycle_{self.experiment_cycle}",
        )
        agent_controller_path = os.path.join(folder_path, "agent_controller.pkl")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.growth_model.set_save_path(folder_path)
        self.growth_model.save()
        with open(agent_controller_path, "wb") as f:
            pickle.dump(self, f)

    @decoratortimer(2)
    def log_episodes(self):
        folder_path = os.path.join(
            self.experiment_path,
            f"policy_iteration_{self.policy_iteration}",
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for uuid, trajectory in self.episode_history.items():
            key = str(uuid)
            log = {
                "actions": [],
                "states": [],
                "gradients": [],
                "rewards": [],
                "log_probs": [],
                "policy_scores": [],
                "skip_probabilities": [],
            }

            for i in trajectory:
                log["actions"].append(i["action"])
                log["states"].append(i["state"].to_log())
                log["gradients"].append(i["calculated_gradients"].tolist())
                log["rewards"].append(i["reward"])
                log["log_probs"].append(i["log_prob"])
                log["policy_scores"].append(i["policy_scores"])
                log["skip_probabilities"].append(i["skip_probability"].tolist())

            log_path = os.path.join(folder_path, f"episode_log-agent({key}).json")
            with open(log_path, "w") as f:
                json.dump(log, f, indent=4)

    @decoratortimer(2)
    def log_summary(
        self,
        new_policy,
        old_policy,
        agent_media_states,
        agent_cardinalities,
    ):

        folder_path = os.path.join(
            self.experiment_path,
            f"policy_iteration_{self.policy_iteration}",
        )
        summary_path = os.path.join(folder_path, "summary_info.json")

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        policy_delta = {k: v - old_policy[k] for k, v in new_policy.items()}
        data = {
            "new_policy": new_policy,
            "policy_delta": policy_delta,
            "media_states": agent_media_states,
            "media_cardinalities": agent_cardinalities,
        }

        with open(summary_path, "w") as f:
            json.dump(data, f, indent=4)

    @decoratortimer(2)
    def save_eval_summary(self, rewards_data):
        folder_path = os.path.join(
            self.experiment_path,
            f"policy_eval_{self.policy_iteration}",
        )
        summary_path = os.path.join(folder_path, "summary_info_eval.json")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        with open(summary_path, "w") as f:
            json.dump(rewards_data, f, indent=4)

    def update_state(self):
        folder_path = os.path.join(
            self.experiment_path,
            f"policy_iteration_{self.policy_iteration}",
            f"cycle_{self.experiment_cycle}",
        )
        # loaded_agent_controller = AgentController.load_state(folder_path)
        # self.__dict__.update(loaded_agent_controller.__dict__)

    @classmethod
    def load_state(cls, iteration_folder_path):
        LOG.info(f"Loading from {iteration_folder_path}")
        agent_controller_path = os.path.join(
            iteration_folder_path, "agent_controller.pkl"
        )
        with open(agent_controller_path, "rb") as f:
            agent_controller = pickle.load(f)
        agent_controller.growth_model = neural.PredictNet.from_save(
            iteration_folder_path
        )

        for agent in agent_controller.agents:
            agent.controller = agent_controller

        return agent_controller

    @decoratortimer(2)
    def retrain_growth_model(self):
        """
        Trains new value model using self.shared_history data
        """
        x, y = (self.shared_history.values[:, :-1], self.shared_history.values[:, -1])
        print(f"Training with {self.shared_history.shape} data...")
        # TODO: retrain from scratch
        self.growth_model = self.growth_model.get_reset_clone()
        self.growth_model.train(x, y)

    @decoratortimer(2)
    def simulate_random_initial_data(
        self,
        n=None,
        supplemental_data_path=None,
        use_oracle=True,
        use_rule=False,
    ):
        LOG.info("Setting initial data")

        try:
            if n is None and supplemental_data_path is None:
                raise Exception("Need one input.")
        except Exception as e:
            LOG.error("simulate_random_initial_data:" + str(e))
        else:
            accum = []
            if n is not None:
                if use_oracle:
                    random_inputs = self.np_state.choice(
                        [0, 1], size=(n, len(self.ingredients))
                    )
                    simulation_data = pd.DataFrame(
                        random_inputs,
                        columns=self.ingredients,
                    )
                    simulation_data["grow"] = self.oracle_model.predict_probability(
                        random_inputs
                    )
                elif use_rule:
                    random_inputs = self.np_state.choice(
                        [0, 1], size=(n, len(self.ingredients))
                    )
                    simulation_data = pd.DataFrame(
                        random_inputs,
                        columns=self.ingredients,
                    )
                    simulation_data["grow"] = self.simulation_rule.evaluate(
                        random_inputs
                    )
                else:
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

    @decoratortimer(2)
    def get_simulation_growth(self, media_state):
        """
        Get growth result from by passing in state dict from difference sources in this order:
        1. oracle NN
        2. DNF simulation rule
        3. Simulation data dictionary

        """
        try:
            if isinstance(media_state, dict):
                if set(media_state.keys()) != set(self.ingredients):
                    raise Exception("Media_state does not include all ingredients.")
                input_data = np.array([tuple(media_state.values())])
            elif isinstance(media_state, tuple):
                if len(media_state) != len(self.ingredients):
                    raise Exception(
                        "Media_state length must match number of all ingredients."
                    )
                input_data = np.array([media_state])

            elif isinstance(media_state, np.ndarray):
                if media_state.ndim == 1:
                    input_data = media_state.reshape((1, -1))
                elif media_state.ndim == 2:
                    input_data = media_state
                else:
                    raise Exception("Must be a 1D or 2D np.ndarray.")
            else:
                raise Exception("Must pass in np.ndarray, tuple or dict.")

            if self.oracle_model is not None:
                results = (
                    self.oracle_model.predict_probability(input_data).flatten().tolist()
                )

            elif self.simulation_rule is not None:
                results = self.simulation_rule.evaluate(input_data).tolist()
            elif self.simulation_data_dict is not None:
                results = [
                    self.simulation_data_dict.get(tuple(d.tolist()), None)
                    for d in input_data
                ]
            else:
                raise Exception(
                    "Could not get simulation growth result: Need either neural oracle, simulation data, or simulation rule."
                )

        except Exception as e:
            LOG.error("get_simulation_growth:" + str(e))
        else:
            return results

    @decoratortimer(2)
    def update_policy_reinforce(
        self,
        policy,
        agents_episodes,
        n_removable_ingredients,
        learning_rate=0.10,
        gamma=0.9924506,
        episilon=1e-5,
        clip_lambda=True,
        use_baseline=True,
    ):
        """

        gamma: discount factor

        """

        for param, value in self.reinforce_params.items():
            if param == "learning_rate":
                learning_rate = value
            elif param == "gamma":
                gamma = value
            elif param == "episilon":
                episilon = value
            elif param == "clip_lambda":
                clip_lambda = value
            elif param == "use_baseline":
                use_baseline = value

        LOG.info("Updating Policy - REINFORCE")
        LOG.debug(f"Current Policy: {policy}")

        policy_updates = []
        policy_loss = []

        n_ingredients = len(self.ingredients)
        for e, trajectory in enumerate(agents_episodes.values()):
            print("trajectory:", e)
            T = len(trajectory)
            gJ = 0  # np.zeros(len(policy))
            rewards = [
                # len(trajectory[i]["action"]) + trajectory[i]["reward"]
                trajectory[i]["reward"]
                for i in range(0, T)
            ]
            n_removable = n_removable_ingredients[e]
            baseline = (
                # n_removable * 2 if use_baseline else 0
                n_removable * (gamma ** (n_removable - 1))
                if use_baseline
                else 0  #
            )

            for t in range(0, T):
                gradient = trajectory[t]["calculated_gradients"]  # g(Log(Policy))
                log_prob = trajectory[t]["log_prob"]

                # compute discounted return with baseline
                all_returns = [
                    rewards[t_i] * (gamma ** (t_i - t)) for t_i in range(t, T)
                ]

                sum_returns = sum(all_returns)
                return_score = sum_returns - baseline

                gJ += return_score * gradient  # vector of param gradients

                print(
                    f"\nt: {t}\tT: {T}\tbaseline: {baseline}\tN_removable: {n_removable_ingredients[e]}"
                )
                print(f"sum: {sum_returns}\tscore: {return_score}")

                print("gradient:", gradient)
                print("gJ after:", gJ)

            # if there are multiple parallel agents, keep track of all updates
            policy_updates.append(learning_rate * gJ)

        policy_updates = np.vstack(policy_updates)
        policy_means = policy_updates.mean(axis=0)  # avg parallel agent updates

        LOG.debug("Policy deltas:")
        LOG.debug(policy_means)

        # Apply updates to policy
        new_policy = {}
        all_updates = []
        for i, param_name in enumerate(policy.keys()):
            avg_update = policy_means[i]
            if clip_lambda and param_name == "lambda":
                new_policy["lambda"] = np.clip(policy[param_name] + avg_update, 0, 1)
            else:
                new_policy[param_name] = policy[param_name] + avg_update

        # Normalize all params to sum of 1 ??
        # total = sum(new_policy.values())
        # for k, v in new_policy.items():
        #     new_policy[k] = v / total

        #### Testing not using rollout mu: ####
        # policy["lambda"] = 0.0

        LOG.info(f"New Policy: {new_policy}")

        below_episilon = False  # hard coded
        # if np.abs(np.mean(all_updates)) <= episilon:
        #     below_episilon = True # stop policy iteration when updates go below epsilon

        return new_policy, policy, below_episilon

    def generate_SPSA_experiment(self, media_states, n_grads=2):
        """returns experiments dict {removed_ingredient: ([(p_plus, p_minus)], perturbation vector)...}"""
        experiments = {}
        for removed_ingredient, media_state in media_states.items():
            # Remove any ingredients not available
            n_remaining = sum(media_state)
            s = spsa.SPSA(W=np.ones(n_remaining))
            expts, perturbation = s.gen_spsa_experiments(n_grads)

            # need experiments including zeros for non-present ingredients
            full_expts = []
            for expt_plus, expt_minus in expts:
                expt_plus = iter(expt_plus)
                expt_minus = iter(expt_minus)
                full_expt_plus = []
                full_expt_minus = []
                for is_present in media_state:
                    if is_present == 1:
                        full_expt_plus.append(next(expt_plus))
                        full_expt_minus.append(next(expt_minus))
                    else:
                        full_expt_plus.append(0)
                        full_expt_minus.append(0)

            full_expts.append((full_expt_plus, full_expt_minus))
            experiments[removed_ingredient] = list(zip(full_expts, perturbation))

        return experiments

    def get_spsa_results(self, experiments, is_simulation=True):
        """experiments dict is {removed_ingredient: ([(p_plus, p_minus)], perturbation vector)...}"""
        all_results = {}

        for removed_ingredient, ingredient_experiments in experiments.items():
            perturb_results = []
            for (p_plus, p_minus), perturb_vector in ingredient_experiments:

                # p_plus, p_minus are the inputs perturbed in the +/- direction
                if is_simulation:
                    result_plus = self.get_simulation_growth(
                        tuple(p_plus)
                    )  # Optimize to single call
                    result_minus = self.get_simulation_growth(
                        tuple(p_minus)
                    )  # Optimize to single call
                else:
                    # TODO: read in/format experimental results from CSV
                    continue

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
                LOG.error("compute_spsa_gradients: NO GRADS")
            if len(ingredient_grads) == 1:
                mean_grad = ingredient_grads

            else:
                mean_grad = np.mean(np.vstack(ingredient_grads), axis=0)

            all_gradients[removed_ingredient] = mean_grad

        return all_gradients

    def initialize_agents(self, policy, env, n_agents):

        """
        Initialize `n_agents` with `policy`.
        """
        self.agents = []
        weights_dir = os.path.join(self.growth_model_dir, "weights.npz")
        for i in range(n_agents):
            LOG.debug(f"Creating agent {i}")
            random_seed = utils.numpy_state_int(self.np_state)
            a = Agent(policy, env, weights_dir, self.ingredients, seed=random_seed)
            self.agents.append(a)

    # def run(
    #     self,
    #     n_agents,
    #     starting_media_state=None,
    #     prev_cycle_results=None,
    #     update_policy_on_completion=True,
    #     online_growth_training=False,
    # ):
    #     if prev_cycle_results is None:
    #         self.experiment_cycle = 0
    #         regrade_lambda = self.np_state.uniform(0, 1)
    #         skip_beta_1 = self.np_state.uniform(0, 1)
    #         skip_beta_2 = self.np_state.uniform(0, 1)
    #         skip_beta_3 = self.np_state.uniform(0, 1)

    #         policy = {
    #             "lambda": regrade_lambda,
    #             "beta_1": skip_beta_1,
    #             "beta_2": skip_beta_2,
    #             "beta_3": skip_beta_3,
    #         }

    #         environment = ExperimentEnvironment()

    #         self.initialize_agents(policy, n_agents)
    #         self.episode_history = {agent.uuid: [] for agent in self.agents}
    #     else:
    #         self.experiment_cycle += 1

    #     (
    #         agent_media_states,
    #         agent_cardinalities,
    #         episodes,
    #         final_scores,
    #     ) = self.perform_round(online_growth_training)

    #     LOG.info(colored(f"Final Media States:", "green"))
    #     for m in agent_media_states:
    #         LOG.info(get_indent(1) + str(m))

    #     LOG.info(
    #         colored(f"Final Media Cardinalities:", "green") + str(agent_cardinalities)
    #     )

    #     if episodes:
    #         for uuid, eps in episodes.items():
    #             self.episode_history[uuid].append(eps)

    #     if len(episodes) == 0 and update_policy_on_completion:
    #         # POLICY ITERATION
    #         old_policy = policy
    #         new_policy, below_episilon = self.update_policy_reinforce(
    #             policy, agent_cardinalities, self.episode_history
    #         )

    #         if below_episilon:
    #             # Stop iterating policy if policy deltas are below threshold
    #             return

    #         self.agents = copy.deepcopy(self.agents)

    #         for agent in self.agents:
    #             agent.update_policy(new_policy)
    #         self.retrain_growth_model()
    #         self.log_summary(new_policy, old_policy)
    #         self.policy_iteration += 1
    #     self.save_state()

    def run_simulation(
        self,
        n_agents,
        n_policy_iterations,
        starting_media_state,
        training=True,
        eval_every=None,
        n_evals=5,
    ):

        try:
            if self.shared_history is None:
                raise Exception("You need to add input data to data shared_history")
            if (
                self.simulation_data_dict is None
                and self.simulation_rule is None
                and self.oracle_model is None
            ):
                raise Exception(
                    "You need to set the oracle, simulation data, or simulation rule"
                )
        except Exception as e:
            LOG.error("simulate:" + str(e))
        else:

            # regrade_lambda = 0.0
            regrade_lambda = self.np_state.uniform(0, 1)
            skip_beta_1 = self.np_state.uniform(0, 1)
            skip_beta_2 = self.np_state.uniform(0, 1)
            skip_beta_3 = self.np_state.uniform(0, 1)

            policy = {
                "lambda": regrade_lambda,
                "beta_1": skip_beta_1,
                "beta_2": skip_beta_2,
                "beta_3": skip_beta_3,
            }

            env = ExperimentEnvironment(starting_media_state)
            self.initialize_agents(policy, env, n_agents)

            policy_history = [policy]
            final_media_states = {}
            n_removable_ingredients = [0] * n_agents
            for policy_i in range(n_policy_iterations):
                LOG.info(
                    colored(
                        f"################# STARTING POLICY ROUND {policy_i} #################",
                        "white",
                        "on_blue",
                        attrs=["bold"],
                    )
                )

                self.policy_iteration = policy_i

                self.perform_experiment_sims(n_stages=1)

                first_iteration = True
                self.episode_history = {str(a.uuid): [] for a in self.agents}
                while len(self.agents):
                    if not first_iteration:
                        self.update_state()
                        self.experiment_cycle += 1
                    else:
                        self.experiment_cycle = 0

                    (
                        agent_media_states,
                        agent_cardinalities,
                        agent_episodes,
                    ) = self.perform_round()

                    if agent_episodes:
                        for uuid, ep in agent_episodes.items():
                            self.episode_history[str(uuid)].append(ep)
                    first_iteration = False
                    # self.save_state()

                LOG.info(colored(f"Final Media States:", "green"))
                LOG.info(get_indent(1) + str(pp.pprint(agent_media_states)))

                LOG.info(
                    colored(f"Final Media Cardinalities:", "green")
                    + str(agent_cardinalities)
                )

                n_removable_ingredients = [
                    max(len(self.ingredients) - n, n_old)
                    for n, n_old in zip(agent_cardinalities, n_removable_ingredients)
                ]

                # POLICY ITERATION if training
                if training is True:

                    policy, old_policy, below_episilon = self.update_policy_reinforce(
                        policy, self.episode_history, n_removable_ingredients
                    )

                    self.retrain_growth_model()
                    self.log_summary(
                        policy,
                        old_policy,
                        agent_media_states,
                        agent_cardinalities,
                    )
                    self.log_episodes()
                    policy_history.append(policy)

                self.reset_agents(policy)

                # Perform 'n_evals' evaluations using policy every 'eval_every' policy updates
                if eval_every is not None and policy_i % eval_every == 0:
                    # TODO: Change seed for each trial
                    all_rewards = {}
                    for eval_n in range(n_evals):
                        self.perform_experiment_sims(n_stages=1, is_evaluation=True)
                        rewards = []
                        while len(self.agents):
                            _, agent_cardinalities, _ = self.perform_round(
                                evaluation_n=eval_n
                            )
                            rewards += agent_cardinalities

                        rewards = [len(self.ingredients) - i for i in rewards]
                        all_rewards[f"eval_{eval_n}"] = rewards
                        self.reset_agents(policy)
                    self.save_eval_summary(all_rewards)

                print("self.agents")
                for a in self.agents:
                    print(a.env.get_current_media(as_binary=True))

                if below_episilon:
                    # Stop iterating policy if policy deltas are below threshold
                    return

                # self.simulate_random_initial_data(
                #     n=2500,
                #     supplemental_data_path="data/iSMU-test/initial_data/train_set_L1OL2O.csv",
                # )

                final_media_states[policy_i] = agent_media_states

            LOG.info(f"All Policies: {policy_history}")
            return final_media_states

    def reset_agents(self, policy):
        for a in self.finished_agents:
            a.reset()
            a.set_policy(policy)
            self.agents.append(a)

        self.finished_agents.clear()

    def perform_round(self, evaluation_n=None):
        if evaluation_n != None:
            round_title = f"EVALUATION {evaluation_n}"
        else:
            round_title = f"STARTING ROUND {self.experiment_cycle}"

        LOG.info(
            colored(
                f"################# {round_title} #################",
                "white",
                "on_magenta",
                attrs=["bold"],
            )
        )
        # LOG.debug(f"Current History: {self.shared_history}")

        # Get next media prediction
        episode = {}
        for i, agent in enumerate(self.agents):
            LOG.info(colored(f"Simulating Agent #{i}", "cyan", attrs=["bold"]))
            LOG.debug(f"Current State: \n{pp.pprint(agent.env.get_current_media())}")

            # Get next move from agent policy
            (
                best_action,
                policy_scores,
                skip_probability,
                gradient,
                log_prob,
            ) = agent.get_action(n_rollout=100)

            # Take step using best_action
            prev_env = copy.deepcopy(agent.env)
            agent.env.step(best_action)

            episode[agent.uuid] = {
                "action": best_action,
                "state": prev_env,
                "reward": 0,
                "calculated_gradients": gradient,
                "log_prob": log_prob,
                "policy_scores": policy_scores,
                "skip_probability": skip_probability,
            }

            # for name, value in reward.items():
            #     if name not in final_scores:
            #         final_scores[name] = []
            #     final_scores[name].append(value)

            # Move to best state. Don't need to update value net, since this next
            # state is chosen from available_actions, which are a subset of L1O at
            # the current state, which have already been tested.

        agents_done = self.perform_experiment_sims(n_stages=1)

        # Remove any agents that need to be terminated
        current_agents = []
        final_media_states = []
        agent_cardinalities = []
        for agent, is_done in zip(self.agents, agents_done):
            if is_done:
                LOG.info(
                    colored(
                        f"Terminating Agent #{i}! Final media state:",
                        "red",
                        attrs=["bold"],
                    )
                    + str(agent.env.get_current_media())
                )
                episode[agent.uuid]["reward"] = agent.env.get_n_removed()
                final_media_states.append(agent.env.get_current_media())
                agent_cardinalities.append(agent.env.get_cardinality())
                self.finished_agents.append(agent)
            else:
                current_agents.append(agent)

        self.agents = current_agents

        return final_media_states, agent_cardinalities, episode

    @decoratortimer(2)
    def perform_experiment_sims(self, n_stages=1, is_evaluation=False):

        # Determine all L1O experiments that need to be run based on each Agent's current state
        expt_to_removed_ingredient = []
        l1o_experiments = set()
        spsa_experiments = []
        for agent in self.agents:
            l1o = agent.env.get_gf_stage_1_experiments()
            l1o_experiments.update(l1o.values())
            expt_to_removed_ingredient.append(l1o)

            # # Generate SPSA experiments from each l1o
            # spsa_expts = self.generate_SPSA_experiment(l1o)
            # spsa_experiments.append(spsa_expts)
        if len(l1o_experiments) == 0:
            agents_done = [True]
            return agents_done

        l1o_list = list(l1o_experiments)
        l1o_experiment_inputs = np.array(l1o_list)

        # Retrive results from simulation dict or oracle model NN
        l1o_oracle_results = self.get_simulation_growth(l1o_experiment_inputs)
        l1o_oracle_results = dict(zip(l1o_list, l1o_oracle_results))

        # Obtain NN prediction from growth model NN when removing each ingredient
        # if self.oracle_model is not None:
        l1o_predicted_results = (
            self.growth_model.predict_probability(l1o_experiment_inputs)
            .flatten()
            .tolist()
        )
        # elif self.simulation_rule is not None:
        #     l1o_predicted_results = self.simulation_rule.evaluate(
        #         l1o_experiment_inputs
        #     ).tolist()

        l1o_predicted_results = dict(zip(l1o_list, l1o_predicted_results))

        # Use for 2-stage lookahead
        # spsa_results = [
        #     self.get_spsa_results(agent_expt) for agent_expt in spsa_experiments
        # ]
        # # print("spsa_results", spsa_results)
        # spsa_gradients = [
        #     self.compute_spsa_gradients(agent_results)
        #     for agent_results in spsa_results
        # ]
        # # print("spsa_gradients", spsa_gradients)

        # Update agent environments and check if done
        agents_done = []
        for agent, experiment_dict in zip(self.agents, expt_to_removed_ingredient):
            # Available actions are only L1Os that grow
            available_actions = []
            l1o_results = {}
            l1o_predictions = {}
            for ingredient, expt in experiment_dict.items():
                # print("l1o_oracle_results", l1o_oracle_results[expt])
                if l1o_oracle_results[expt] >= self.growth_threshold:
                    available_actions.append(ingredient)

                    l1o_results[ingredient] = l1o_oracle_results[expt]
                    l1o_predictions[ingredient] = l1o_predicted_results[expt]

            done = agent.env.update(available_actions, l1o_results, l1o_predictions)
            agents_done.append(done)
            print(f"Agent {agent.uuid} done: {done}")

        if not is_evaluation:
            # Train value net based on these L1Os
            media_columns = [c for c in self.shared_history.columns if c != "grow"]
            l1o_df = pd.DataFrame(l1o_oracle_results.keys(), columns=media_columns)
            l1o_df["grow"] = l1o_oracle_results.values()

            # LOG.debug(f"Added L1O data: {l1o_df}")
            self.update_history(l1o_df)

        return agents_done


class ExperimentEnvironment:
    """The current environment of the experiment for our RL agent to use.

    media: dict(str -> int)
        The current media configuration, 0 means ingredient not present, 1 means ingredient is present.
    step_progress: float
        Ranged from [0, 1], this represents how close the media is to having all of its ingredients
        removed. Computed as (number of removed ingredients)/(number of media ingredients)
    growth: float
        Ranged from [0, 1], this is the experimental result.
    agreement: float
        This value represents the "agreement" between the local search (gradient following results) and
        the predicted neural network results.

    """

    def __init__(self, starting_media_state, seed=None):
        self._current_media = starting_media_state
        self.available_actions = None

        self.step_progress = None
        self.growth = None
        self.agreement = None

        self.gf_stage_1 = None
        self.gf_stage_2 = None
        self.growth_ods_predicted = None

        self.np_state = utils.seed_numpy_state(seed)

    def __str__(self):
        return f"ExperimentEnvironment: \n\tMedia:\n{self.get_current_media()}"

    def to_log(self):
        compatible_export = copy.deepcopy(self.__dict__)
        del compatible_export["np_state"]
        return compatible_export

    def get_current_media(self, as_binary=False):
        if as_binary:
            return list(self._current_media.values())
        return self._current_media

    def set_current_media(self, new_state):
        self._current_media = new_state

    current_state = property(get_current_media, set_current_media)

    def update(self, available_actions, stage_1_results, predicted_ods):
        self.available_actions = available_actions
        self.gf_stage_1 = stage_1_results
        self.growth_ods_predicted = predicted_ods
        return len(available_actions) <= 0  # is_done

    def set_growth(self, growth):
        self.growth = growth

    def get_n_removed(self):
        return len(self.get_current_media()) - self.get_cardinality()

    def get_cardinality(self):
        return sum(self.get_current_media(as_binary=True))

    def get_removal_progress(self):
        """
        Compute the ingredient removal 'progress' as
        (# currently removed ingredients/# total ingredients)
        """

        progress = self.get_n_removed() / len(self.get_current_media())
        self.step_progress = progress
        return progress

    def calc_agreement(self):
        """
        Calculate the agreement between local search gf_stage_1 results (L1O experiments)
        and the agent neural network predictions using the Spearman Correlation coefficient.
        If the number of ingredients is 1, return the absolute difference as agreement.
        """

        gf_stage_1_values = []
        nn_values = []
        is_equal = True
        for ingredient, l1o_result in self.gf_stage_1.items():
            l1o_prediction = self.growth_ods_predicted[ingredient]
            gf_stage_1_values.append(l1o_result)
            nn_values.append(l1o_prediction)
            if l1o_prediction != l1o_result:
                is_equal = False

        if is_equal:
            agreement = 1
        elif len(gf_stage_1_values) == len(nn_values) == 1:
            agreement = 1 - abs(gf_stage_1_values[0] - nn_values[0])
        else:
            agreement = scipy.stats.spearmanr(gf_stage_1_values, nn_values).correlation

        self.agreement = agreement
        return agreement

    def step(self, ingredient):
        """
        Remove one or more ingredients from the media
        """
        if isinstance(ingredient, str):
            self._current_media[ingredient] = 0
        elif isinstance(ingredient, list) or isinstance(ingredient, tuple):
            for i in ingredient:
                self._current_media[i] = 0

    def get_normalized_gf_scores(self):
        st_one = utils.normalize_dict_values(self.gf_stage_1)
        # st_two = utils.normalize_dict_values(self.gf_stage_2)
        st_two = None
        return st_one, st_two

    @decoratortimer(2)
    def get_gf_stage_1_experiments(self):
        """
        Calculate L10 experiments on current state, returns a dict of
        the removed ingredient str -> experiment state tuple
        """
        # First, generate dict of media states to test
        l1o_experiments = {}
        still_present = [
            i for i, present in self.get_current_media().items() if present
        ]
        print("still_present:", still_present)
        for ingredient in still_present:
            new_expt = copy.deepcopy(self.get_current_media())
            new_expt[ingredient] = 0
            l1o_experiments[ingredient] = tuple(new_expt.values())
        return l1o_experiments


class Agent:
    def __init__(self, policy, env, growth_model_weights_dir, ingredients, seed=None):

        self.uuid = uuid.uuid4()
        self.policy = policy
        self.env = env
        self.growth_model_weights_dir = growth_model_weights_dir
        # self.experiment_history = pd.DataFrame()

        self.original_policy = copy.deepcopy(policy)
        self.original_env = copy.deepcopy(env)

        self.np_state = utils.seed_numpy_state(seed)

        # Initialize most updated MCTS
        rand_seed = utils.numpy_state_int(self.np_state)
        self.tree_search = mcts.MCTS(
            self.growth_model_weights_dir,
            ingredients,
            # state_memory,
            seed=rand_seed,
        )

    def reset(self):
        """
        Resets agents and policies
        """

        print(f"reseting agent {self.uuid}")
        # self.tree_search.save_state_memory()
        self.tree_search.load_value_weights()
        self.policy = copy.deepcopy(self.original_policy)
        self.env = copy.deepcopy(self.original_env)

    def __str__(self):
        return f"Agent({self.uuid}): \n\tPolicy:\n{self.policy}, \n\tEnv:\n{self.env}"

    def softmax_with_tiebreak(self, score_totals):
        """
        Compute softmax with random tiebreak.
        """

        exps = np.exp(list(score_totals.values()))
        softmax_scores = exps / exps.sum()

        max_actions = []
        current_max_score = 0
        for a, score in zip(score_totals.keys(), softmax_scores):
            if score > current_max_score:
                max_actions = [a]
                current_max_score = score
            elif score == current_max_score:
                max_actions.append(a)

        if len(max_actions) == 1:
            return max_actions[0]
        else:
            return self.np_state.choice(max_actions)

    def eval_regrade_policy(self, lamb, rollout_score, gf_score):
        """Evaluate REGRADE policy:
        P(ingredient) = lambda * rollout_mu_score + (1-lamb) * gradient_following_score
        """

        return tf.Variable(
            lamb * rollout_score + (1 - lamb) * gf_score, name="regrade_eval"
        )

    def eval_skip_policy(self, betas, removal_progress, growth, agreement):
        """Evaluate skip policy:
        P(ingredient) = (beta_1*removal_progress) + (beta_2*growth) +(beta_3*agreement)
        """

        values = tf.constant([removal_progress, growth, agreement])
        betas = tf.cast(betas, tf.float32, name="betas")
        y = tf.math.reduce_sum(tf.math.multiply(betas, values))
        # y = tf.math.exp(x)
        # y = tf.math.sigmoid(x)
        return y

    def compute_lambda_grad(self, rollout_score, gf_score):
        """Compute lambda gradient"""

        lamb = tf.Variable(self.policy["lambda"], name="lambda")
        lamb = tf.cast(lamb, tf.float32, name="lambda")

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(lamb)
            # y = self.eval_regrade_policy(lamb, rollout_score, gf_score) # TF can't track vars in this fn?
            y = tf.math.log(
                lamb * rollout_score + (1 - lamb) * gf_score
            )  # TF can't track vars unless we directly call this here use tf.py_function

        dy_dlamb = tape.gradient(y, lamb)
        return np.array([dy_dlamb.numpy()])

    def compute_beta_grad(self, growth, agreement):
        """Compute beta gradient"""

        removal_progress = self.env.get_removal_progress()
        betas = [self.policy["beta_1"], self.policy["beta_2"], self.policy["beta_3"]]
        betas = tf.Variable(betas, name="betas")
        betas = tf.cast(betas, tf.float32, name="betas")

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(betas)
            # y = self.eval_skip_policy(betas, removal_progress, growth, agreement)
            values = tf.constant([removal_progress, growth, agreement])
            y = tf.math.log(
                tf.math.reduce_sum(tf.math.multiply(betas, values))
            )  # Log(Skip Policy)
            # y = tf.math.sigmoid(x)
            # y = tf.math.exp(x)

        grad_betas = tape.gradient(y, betas)
        return grad_betas.numpy()

    def set_policy(self, new):
        if self.policy.keys() != new.keys():
            raise KeyError("Matching keys not found.")
        self.policy = new

    def set_env(self, new):
        self.env = new

    @decoratortimer(2)
    def get_action(
        self,
        n_rollout=100,
        state_memory=None,
        log=False,
    ):
        """
        Picks and action based on the provided sources of information.

        Implement the REGRADE algorithm, calculate skip policy
        """

        # Perform rollout
        rollout_scores = self.run_rollout(n_rollout, return_value="mean")
        if log:
            self.log_rollout_results(rollout_scores)

        print("\nRollout means:")
        pp.pprint(rollout_scores)
        print("\ngrowth_ods_predicted:")
        pp.pprint(self.env.growth_ods_predicted)
        print("\ngf_stage_1:")
        pp.pprint(self.env.gf_stage_1)
        print("\ngf_stage_2:")
        pp.pprint(self.env.gf_stage_2)

        rollout_scores = utils.normalize_dict_values(rollout_scores)
        print("normalized rollout_scores:")
        pp.pprint(rollout_scores)

        if self.env.growth_ods_predicted is not None:
            # Process OD results
            for ingredient, score in self.env.growth_ods_predicted.items():
                pass

        if self.env.gf_stage_1 is not None:
            # Process L1O gradients
            gf_stage_1_scores, _ = self.env.get_normalized_gf_scores()

            print("normalized gf 1:")
            pp.pprint(gf_stage_1_scores)

        if self.env.gf_stage_2 is not None:
            # TODO: process SPSA gradients
            print("USING GRADIENTS:")
            pp.pprint(self.env.gf_stage_2)
            # scores[ingredient]["spsa_gradients"] = policy["spsa_gradients"] * self.process_gradients(spsa_gradients[ingredient])

        # Compute REGRADE policy score
        # Calculate scores for all single ingredient removals
        policy_scores = {}
        for ingredient in self.env.available_actions:
            # compute REGRADE scores for ingredient
            regrade_prob = self.eval_regrade_policy(
                self.policy["lambda"],
                rollout_scores[ingredient],
                gf_stage_1_scores[ingredient],
            ).numpy()

            if np.isnan(regrade_prob):
                regrade_prob = 0

            policy_scores[ingredient] = regrade_prob

        print("########### POLICY SCORES ###########\n")
        policy_sum = sum(policy_scores.values())
        if policy_sum != 0:
            normalized_policy_scores = {
                k: v / policy_sum for k, v in policy_scores.items()
            }
            choice_probs = list(normalized_policy_scores.values())
        else:
            normalized_policy_scores = policy_scores
            choice_probs = np.ones(len(policy_scores)) / len(policy_scores)

        pp.pprint(normalized_policy_scores)
        print()

        # best_action = self.softmax_with_tiebreak(policy_scores)
        best_action = self.np_state.choice(
            self.env.available_actions,
            size=1,
            p=choice_probs,
        )[0]

        roll_score = rollout_scores[best_action]
        gf_score = gf_stage_1_scores[best_action]
        log_prob = np.log(normalized_policy_scores[best_action])
        print("log_prob:", log_prob)

        # compute Skip Probability
        removal_progress = self.env.get_removal_progress()
        print(f"Removal Progress: {removal_progress} - {int(removal_progress*20)}/20")

        betas = tf.Variable(
            [self.policy["beta_1"], self.policy["beta_2"], self.policy["beta_3"]]
        )
        growth_result = self.env.gf_stage_1[best_action]
        self.env.set_growth(growth_result)
        agreement = self.env.calc_agreement()
        print(f"Agreement: {agreement}")

        skip_probability = self.eval_skip_policy(
            betas, removal_progress, growth_result, agreement
        ).numpy()

        if skip_probability >= 0.50:
            LOG.info(
                colored(
                    "Skipping!",
                    "green",
                    attrs=["bold"],
                )
            )
            print(f"SKIP PROBABILITY: {skip_probability}")
            # Pick second ingredient to remove by performing rollout simulations
            # on media after removing first best action from above
            new_state = copy.deepcopy(self.env.get_current_media())
            new_state[best_action] = 0

            new_actions = list(set(self.env.available_actions) - set([best_action]))
            second_rollout_scores = None
            if len(new_actions):
                second_rollout_scores = self.run_rollout(
                    n_rollout,
                    return_value="mean",
                    override_state=new_state,
                    override_actions=new_actions,
                )

                # Filter out NaNs:
                second_rollout_scores = {
                    k: v for k, v in second_rollout_scores.items() if v > 0
                }

            # Only remove second ingredient if there are sufficient successful rollouts
            if second_rollout_scores is not None and len(second_rollout_scores):
                second_rollout_scores = utils.normalize_dict_values(
                    second_rollout_scores
                )
                print("normalized second_rollout_scores:")
                pp.pprint(second_rollout_scores)

                second_action = self.softmax_with_tiebreak(second_rollout_scores)
                best_action = (best_action, second_action)
            else:
                LOG.info(
                    colored(
                        "Could not pick second ingredient. Rollout failed!",
                        "red",
                        attrs=["bold"],
                    )
                )

        gradient_lambda = self.compute_lambda_grad(roll_score, gf_score)
        gradient_betas = self.compute_beta_grad(growth_result, agreement)

        gradient = np.hstack((gradient_lambda, gradient_betas))
        print("get_action gradient:", gradient)

        LOG.info(
            "Chosen ingredient(s) to remove: "
            + colored(f"{best_action}", "green", attrs=["bold"])
        )

        return best_action, policy_scores, skip_probability, gradient, log_prob

    @decoratortimer(2)
    def run_rollout(
        self,
        limit,
        return_value,
        override_state=None,
        override_actions=None,
        state_memory=None,
    ):
        """Run rollout using MCTS"""

        state = (
            override_state
            if override_state is not None
            else self.env.get_current_media()
        )

        available_actions = (
            override_actions
            if override_actions is not None
            else self.env.available_actions
        )

        state = [ingredient for ingredient, present in state.items() if present]

        print(
            f"(run_rollout func) state: {state}\tavailable_actions: {available_actions}"
        )

        # Perform rollout
        rollout_scores = self.tree_search.perform_rollout(
            state=state,
            limit=limit,
            horizon=5,
            available_actions=available_actions,
            log_graph=False,
            threads=16,
        )

        # Process rollout results, use only mean currently
        return_scores = {}
        for ingredient, (mini, maxi, mean) in rollout_scores.items():
            if return_value == "min":
                val = mini
            elif return_value == "mean":
                val = mean
            elif return_value == "max":
                val = maxi
            return_scores[ingredient] = val

        # Use results of state_memory to speed up subsequent rollouts
        # state_memory = search.get_state_memory()
        return return_scores

    def log_rollout_results(self, rollout_scores):
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

        filename = f"avg_rollout_score_reinforce_agent{self.uuid}.csv"
        with open(filename, "a") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    mean_min_rollout_score,
                    mean_max_rollout_score,
                    mean_avg_rollout_score,
                ]
            )


if __name__ == "__main__":
    # Set GPU
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
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

        # Starting state initialzation, everything is in the media
        ingredients = [
            "ala",
            "arg",
            "asn",
            "asp",
            "cys",
            "glu",
            "gln",
            "gly",
            "his",
            "ile",
            "leu",
            "lys",
            "met",
            "phe",
            "pro",
            "ser",
            "thr",
            "trp",
            "tyr",
            "val",
        ]
        starting_media_state = {i: 1 for i in ingredients}
        # starting_media_state = {
        #     "ala": 0,
        #     "arg": 1,
        #     "asn": 0,
        #     "asp": 0,
        #     "cys": 1,
        #     "glu": 0,
        #     "gln": 1,
        #     "gly": 0,
        #     "his": 0,
        #     "ile": 1,
        #     "leu": 1,
        #     "lys": 0,
        #     "met": 0,
        #     "phe": 0,
        #     "pro": 1,
        #     "ser": 0,
        #     "thr": 1,
        #     "trp": 1,
        #     "tyr": 1,
        #     "val": 0,
        # }
        # starting_media_state = {i: np.random.randint(0, 2) for i in ingredients}

        experiment_name = f"experiment-{datetime.datetime.now().isoformat()}"
        experiment_path = os.path.join("data", "agent_logs", experiment_name)
        trials = {
            "reinforce_trial_1": {
                "learning_rate": 0.005,
                "gamma": 0.90,
                # "gamma": 0.9924506,
                "use_baseline": True,
            },
            # "reinforce_trial_2": {
            #     "learning_rate": 0.10,
            #     "gamma": 0.9924506,
            #     "use_baseline": False,
            # },
            # "reinforce_trial_3": {
            #     "learning_rate": 0.15,
            #     "gamma": 0.9924506,
            #     "use_baseline": True,
            # },
            # "reinforce_trial_4": {
            #     "learning_rate": 0.10,
            #     "gamma": 1.0,
            #     "use_baseline": True,
            # },
            # "reinforce_trial_5": {
            #     "learning_rate": 0.10,
            #     "gamma": 0.9924506,
            #     "use_baseline": True,
            #     "clip_lambda": False,
            # },
            # "reinforce_trial_6": {
            #     "learning_rate": 0.10,
            #     "gamma": 1.0,
            #     "use_baseline": True,
            #     "clip_lambda": False,
            # },
            # "reinforce_trial_7": {
            #     "learning_rate": 0.10,
            #     "gamma": 1.0,
            #     "use_baseline": False,
            #     "clip_lambda": False,
            # },
            # "reinforce_trial_8": {
            #     "learning_rate": 0.25,
            #     "gamma": 0.9924506,
            #     "use_baseline": True,
            # },
            # "reinforce_trial_9": {
            #     "learning_rate": 0.10,
            #     "gamma": 0.95,
            #     "use_baseline": True,
            # },
        }

        os.makedirs(experiment_path)
        with open(os.path.join(experiment_path, "exp_trials.json"), "w") as f:
            json.dump(trials, f, indent=4)

        for trial_name, reinforce_params in trials.items():
            print(f"#### Starting Experiment: {trial_name}####")
            print("Experiment Parameters: ")
            print(pp.pprint(reinforce_params))

            #### USING ORACLE SIMULATION
            controller = AgentController(
                experiment_path=os.path.join(experiment_path, trial_name),
                ingredients=ingredients,
                growth_model_dir="models/untrained_growth_NN",
                # simulation_data_path="models/iSMU-test/data_20_extrapolated.csv",
                oracle_model_dir="models/SMU_NN_oracle",
                reinforce_params=reinforce_params
                # seed=0,
            )
            controller.simulate_random_initial_data(
                n=1000,
                supplemental_data_path="models/SMU_NN_oracle/SMU_training_data_L1OL2OL1IL2I.csv",
            )

            #### USING RULE SIMULATION
            # controller = AgentController(
            #     experiment_path=os.path.join(experiment_path, trial_name),
            #     ingredients=ingredients,
            #     growth_model_dir="models/untrained_growth_NN",
            #     simulation_rule=dnf.Rule(
            #         20,
            #         poisson_mu_OR=4,
            #         poisson_mu_AND=15,
            #         ingredient_names=ingredients,
            #     ),
            #     reinforce_params=reinforce_params,
            #     growth_threshold=1,
            #     # seed=0,
            # )
            # controller.simulate_random_initial_data(
            #     n=10, use_oracle=False, use_rule=True
            # )

            #### RUNNING SIMULATION
            controller.retrain_growth_model()
            controller.run_simulation(
                n_agents=1,
                n_policy_iterations=500,
                starting_media_state=copy.deepcopy(starting_media_state),
                training=True,
                eval_every=10,
                n_evals=10,
            )

        # controller.run(
        #     n_agents=1, starting_media_state=starting_media_state, prev_cycle_results=None,
        # )

        # controller = AgentController.load_state("data/agent_state_save_testing/policy_iteration_0/cycle_0")
        # prev_cycle_data_path = "data/agent_state_save_testing/policy_iteration_0/cycle_0/results.csv"
        # controller.run(
        #     n_agents=1,
        #     output_files_dir,
        #     starting_media_state=starting_media_state,
        #     prev_cycle_results=prev_cycle_data_path,
        # )
