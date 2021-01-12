import argparse
import collections
import copy
import csv
from dataclasses import dataclass
import itertools
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
    "-mp",
    "--no_multiprocessing",
    action="store_true",
    default=False,
    help="Don't use multiprocessing.",
)

parser.add_argument(
    "-g", "--gpu", type=int, default=0, choices=(0, 1), help="Choose GPU (0 or 1).",
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
        self.growth_threshold = 0.25

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

    def save_summary(
        self, new_policy, old_policy, agent_media_states, agent_cardinalities
    ):
        print("save_policy_summary")
        folder_path = os.path.join(
            self.experiment_path, f"policy_iteration_{self.policy_iteration}",
        )
        summary_path = os.path.join(folder_path, "summary_info.txt")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print("summary path", summary_path)
        with open(summary_path, "w") as f:
            f.write("NEW POLICY:\n")
            for k, v in new_policy.items():
                f.write(f"\t{k}: {v}\n")
            f.write("OLD POLICY:\n")
            for k, v in old_policy.items():
                f.write(f"\t{k}: {v}\n")

            f.write("MEDIA STATES:\n")
            for m in agent_media_states:
                f.write(f"\t{m}\n")

            f.write("MEDIA CARDINALITIES:\n")
            for c in agent_cardinalities:
                f.write(f"\t{c}\n")

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
        print("Loading from", iteration_folder_path)
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

    def retrain_growth_model(self):
        """
        Trains new value model using self.shared_history data
        """
        x, y = (self.shared_history.values[:, :-1], self.shared_history.values[:, -1])
        # TODO: retrain from scratch
        self.growth_model = self.growth_model.get_reset_clone()
        self.growth_model.train(x, y)

    def simulate_random_initial_data(
        self, n=None, supplemental_data_path=None, use_oracle=True
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
                        random_inputs, columns=self.ingredients,
                    )
                    simulation_data["grow"] = self.oracle_model.predict_probability(
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

    def get_simulation_growth(self, media_state):
        """
        Get growth result from by passing in state dict from difference sources in this order:
        1. oracle NN
        2. DNF simulation rule
        3. Simulation data dictionary

        """
        try:
            if isinstance(media_state, dict):
                if set(media_state.keys()) != set(self.ingredients.keys()):
                    raise Exception("Media_state does not include all ingredients.")
                input_data = tuple(media_state.values())
            elif isinstance(media_state, tuple):
                if len(media_state) != len(self.ingredients.keys()):
                    raise Exception(
                        "Media_state length must match number of all ingredients."
                    )
                input_data = media_state
            else:
                raise Exception("Must pass in tuple or dict.")

            if self.oracle_model is not None:
                result = float(
                    self.oracle_model.predict_probability(np.array([input_data]))
                )
            elif self.simulation_rule is not None:
                result = float(self.simulation_rule.evaluate(np.array(input_data)))
            elif self.simulation_data_dict is not None:
                result = self.simulation_data_dict.get(input_data, None)
            else:
                raise Exception(
                    "Could not get simulation growth result: Need either neural oracle, simulation data, or simulation rule."
                )

        except Exception as e:
            LOG.error("get_simulation_growth:" + str(e))
        else:
            return result

    def update_policy_reinforce(
        self,
        policy,
        final_cardinality,
        episodes,
        learning_rate=0.10,
        discount_factor=1,
        episilon=1e-5,
        clip=False,
    ):
        LOG.info("Updating Policy - REINFORCE")
        LOG.debug(f"Current Policy: {policy}")

        policy_updates = []
        policy_loss = []
        for e, episode in enumerate(episodes.values()):
            print("episode:", e)
            T = len(episode)
            gJ = 0  # np.zeros(len(policy))
            for t in range(0, T):
                gLogPolicy = episode[t]["calculated_gradients"]
                log_prob = episode[t]["log_prob"]
                reward = episode[t]["reward"]

                return_score = sum(
                    [reward * (discount_factor ** (t_i - t)) for t_i in range(t, T)]
                )  # compute discounted return

                print("gLogPolicy", gLogPolicy)
                print("return score:", return_score, "reward", reward)

                print("gJ", gJ)
                gJ += return_score * gLogPolicy  # vector of param gradients
                print("gJ after", gJ)

            policy_updates.append(
                learning_rate * gJ
            )  # if there are multiple parallel agents, keep track of all updates

        policy_updates = np.vstack(policy_updates)
        policy_means = policy_updates.mean(axis=0)  # avg parallel agent updates

        LOG.debug("Policy deltas:")
        LOG.debug(policy_means)

        # Apply updates to policy
        new_policy = {}
        all_updates = []
        for i, param_name in enumerate(policy.keys()):
            avg_update = policy_means[i]
            LOG.debug(get_indent(1) + str(avg_update))
            if clip:
                new_policy[param_name] = np.clip(policy[param_name] + avg_update, 0, 1)
                # Clip only lambda
                # if param_name == "lambda":
                #     new_policy["lambda"] = np.clip(policy[param_name] + avg_update, 0, 1)
            else:
                new_policy[param_name] = policy[param_name] + avg_update

        # Normalize all params to sum of 1 ??
        # total = sum(new_policy.values())
        # for k, v in new_policy.items():
        #     new_policy[k] = v / total

        #### Testing not using rollout mu: ####
        policy["lambda"] = 0.0

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
        # print("exp", experiments)
        all_results = {}
        for removed_ingredient, ingredient_experiments in experiments.items():
            perturb_results = []
            # print(ingredient_experiments)
            for (p_plus, p_minus), perturb_vector in ingredient_experiments:

                # p_plus, p_minus are the inputs perturbed in the +/- direction
                if is_simulation:
                    result_plus = self.get_simulation_growth(tuple(p_plus))
                    result_minus = self.get_simulation_growth(tuple(p_minus))
                else:
                    # TODO: read in/format experimental results from CSV
                    continue
                # print(result_plus, result_minus)
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
            a = Agent(policy, env, weights_dir, seed=random_seed,)
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

    #         self.agents = copy.copy(self.agents)

    #         for agent in self.agents:
    #             agent.update_policy(new_policy)
    #         self.retrain_growth_model()
    #         self.save_summary(new_policy, old_policy)
    #         self.policy_iteration += 1
    #     self.save_state()

    def run_simulation(
        self,
        n_agents,
        n_policy_iterations,
        starting_media_state,
        online_growth_training=False,
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

            regrade_lambda = 0.0
            # regrade_lambda = self.np_state.uniform(0, 1)
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
                self.episode_history = {a.uuid: [] for a in self.agents}
                while len(self.agents):
                    if not first_iteration:
                        self.update_state()
                        self.experiment_cycle += 1
                    else:
                        self.experiment_cycle = 0

                    (
                        agent_media_states,
                        agent_cardinalities,
                        episodes,
                    ) = self.perform_round(online_growth_training)

                    if episodes:
                        for uuid, eps in episodes.items():
                            self.episode_history[uuid].append(eps)
                    first_iteration = False
                    # self.save_state()

                LOG.info(colored(f"Final Media States:", "green"))
                for m in agent_media_states:
                    LOG.info(get_indent(1) + str(m))

                LOG.info(
                    colored(f"Final Media Cardinalities:", "green")
                    + str(agent_cardinalities)
                )

                # POLICY ITERATION
                policy, old_policy, below_episilon = self.update_policy_reinforce(
                    policy, agent_cardinalities, self.episode_history
                )

                for a in self.finished_agents:
                    a.reset()
                    a.set_policy(policy)
                    self.agents.append(a)

                self.finished_agents.clear()
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

                self.retrain_growth_model()
                self.save_summary(
                    policy, old_policy, agent_media_states, agent_cardinalities
                )
                policy_history.append(policy)
                final_media_states[policy_i] = agent_media_states

            LOG.info(f"All Policies: {policy_history}")
            return final_media_states

    def perform_round(
        self, online_growth_training,
    ):
        LOG.info(
            colored(
                f"################# STARTING EVALUATION ROUND {self.experiment_cycle} #################",
                "white",
                "on_magenta",
                attrs=["bold"],
            )
        )
        LOG.debug(f"Current History: {self.shared_history}")

        # if online_growth_training:
        #     self.retrain_growth_model()

        # Get next media prediction
        episodes = {}
        for i, agent in enumerate(self.agents):
            LOG.info(colored(f"Simulating Agent #{i}", "cyan", attrs=["bold"]))
            LOG.debug(f"Current State: {agent.env.get_current_media(as_binary=True)}")

            # Get next move from agent policy
            best_action, policy_scores, gradient, log_prob = agent.get_action(
                n_rollout=100
            )

            # Take step using best_action
            prev_env = agent.env
            agent.env.step(best_action)

            episodes[agent.uuid] = {
                "action": best_action,
                "state": prev_env,
                "reward": 0,
                "calculated_gradients": gradient,
                "log_prob": log_prob,
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
                episodes[agent.uuid]["reward"] = agent.env.get_n_removed()
                final_media_states.append(agent.env.get_current_media(as_binary=True))
                agent_cardinalities.append(agent.env.get_cardinality())
                self.finished_agents.append(agent)
            else:
                current_agents.append(agent)

        self.agents = current_agents

        return final_media_states, agent_cardinalities, episodes

    def perform_experiment_sims(self, n_stages=1):

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

        # Retrive results from simulation dict or oracle model NN
        l1o_oracle_results = {
            expt: self.get_simulation_growth(expt) for expt in l1o_experiments
        }

        # Obtain NN prediction from growth model NN when removing each ingredient
        l1o_predicted_results = {
            expt: float(self.growth_model.predict_probability(np.array([expt]))[0])
            for expt in l1o_experiments
        }

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

        # Get a list of available actions for each agent based on L1O results
        agent_available_actions = []
        agent_l1o_results = []
        agents_done = []
        for agent, experiment_dict in zip(self.agents, expt_to_removed_ingredient):
            # Available actions are only L1Os that grow
            actions = []
            l1o_results = {}
            l1o_predictions = {}
            for ingredient, expt in experiment_dict.items():
                # print("l1o_oracle_results", l1o_oracle_results[expt])
                if l1o_oracle_results[expt] >= self.growth_threshold:
                    actions.append(ingredient)

                    l1o_results[ingredient] = l1o_oracle_results[expt]
                    l1o_predictions[ingredient] = l1o_predicted_results[expt]

            done = agent.env.update(actions, l1o_results, l1o_predictions)
            print(f"Agent {agent.uuid} done: {done}")
            agent_available_actions.append(actions)
            agent_l1o_results.append(l1o_results)
            agents_done.append(done)
        # Train value net based on these L1Os
        media_columns = [c for c in self.shared_history.columns if c != "grow"]
        l1o_df = pd.DataFrame(l1o_oracle_results.keys(), columns=media_columns)
        l1o_df["grow"] = l1o_oracle_results.values()

        LOG.debug(f"Added L1O data: {l1o_df}")
        self.update_history(l1o_df)

        return agents_done
        # return agent_available_actions, agent_l1o_results


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

    def get_current_media(self, as_binary=False):
        if as_binary:
            return list(self._current_media.values())
        return self._current_media

    def set_current_media(self, new_state):
        self._current_media = new_state

    current_state = property(get_current_media, set_current_media)

    def update(self, actions, stage_1_results, predicted_ods):
        self.available_actions = actions
        self.gf_stage_1 = stage_1_results
        self.growth_ods_predicted = predicted_ods

        self.update_agreement()
        return len(actions) <= 0  # is_done

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
        return progress

    def update_agreement(self):
        """
        Calculate the agreement between local search gf_stage_1 results (L1O expts.)
        and the agent neural network predictions.
        """

        agreement = {}
        for ingredient, l1o_result in self.gf_stage_1.items():
            l1o_prediction = self.growth_ods_predicted[ingredient]
            value = 1 - abs(l1o_prediction - l1o_result)
            agreement[ingredient] = value

        self.agreement = utils.normalize_dict_values(agreement)

    def step(self, ingredient):
        self._current_media[ingredient] = 0

    def get_normalized_gf_scores(self):
        st_one = utils.normalize_dict_values(self.gf_stage_1)
        # st_two = utils.normalize_dict_values(self.gf_stage_2)
        st_two = None
        return st_one, st_two

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
        for ingredient in still_present:
            new_expt = copy.copy(self.get_current_media())
            new_expt[ingredient] = 0
            l1o_experiments[ingredient] = tuple(new_expt.values())
        return l1o_experiments


class Agent:
    def __init__(self, policy, env, growth_model_weights_dir, seed=None):

        self.uuid = uuid.uuid4()
        self.policy = policy
        self.env = env
        self.growth_model_weights_dir = growth_model_weights_dir
        # self.experiment_history = pd.DataFrame()

        self.original_policy = copy.deepcopy(policy)
        self.original_env = copy.deepcopy(env)

        self.np_state = utils.seed_numpy_state(seed)

    def reset(self):
        """
        Resets agents and policies
        """

        print(f"reseting agent {self.uuid}")
        self.policy = copy.deepcopy(self.original_policy)
        self.env = copy.deepcopy(self.original_env)

    def __str__():
        print(f"Agent({self.uuid}):")
        print("policy:")
        for k, v in policy.items():
            print(f"\t{k}: {v}")
        print(env)

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
        x = tf.math.reduce_sum(tf.math.multiply(betas, values))
        return tf.math.sigmoid(x)

    def compute_lambda_grad(self, rollout_score, gf_score):
        """Compute lambda gradient"""

        lamb = tf.Variable(self.policy["lambda"], name="lambda")
        lamb = tf.cast(lamb, tf.float32, name="lambda")

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(lamb)
            # y = self.eval_regrade_policy(lamb, rollout_score, gf_score) # TF can't track vars in this fn?
            y = (
                lamb * rollout_score + (1 - lamb) * gf_score
            )  # TF can't track vars unless we directly call this here

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
            x = tf.math.reduce_sum(tf.math.multiply(betas, values))
            y = tf.math.sigmoid(x)

        grad_betas = tape.gradient(y, betas)
        return grad_betas.numpy()

    def set_policy(self, new):
        if self.policy.keys() != new.keys():
            raise KeyError("Matching keys not found.")
        self.policy = new

    def set_env(self, new):
        self.env = new

    def get_action(
        self, n_rollout=100, state_memory=None, log=False,
    ):
        """
        Picks and action based on the provided sources of information.

        Implement the REGRADE algorithm, calculate skip policy
        """

        # Perform rollout
        rollout_predictions = self.run_rollout(n_rollout)
        if log:
            self.log_rollout_results(rollout_predictions)

        print("\nRollout predictions:")
        pp.pprint(rollout_predictions)
        print("\ngrowth_ods_predicted:")
        pp.pprint(self.env.growth_ods_predicted)
        print("\ngf_stage_1:")
        pp.pprint(self.env.gf_stage_1)
        print("\ngf_stage_2:")
        pp.pprint(self.env.gf_stage_2)

        # Process rollout results, use only mean currently
        rollout_scores = {}
        for ingredient, (mini, maxi, mean) in rollout_predictions.items():
            rollout_scores[ingredient] = mean

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

        print("Agreement:")
        pp.pprint(self.env.agreement)

        removal_progress = self.env.get_removal_progress()
        print(f"Removal Progress: {removal_progress}")

        # Compute combined policy score (REGRADE policy and Skip Policy)
        # Calculate scores for all single ingredient removals
        policy_scores = {}
        for ingredient in self.env.available_actions:
            # compute REGRADE scores for ingredient
            regrade_policy_val_1 = self.eval_regrade_policy(
                self.policy["lambda"],
                rollout_scores[ingredient],
                gf_stage_1_scores[ingredient],
            ).numpy()

            # compute Skip scores
            betas = tf.Variable(
                [self.policy["beta_1"], self.policy["beta_2"], self.policy["beta_3"]]
            )
            growth = self.env.growth_ods_predicted[ingredient]
            agreement = self.env.agreement[ingredient]
            skip_policy_val_1 = self.eval_skip_policy(
                betas, removal_progress, growth, agreement
            ).numpy()

            policy_scores[ingredient] = regrade_policy_val_1 * skip_policy_val_1

        # Skip distribution computations
        # removal_permutations = list(
        #     itertools.product(self.env.available_actions, repeat=2)
        # )
        # double_remove_policy_scores = {i: {} for i in self.env.available_actions}
        # for ingredient1, ingredient2 in removal_permutations:
        #     if ingredient1 == ingredient2:
        #         continue
        #     double_remove_policy_scores[ingredient1][ingredient2] = (
        #         policy_scores[ingredient1] * policy_scores[ingredient2]
        #     )
        # print("########### DOUBLE POLICY SCORES ###########\n")
        # pp.pprint(double_remove_policy_scores)

        print("########### POLICY SCORES ###########\n")
        policy_sum = sum(policy_scores.values())
        normalized_policy_scores = {k: v / policy_sum for k, v in policy_scores.items()}
        pp.pprint(normalized_policy_scores)
        print()

        # best_action = self.softmax_with_tiebreak(policy_scores)
        best_action = self.np_state.choice(
            self.env.available_actions,
            size=1,
            p=list(normalized_policy_scores.values()),
        )[0]

        roll_score = rollout_scores[best_action]
        gf_score = gf_stage_1_scores[best_action]

        log_prob = np.log(normalized_policy_scores[best_action])
        print("log_prob:", log_prob)

        gradient_lambda = self.compute_lambda_grad(roll_score, gf_score)
        gradient_betas = self.compute_beta_grad(
            self.env.growth_ods_predicted[ingredient], self.env.agreement[ingredient]
        )

        gradient = np.hstack((gradient_lambda, gradient_betas))
        print("gradient:", gradient)

        LOG.info(f"Chosen ingredient to remove: {best_action}")

        return best_action, policy_scores, gradient, log_prob

    def run_rollout(self, limit, state_memory=None):
        """Run rollout using MCTS"""

        # Initialize most updated MCTS
        rand_seed = utils.numpy_state_int(self.np_state)
        search = mcts.MCTS(
            self.growth_model_weights_dir,
            list(ingredients.keys()),
            state_memory,
            seed=rand_seed,
        )

        # Perform rollout
        rollout_scores = search.perform_rollout(
            state=self.env.get_current_media(),
            limit=limit,
            horizon=5,
            available_actions=self.env.available_actions,
            log_graph=False,
            use_multiprocessing=True,
        )

        # Use results of state_memory to speed up subsequent rollouts
        # state_memory = search.get_state_memory()
        return rollout_scores

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
            "arg_exch": 0.1,
            "asn_exch": 0.1,
            "asp_exch": 0.1,
            "cys_exch": 0.65,
            "glu_exch": 0.1,
            "gln_exch": 0.2,
            "gly_exch": 0.1,
            "his_exch": 0.1,
            "ile_exch": 0.1,
            "leu_exch": 0.1,
            "lys_exch": 0.1,
            "met_exch": 0.1,
            "phe_exch": 0.1,
            "pro_exch": 0.2,
            "ser_exch": 0.1,
            "thr_exch": 0.2,
            "trp_exch": 0.1,
            "tyr_exch": 0.1,
            "val_exch": 0.1,
        }

        # Starting state initialzation, everything is in the media
        starting_media_state = {i: 1 for i in ingredients.keys()}
        # starting_media_state = {
        #     "ala_exch": 0,
        #     "arg_exch": 1,
        #     "asn_exch": 0,
        #     "asp_exch": 0,
        #     "cys_exch": 1,
        #     "glu_exch": 0,
        #     "gln_exch": 1,
        #     "gly_exch": 0,
        #     "his_exch": 0,
        #     "ile_exch": 1,
        #     "leu_exch": 1,
        #     "lys_exch": 0,
        #     "met_exch": 0,
        #     "phe_exch": 0,
        #     "pro_exch": 1,
        #     "ser_exch": 0,
        #     "thr_exch": 1,
        #     "trp_exch": 1,
        #     "tyr_exch": 1,
        #     "val_exch": 0,
        # }
        # starting_media_state = {i: np.random.randint(0, 2) for i in ingredients}

        controller = AgentController(
            experiment_path="data/agent_state_save_fixed_reinforce4",
            ingredients=ingredients,
            growth_model_dir="models/untrained_growth_NN",
            # simulation_data_path="models/iSMU-test/data_20_extrapolated.csv",
            oracle_model_dir="models/SMU_NN_oracle",
            # seed=0,
        )
        controller.simulate_random_initial_data(
            n=1000,
            supplemental_data_path="models/SMU_NN_oracle/SMU_training_data_L1OL2OL1IL2I.csv",
        )
        controller.retrain_growth_model()
        controller.run_simulation(
            n_agents=1,
            n_policy_iterations=100,
            starting_media_state=starting_media_state,
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

        reagent_name_mapping = {
            "ala_exch": "dl_alanine",
            "arg_exch": "l_arginine",
            "asn_exch": "l_asparagine",
            "asp_exch": "l_aspartic_acid",
            "cys_exch": "l_cysteine",
            "glu_exch": "l_glutamic_acid",
            "gln_exch": "l_glutamine",
            "gly_exch": "glycine",
            "his_exch": "l_histidine",
            "ile_exch": "l_isoleucine",
            "leu_exch": "l_leucine",
            "lys_exch": "l_lysine",
            "met_exch": "l_methionine",
            "phe_exch": "l_phenylalanine",
            "pro_exch": "prolines",
            "ser_exch": "l_serine",
            "thr_exch": "l_threonine",
            "trp_exch": "l_tryptophan",
            "tyr_exch": "l_tyrosine",
            "val_exch": "l_valine",
        }
