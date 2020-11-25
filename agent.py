import argparse
import collections
import copy
import csv
from dataclasses import dataclass
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

pp = pprint.PrettyPrinter(indent=2)


@dataclass
class AgentState:
    """Agent state."""

    media: dict  # str -> int
    step_progress: float  # n_removed/n_media_ingredients
    growth: float
    agreement_rollout_gf: float
    agreement_local_search_NN: float


class AgentController(object):
    def __init__(
        self,
        experiment_path,
        ingredients,
        growth_model_dir,
        agents=[],
        simulation_data_path=None,
        simulation_rule=None,
        shared_history=None,
        seed=None,
    ):
        self.experiment_path = experiment_path
        self.experiment_cycle = 0
        self.policy_iteration = 0
        self.agents = agents  # list of Agent()
        self.ingredients = ingredients  # list of str for each possible ingredient
        self.growth_model = neural.PredictNet.from_save(growth_model_dir)
        self.growth_model_dir = growth_model_dir
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

        self.policy = {}
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

    def save_summary(self, old_policy, agent_media_states, agent_cardinalities):
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
            for k, v in self.policy.items():
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
        loaded_agent_controller = AgentController.load_state(folder_path)
        self.__dict__.update(loaded_agent_controller.__dict__)

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

    def get_simulation_growth(self, media_state):
        """
        Get growth result from simuation dict by passing in state dict
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

    def get_l1o_at_state(self, media_state):
        """
        Calculate L10 experiments on current state, returns a dict of 
        the removed ingredient str -> experiment state tuple
        """
        # First, generate dict of media states to test
        l1o_experiments = {}
        still_present = [i for i, present in media_state.items() if present]
        for ingredient in still_present:
            new_expt = copy.copy(media_state)
            new_expt[ingredient] = 0
            l1o_experiments[ingredient] = tuple(new_expt.values())
        return l1o_experiments

    def update_policy(self, results, scores, learning_rate=0.001):
        LOG.info("Updating Policy")
        LOG.debug(f"Current Policy: {self.policy}")

        # all_keys = set()
        # for i in scores.keys()
        #     all_keys.add(i)

        gradients = {i: [] for i in self.policy.keys()}
        for idx, target_score in enumerate(results):
            agent_scores = zip(*scores[idx].values())
            for s in agent_scores:
                total = sum(s)
                diff = total - target_score
                for key, val in self.policy.items():
                    grad = diff * val
                    gradients[key].append(grad)

        LOG.debug(f"Gradients: {gradients}")
        new_policy = {}
        for key, grads in gradients.items():
            sum_grad = sum(grads)
            LOG.debug(f"Grad({key}) sum: {sum_grad}")
            new_policy[key] = self.policy[key] - learning_rate * sum_grad
        LOG.info(f"New Policy: {new_policy}")
        self.policy = new_policy

    def update_policy_reinforce(
        self,
        final_cardinality,
        episodes,
        learning_rate=0.10,
        discount_factor=1,
        episilon=1e-5,
    ):
        LOG.info("Updating Policy - REINFORCE")
        LOG.debug(f"Current Policy: {self.policy}")
        print("RESULTS", final_cardinality)
        print("EPISODES")
        pp.pprint(episodes)
        policy_updates = {theta_name: [] for theta_name in self.policy.keys()}
        for e, episode in enumerate(episodes.values()):
            print("episode:", e)
            T = len(episode)
            gJ = 0  # np.zeros(len(self.policy))
            for t in range(0, T):
                regrade_scores, gLogPolicy = episode[t]  # ignore this reward value
                # print(f"T:{t}, S:{s}, A:{a}, R:{r}")

                if t == T - 1:
                    final_reward = (
                        len(self.ingredients) - final_cardinality[e]
                    )  # max num of ingred - final cardinality
                else:
                    final_reward = 0

                return_score = sum(
                    [
                        final_reward * (discount_factor ** (t_i - t))
                        for t_i in range(t, T)
                    ]
                )

                print("regrade_scores", regrade_scores)
                print("gLogPolicy", gLogPolicy)
                print("return score:", return_score, "final_reward", final_reward)

                print("gJ", gJ)
                gJ += return_score * gLogPolicy
                print("gJ after", gJ)
                # policy_score = sum(
                #     [theta * s[theta_name] for theta_name, theta in self.policy.items()]
                # )
                # print(f"POLICY SCORE: {policy_score}")

            # for theta_name in self.policy.keys():
            #     theta_deriv = (1 / policy_score) * s[theta_name]
            #     policy_updates[theta_name].append(
            #         learning_rate * (discount_factor) * R * theta_deriv
            #     )
            policy_updates["regrade_lambda"].append(learning_rate * gJ)

        LOG.debug("Policy deltas")
        new_policy = {}
        all_updates = []
        for theta_name in self.policy.keys():
            avg_update = np.mean(policy_updates[theta_name])
            all_updates.append(avg_update)
            LOG.debug(get_indent(1) + str(avg_update))
            new_policy[theta_name] = self.policy[theta_name] + avg_update

        total = sum(new_policy.values())
        for k, v in new_policy.items():
            new_policy[k] = v / total

        LOG.info(f"New Policy: {new_policy}")
        self.policy = new_policy

        below_episilon = False
        if np.mean(all_updates) <= episilon:
            below_episilon = True
        return below_episilon

    def generate_SPSA_experiment(self, media_states, n_grads=2):
        """returns experiments dict {removed_ingredient: ([(p_plus, p_minus)], perturbation vector)...}"""
        experiments = {}
        for removed_ingredient, media_state in media_states.items():
            # Remove any ingredients not available
            # still_present = [i for i, present in state.items() if present == 1]
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
                # print(
                #     "full_expt_plus, full_expt_minus", full_expt_plus, full_expt_minus
                # )
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
            # print(ingredient_experiments)
            for (p_plus, p_minus), perturb_vector in ingredient_experiments:

                # print(p_plus, p_minus)
                # print(perturb_vector)

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

            # TODO: Ensure dimensionality
            # ingredient_grads.append(mean_grad)
            all_gradients[removed_ingredient] = mean_grad
        return all_gradients

    def initialize_agents(self, n_agents, starting_media_state):
        self.agents = []
        # Initialze Agents
        weights_dir = os.path.join(self.growth_model_dir, "weights.npz")
        for i in range(n_agents):
            LOG.debug(f"Creating agent {i}")
            random_seed = utils.numpy_state_int(self.np_state)
            a = Agent(
                self,
                weights_dir,
                starting_media_state=starting_media_state,
                seed=random_seed,
            )
            self.agents.append(a)

    def run(
        self,
        n_agents,
        starting_media_state=None,
        prev_cycle_results=None,
        update_policy_on_completion=True,
        online_growth_training=False,
    ):
        if prev_cycle_results is None:
            self.experiment_cycle = 0
            regrade_lambda = self.np_state.uniform(0.25, 0.75)
            self.policy = {
                "regrade_lambda": regrade_lambda,
            }
            self.initialize_agents(n_agents, starting_media_state)
            self.episode_history = {agent.uuid: [] for agent in self.agents}
        else:
            self.experiment_cycle += 1

        (
            agent_media_states,
            agent_cardinalities,
            episodes,
            final_scores,
        ) = self.perform_round(online_growth_training)

        LOG.info(colored(f"Final Media States:", "green"))
        for m in agent_media_states:
            LOG.info(get_indent(1) + str(m))

        LOG.info(
            colored(f"Final Media Cardinalities:", "green") + str(agent_cardinalities)
        )

        if episodes:
            for uuid, eps in episodes.items():
                self.episode_history[uuid].append(eps)

        if len(episodes) == 0 and update_policy_on_completion:
            # POLICY ITERATION
            old_policy = self.policy
            if args.no_reinforce:
                self.update_policy(agent_cardinalities, final_scores)
                below_episilon = False
            else:
                below_episilon = self.update_policy_reinforce(
                    agent_cardinalities, self.episode_history
                )

            if below_episilon:
                # Stop iterating policy if policy deltas are below threshold
                return

            self.agents = copy.copy(self.agents)

            self.retrain_growth_model()
            self.save_summary(old_policy)
            self.policy_iteration += 1
        self.save_state()

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
            if self.simulation_data_dict is None and self.simulation_rule is None:
                raise Exception("You need to set the simulation data or rule")
        except Exception as e:
            LOG.error("simulate:" + str(e))
        else:
            # policy = {
            #     "min_rollout": self.np_state.uniform(0, 1),
            #     "max_rollout": self.np_state.uniform(0, 1),
            #     "mean_rollout": self.np_state.uniform(0, 1),
            #     "spsa_gradients": self.np_state.uniform(0, 1),
            # }
            regrade_lambda = self.np_state.uniform(0.25, 0.75)
            # self.policy = {
            #     "mean_rollout": regrade_lambda,
            #     "gf_stage_1": 1 - regrade_lambda,
            # }
            self.policy = {
                "regrade_lambda": regrade_lambda,
            }
            policy_iterations = [self.policy]
            final_media_states = {}

            for policy_i in range(n_policy_iterations):
                self.policy_iteration = policy_i
                self.initialize_agents(n_agents, starting_media_state)

                LOG.info(
                    colored(
                        f"################# STARTING POLICY ROUND {policy_i} #################",
                        "white",
                        "on_blue",
                        attrs=["bold"],
                    )
                )

                first_iteration = True
                self.episode_history = {agent.uuid: [] for agent in self.agents}
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
                        final_scores,
                    ) = self.perform_round(online_growth_training)

                    if episodes:
                        for uuid, eps in episodes.items():
                            self.episode_history[uuid].append(eps)
                    first_iteration = False
                    self.save_state()

                LOG.info(colored(f"Final Media States:", "green"))
                for m in agent_media_states:
                    LOG.info(get_indent(1) + str(m))

                LOG.info(
                    colored(f"Final Media Cardinalities:", "green")
                    + str(agent_cardinalities)
                )

                # POLICY ITERATION
                old_policy = self.policy
                if args.no_reinforce:
                    self.update_policy(agent_cardinalities, final_scores)
                    below_episilon = False
                else:
                    below_episilon = self.update_policy_reinforce(
                        agent_cardinalities, self.episode_history
                    )

                if below_episilon:
                    # Stop iterating policy if policy deltas are below threshold
                    return

                #### REFACTOR FOR AGENT POLICY ITERATION ####
                self.agents = copy.copy(self.agents)
                # self.simulate_random_initial_data(
                #     n=2500,
                #     supplemental_data_path="data/iSMU-test/initial_data/train_set_L1OL2O.csv",
                # )

                self.retrain_growth_model()
                self.save_summary(old_policy, agent_media_states, agent_cardinalities)
                policy_iterations.append(self.policy)
                final_media_states[policy_i] = agent_media_states

            LOG.info(f"All Policies: {policy_iterations}")
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
        (
            agent_available_actions,
            agent_l1o_results,
        ) = self.get_gradient_following_experiments(n_stages=1)

        # if online_growth_training:
        #     self.retrain_growth_model()

        # Get next media prediction
        episodes = {}
        final_scores = {}
        agents_to_stop = set()
        agent_media_states = []
        agent_cardinalities = []
        for i, (agent, actions, l1o_results) in enumerate(
            zip(self.agents, agent_available_actions, agent_l1o_results)
        ):
            LOG.info(colored(f"Simulating Agent #{i}", "cyan", attrs=["bold"]))
            LOG.debug(f"Current State: {agent.get_current_state(as_binary=True)}")
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
                agent_media_states.append(agent.get_current_state(as_binary=True))
                agent_cardinalities.append(agent.get_cardinality())
                agents_to_stop.add(i)
                continue

            # Run simulation on Agent's next choice
            next_media, regrade_scores, gradients = agent.get_next_media(
                available_actions=actions,
                gf_stage_1=l1o_results,
                # gf_stage_2=spsa_gradients[i],
                policy=self.policy,
                n_rollout=100,
                agent_i=i,
            )
            episodes[agent.uuid] = (regrade_scores, gradients)

            # for name, value in reward.items():
            #     if name not in final_scores:
            #         final_scores[name] = []
            #     final_scores[name].append(value)

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

        return agent_media_states, agent_cardinalities, episodes, final_scores

    def get_gradient_following_experiments(self, n_stages=1):
        # Determine all L1O expertiments that need to be run based on each Agent's current state
        expt_to_removed_ingredient = []
        l1o_experiments = set()
        spsa_experiments = []
        for agent in self.agents:
            l1o = self.get_l1o_at_state(agent.current_state)
            l1o_experiments.update(l1o.values())
            expt_to_removed_ingredient.append(l1o)

            # # Generate SPSA experiments from each l1o
            # spsa_expts = self.generate_SPSA_experiment(l1o)
            # spsa_experiments.append(spsa_expts)

        # Retrive results from simulation dict
        l1o_raw_results = {
            expt: self.get_simulation_growth(expt) for expt in l1o_experiments
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
        for agent, experiment_dict in zip(self.agents, expt_to_removed_ingredient):
            # Available actions are only L1Os that grow
            actions = []
            l1o_results = {}
            for ingredient, expt in experiment_dict.items():
                if l1o_raw_results[expt] >= self.growth_threshold:
                    actions.append(ingredient)
                    l1o_results[ingredient] = l1o_raw_results[expt]
            agent_available_actions.append(actions)
            agent_l1o_results.append(l1o_results)

        # Train value net based on these L1Os
        media_columns = [c for c in self.shared_history.columns if c != "grow"]
        l1o_df = pd.DataFrame(l1o_raw_results.keys(), columns=media_columns)
        l1o_df["grow"] = l1o_raw_results.values()

        LOG.debug(f"Added L1O data: {l1o_df}")
        self.update_history(l1o_df)

        return agent_available_actions, agent_l1o_results


class Agent:
    def __init__(
        self, controller, growth_model_weights_dir, starting_media_state, seed=None
    ):

        self.controller = controller
        self.uuid = uuid.uuid4()
        self.growth_model_weights_dir = growth_model_weights_dir
        self.experiment_history = pd.DataFrame()

        # self._current_state = None
        # self.starting_state = None
        # if starting_media_state:
        self._current_state = starting_media_state

        self.np_state = utils.seed_numpy_state(seed)

    def __getstate__(self):
        """
        Automatically called when pickling this class, to avoid pickling errors 
        when saving controller object.
        """
        state = dict(self.__dict__)
        state["controller"] = None
        return state

    def get_current_state(self, as_binary=False):
        if as_binary:
            return list(self._current_state.values())
        return self._current_state

    def set_current_state(self, new_state):
        self._current_state = new_state

    current_state = property(get_current_state, set_current_state)

    def get_cardinality(self):
        return sum(self.get_current_state(as_binary=True))

    def softmax_with_tiebreak(self, score_totals):
        # score_totals = {ingredient: s["total"] for ingredient, s in scores.items()}
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

    def pick_best_action(
        self,
        policy,
        available_actions,
        rollout_predictions,
        growth_ods,
        gf_stage_1,
        gf_stage_2,
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

        print("\nRollout predictions:")
        pp.pprint(rollout_predictions)
        print("\ngrowth_ods:")
        pp.pprint(growth_ods)
        print("\ngf_stage_1:")
        pp.pprint(gf_stage_1)
        print("\ngf_stage_2:")
        pp.pprint(gf_stage_2)

        has_od = growth_ods is not None
        has_gf_stage_1 = gf_stage_1 is not None
        has_gf_stage_2 = gf_stage_2 is not None

        n_choices = 3 + int(has_od) + int(has_gf_stage_1) + int(has_gf_stage_2)

        # keys = ["min_rollout", "max_rollout", "mean_rollout"]
        keys = ["mean_rollout"]
        if has_od:
            keys.append("growth_ods")
        if has_gf_stage_1:
            keys.append("gf_stage_1")
        if has_gf_stage_2:
            keys.append("gf_stage_2")

        # if policy is None:
        #     policy = {k: 1 / n_choices for k in keys}
        # elif len(policy) != n_choices:
        #     raise Exception(
        #         f"ERROR: len(policy) must match number of choices passed in. ({len(policy)}, {n_choices})"
        #     )
        # elif round(sum(weights), 5) != 1:
        #     raise "ERROR: sum(weights) must equal 1."

        # Process rollout results
        rollout_scores = {}
        for ingredient, score in rollout_predictions.items():
            mean_rollout_score = score[2]
            rollout_scores[ingredient] = mean_rollout_score

        rollout_scores = utils.normalize_dict_values(rollout_scores)
        print("normalized rollout_scores:")
        pp.pprint(rollout_scores)

        # if ingredient not in scores.keys():
        #     scores[ingredient] = {"total": 0}
        # min_score = policy["min_rollout"] * rollout_predictions[ingredient][0]
        # max_score = policy["max_rollout"] * rollout_predictions[ingredient][1]
        # mean_score = policy["mean_rollout"] * rollout_predictions[ingredient][2]

        # scores[ingredient]["min_rollout"] = min_score
        # scores[ingredient]["max_rollout"] = max_score
        # scores[ingredient]["mean_rollout"] = mean_score
        # scores[ingredient]["total"] += min_score + max_score + mean_score

        if has_od:
            # Process OD results
            for ingredient, score in growth_ods.items():
                pass
                # if ingredient not in scores.keys():
                #     scores[ingredient] = {"total": 0}
                # growth_score = policy["growth_ods"] * growth_ods[ingredient]
                # scores[ingredient]["growth_ods"] = growth_score
                # scores[ingredient]["total"] += growth_score
        if has_gf_stage_1:
            # Process L1O gradients
            # print("L1O results:")
            # for ingredient, gradient in gf_stage_1.items():
            #     print(ingredient)
            #     print("\t", gradient)
            gf_stage_1_scores = utils.normalize_dict_values(gf_stage_1)
            print("normalized gf 1:")
            pp.pprint(gf_stage_1_scores)

        if has_gf_stage_2:
            # TODO: process SPSA gradients
            print("USING GRADIENTS:")
            pp.print(gf_stage_2)
            # scores[ingredient]["spsa_gradients"] = policy["spsa_gradients"] * self.process_gradients(spsa_gradients[ingredient])

        # logit_scores = {
        #     ingredient: self.logit_eval(score["total"])
        #     for ingredient, score in scores.items()
        # }
        # Compute REGRADE score
        regrade_scores = {}
        for ingredient in rollout_predictions.keys():
            rollout_score = rollout_scores[ingredient] * policy["regrade_lambda"]
            gf_stage_1_score = gf_stage_1_scores[ingredient] * (
                1 - policy["regrade_lambda"]
            )
            components = [rollout_score, gf_stage_1_score]
            # regrade_score = sum(regrade_score)
            regrade_scores[ingredient] = sum(components)

        print("########### REGRADE SCORES ###########\n")
        print("total:", sum(regrade_scores.values()))
        pp.pprint(regrade_scores)
        print()
        action = self.softmax_with_tiebreak(regrade_scores)

        # reward = scores[action]
        # reward = {
        #     "mean_rollout": rollout_scores[ingredient] * policy["regrade_lambda"],
        #     "gf_stage_1": gf_stage_1_scores[ingredient] * (1 - policy["regrade_lambda"]),
        # }
        # reward.pop("total")

        # state = {k: 0 for k in keys}
        # state["min_rollout"] = rollout_predictions[action][0]
        # state["max_rollout"] = rollout_predictions[action][1]
        # state["mean_rollout"] = rollout_scores[action]
        # if has_gf_stage_1:
        #     state["gf_stage_1"] = gf_stage_1_scores[action]

        # media: dict  # str -> int
        # step_progress
        # growth: float
        # agreement_rollout_gf: float
        # agreement_local_search_NN: float
        gradient = 1 / (rollout_scores[action] - gf_stage_1_scores[action] + 1)

        n_full_media = len(self.get_current_state())
        step_progress = (
            n_full_media - sum(self.get_current_state(True))
        ) / n_full_media

        rollout_scores = list(rollout_scores.values())
        gf_stage_1_scores = list(gf_stage_1_scores.values())
        agreement_rollout_gf = self.cos_similiarity(rollout_scores, gf_stage_1_scores)

        print("STATE:", step_progress, agreement_rollout_gf)
        # state = AgentState(avail)

        # LOG.info(f"State: {state}")
        # LOG.info(f"Action: {action}")
        # LOG.info(f"Reward: {reward}, ")
        return action, regrade_scores, gradient

    def cos_similiarity(self, r1, r2):
        r1 = np.array(r1)
        r2 = np.array(r2)
        s = r1.dot(r2) / (np.sqrt(r1.dot(r1)) * np.sqrt(r2.dot(r2)))
        return s

    def get_next_media(
        self,
        available_actions,
        gf_stage_1=None,
        gf_stage_2=None,
        growth_ods=None,
        policy=None,
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
        action, regrade_scores, gradients = self.pick_best_action(
            policy,
            available_actions,
            rollout_scores,
            growth_ods,
            gf_stage_1,
            gf_stage_2,
        )
        LOG.info(f"Chosen ingredient to remove: {action}")

        # Generate new media dict and return if valid action
        next_media = copy.copy(self.get_current_state())
        next_media[action] = 0
        return next_media, regrade_scores, gradients

    def run_rollout(self, available_actions, limit, state_memory=None):
        # Initialize most updated MCTS
        rand_seed = utils.numpy_state_int(self.np_state)
        search = mcts.MCTS(
            self.growth_model_weights_dir,
            list(ingredients.keys()),
            state_memory,
            seed=rand_seed,
        )
        # Perform rollout to determine next best media
        rollout_scores = search.perform_rollout(
            state=self.get_current_state(),
            limit=limit,
            horizon=5,
            available_actions=available_actions,
            log_graph=False,
            use_multiprocessing=True,
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
        # starting_media_state = {i: 1 for i in ingredients}
        starting_media_state = {
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
        # starting_media_state = {i: np.random.randint(0, 2) for i in ingredients}

        controller = AgentController(
            experiment_path="data/agent_state_save_ROLL1_3",
            ingredients=ingredients,
            growth_model_dir="data/neuralpy_optimization_expts/052220-sparcity-3/no_training",
            simulation_data_path="models/iSMU-test/data_20_extrapolated.csv",
            # seed=0,
        )
        controller.simulate_random_initial_data(
            n=2500,
            supplemental_data_path="data/iSMU-test/initial_data/train_set_L1OL2O.csv",
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
            "gly_exch": "glycine",
            "arg_exch": "l_arginine",
            "asn_exch": "l_asparagine",
            "asp_exch": "l_aspartic_acid",
            "cys_exch": "l_cysteine",
            "glu_exch": "l_glutamic_acid",
            "gln_exch": "l_glutamine",
            "his_exch": "l_histidine",
            "ile_exch": "l_isoleucine",
            "leu_exch": "l_leucine",
            "lys_exch": "l_lysine",
            "met_exch": "l_methionine",
            "phe_exch": "l_phenylalanine",
            "ser_exch": "l_serine",
            "thr_exch": "l_threonine",
            "trp_exch": "l_tryptophan",
            "tyr_exch": "l_tyrosine",
            "val_exch": "l_valine",
            "pro_exch": "prolines",
        }
