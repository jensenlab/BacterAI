import csv
import datetime
import os
import shutil

import numpy as np
import pandas as pd
from prompt_toolkit import HTML, print_formatted_text
from prompt_toolkit.styles import Style
from yaspin import yaspin

# Suppress Tensorflow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import agent
import dnf
import neural_pretrain as neural
import utils

# Spinner text style
style = Style.from_dict(
    {"title": "#ba52ff bold", "msg": "#03c2fc bold", "sub-msg": "#616161"}
)


class Benchmark(object):
    def __init__(self, seed=None):
        self.input_seed = seed
        self.np_state = utils.seed_numpy_state(seed)
        self.spinner = yaspin(text="Running", color="cyan")
        self.start_time = datetime.datetime.now()

    def generate_test_case(
        self, n, output_file, card_low=5, card_high=15, mu_low=2, mu_high=10, **kwargs
    ):
        """
        Generate a new dnf.Rule for use as a test case
        """
        writer = csv.writer(output_file, delimiter=",")
        writer.writerow(
            [
                "Test Case Parameters:",
                f"Rule: ({card_low}, {card_high})",
                f"Mu: ({mu_low}, {mu_high})",
            ]
        )

        rule_length = self.np_state.randint(card_low, card_high)
        ors_mu = self.np_state.randint(mu_low, mu_high)
        ands_mu = self.np_state.randint(mu_low, mu_high)

        rule = dnf.Rule(
            rule_length,
            poisson_mu_OR=ors_mu,
            poisson_mu_AND=ands_mu,
            seed=utils.numpy_state_int(self.np_state),
        )
        return rule

    def create_dummy_data(self, rule, output_path, training_percentage=0.001):
        """
        Create temporary date used for training the growth value neural network 
        """

        n_experiments = 2 ** rule.dimension
        n_training_examples = 0

        self.spin_text(f"Creating training data ({n_experiments})")
        while n_training_examples is 0:
            n_training_examples = int(n_experiments * training_percentage)
            training_percentage *= 2  # Double percentage each time

        col_names = rule.generate_data_csv(output_path, quantity=n_training_examples)
        return col_names

    def get_working_dir(self, make=True):
        """
        Find and/or make the working directory for temporary files 
        """

        dir_name = ".test_suite_working_dir"
        if not os.path.isdir(dir_name) and make:
            os.makedirs(dir_name)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name)

    def clean_working_dir(self):
        """
        Clean (delete) the working directory
        """

        w_dir = self.get_working_dir(make=False)
        if os.path.isdir(w_dir):
            shutil.rmtree(w_dir)

    def score(self, final_state, rule):
        """
        Compute a score based on the test results. 
        Return True if it succeeded in indentifying a minimal cardinality media 
        or False if it failed, along with the score.
        """

        target_cardinality = rule.minimum_cardinality
        full_length = rule.dimension

        final_cardinality = sum(final_state)

        n_removable = full_length - target_cardinality
        n_removed = full_length - final_cardinality
        score = int(n_removed / n_removable * 100 // 1)

        success = score == 100
        return success, score

    def process_results(self, results, rule, output_file):
        target = rule.minimum_cardinality

        # File outputs
        writer = csv.writer(output_file, delimiter=",")
        writer.writerows(
            [
                ["Rule:"] + [" ".join(map(str, i)) for i in rule.definition],
                ["Target Cardinality:", target],
                [""],
            ]
        )

        self.spinner.hide()
        for policy_n, states in results.items():
            self.print_formatted("Policy iteration:", policy_n)
            writer.writerows(
                [[f"Policy iteration {policy_n}"], ["Result", "Score", "Final state"]]
            )

            for s in states:
                success, score = self.score(s, rule)
                self.spin_text(f"Score: {score}, Final cardinality: {sum(s)}")
                if success:
                    self.spinner.green.ok("✔ SUCCESS")
                    output_result = "SUCCESS"
                else:
                    self.spinner.red.fail("✘ FAIL")
                    output_result = "FAIL"

                self.print_formatted(sub_msg=f"Final state: {s}", prefix="")
                writer.writerow([output_result, score, s])

        self.spinner.show()

    def perform(self, n_cases=5, n_agents=2, output_dir="benchmark_results", **kwargs):
        # Set up output file
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        file_name = f"BacterAI-Bench-{self.start_time.strftime('%Y%m%d-%H%M%S')}.csv"
        output_file = open(os.path.join(output_dir, file_name), "w")

        # File Header
        writer = csv.writer(output_file, delimiter=",")
        writer.writerows(
            [
                ["BacterAI Benchmark"],
                ["Seed:", self.input_seed if self.input_seed != None else "N/A"],
                [""],
            ]
        )

        for case in range(1, n_cases + 1):
            self.spinner.start()
            self.spinner.hide()
            self.print_formatted(title=f"Test Case {case}", prefix="")

            self.spinner.show()
            writer.writerow([f"Test Case {case}"])
            try:
                # Create test case
                self.spin_text("Creating test case")
                rule = self.generate_test_case(n_cases, output_file, **kwargs)

                self.print_formatted("Target cardinality:", rule.minimum_cardinality)
                self.print_formatted("Input dimension:", rule.dimension)
                self.print_formatted("Rule:", rule.get_definition())

                self.spin_text("Initializing working directory")
                working_dir = self.get_working_dir()
                train_data_path = os.path.join(working_dir, "train_data.csv")

                col_names = self.create_dummy_data(rule, train_data_path, 0.001)

                log_dir = os.path.join(working_dir, "tf_logs")
                if not os.path.isdir(log_dir):
                    os.makedirs(log_dir)
                model_dir = os.path.join(working_dir, "model")
                if not os.path.isdir(model_dir):
                    os.makedirs(model_dir)

                # Create new growth predict net
                self.spin_text("Creating growth net")
                rand_seed = utils.numpy_state_int(self.np_state)
                net = neural.PredictNet(
                    n_test=0,
                    exp_id=0,
                    parent_logdir=log_dir,
                    save_model_path=model_dir,
                    n_epochs=10,
                    seed=rand_seed,
                )

                # Train growth predict net
                self.spin_text("Training growth net")
                train_data = pd.read_csv(train_data_path, index_col=None)
                x_train, y_train = train_data.values[:, :-1], train_data.values[:, -1]

                net.train(x_train, y_train)
                net.save()  # Saves model and weights for MCTS use

                # Initialize agent controller
                starting_state = {n: 1 for n in col_names if n != "grow"}
                data_path = os.path.join(working_dir, "train_data.csv")

                self.spin_text("Initializing agent controller")
                rand_seed = utils.numpy_state_int(self.np_state)
                controller = agent.AgentController(
                    ingredients=[c for c in col_names if c != "grow"],
                    growth_model_dir=model_dir,
                    simulation_rule=rule,
                    seed=rand_seed,
                )

                # Update agent controller history with training data for use in all policy iterations
                controller.update_history(train_data)

                # Simulate agents
                self.spin_text(f"Simulating {n_agents} agents")
                results = controller.simulate(
                    n_agents=n_agents,
                    n_policy_iterations=10,
                    starting_state=starting_state,
                )

                # Process simulation results
                self.process_results(results, rule, output_file)
            except Exception as e:
                self.spinner.write(f"> Error: {e}")
            finally:
                # File clean up
                self.clean_working_dir()
                self.spinner.write("> Cleaned working directory")

        self.spinner.write("")
        self.spinner.stop()
        output_file.close()

    def print_formatted(self, msg="", sub_msg="", title="", prefix=">"):
        print_formatted_text(
            HTML(
                f"{prefix}<title>{title}</title> <msg>{msg}</msg> <sub-msg>{sub_msg}</sub-msg>"
            ),
            style=style,
        )

    def spin_text(self, text):
        self.spinner.text = text


if __name__ == "__main__":
    test = Benchmark(seed=0)
    test.perform(1, 1, card_low=20, card_high=21, mu_low=5, mu_high=15)

    # rand()
