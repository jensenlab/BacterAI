import argparse
import collections
import csv
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dnf
import utils

INDENT = "  "


def get_indent(n):
    return "".join([INDENT for _ in range(n)])


# Logging set up
logger = logging.getLogger(__name__)
LOGLEVELS = (logging.DEBUG, logging.INFO, logging.ERROR)
# Levels, descending
LOGTYPE = collections.namedtuple("LOGTYPE", "debug info error")
LOG = LOGTYPE(logger.debug, logger.info, logger.error)

INDENT = "  "

# CLI argument set up
parser = argparse.ArgumentParser(description="Run spsa.py")

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


class SPSA(object):
    def __init__(self, W, c=0.20, seed=None, a=1e-3, A=1000, alpha=0.602, gamma=0.101):
        self.W = np.array(W)
        self.current_iteration = 0
        self.c = c
        self.np_state = utils.seed_numpy_state(seed)
        self.A = A
        self.a = a
        self.alpha = alpha
        self.gamma = gamma

        self.W_fd = np.array(W)

    def generate_perturbation(self):
        """ 
        Generates a perturbation vector using a Bernoulli distrbution, 
        choosing between -1 and 1
        """

        b = self.np_state.choice(
            [-1, 1], size=self.W.shape[0], p=[0.5, 0.5], replace=True
        )
        return b

    def get_c(self):
        """
        Calculates new c_k value based on current iteration. 
        Good rule of thumb: c at a level approximately equal to the standard deviation of the measurement noise in y() -- loss_fn
        """
        return self.c
        # return self.c / ((self.current_iteration + 1)**self.gamma)

    def get_a(self):
        """
        Calculates new a_k value based on current iteration
        """
        return self.a
        # return self.a / (self.current_iteration + 1 + self.A) ** self.alpha

    def gen_spsa_experiments(self, n_grads):
        # Generate n_grads. Larger number of gradients -->
        # more accurate average gradient approx
        experiments = []
        perturbations = []
        for n in range(n_grads):
            # Make perturbations
            p = self.generate_perturbation() * self.get_c()
            p_plus = self.W + p
            p_minus = self.W - p
            experiments.append((p_plus, p_minus))
            perturbations.append(p)

        return experiments, perturbations

    def spsa_step(self, n_grads, loss_fn):
        """SPSA step"""

        # print(f"\n###### Iteration {self.current_iteration} ######")
        grads = np.empty((n_grads, self.W.shape[0]))

        # gen perturbation experiments
        experiments, perturbations = self.gen_spsa_experiments(n_grads)

        for i, (expt, perturb) in enumerate(zip(experiments, perturbations)):
            # print(f"Grad {n}")
            p_plus, p_minus = expt

            x_data = self.np_state.choice((0, 1), size=self.W.shape[0])
            # x_true = data_x[index, :]
            # y_true = data_y[index]

            result_plus = x_data.dot(p_plus)
            result_minus = x_data.dot(p_minus)
            # print("results (-, +):", result_plus, result_minus)
            loss_plus = loss_fn(x_data, result_plus)
            loss_minus = loss_fn(x_data, result_minus)

            grad = compute_spsa_gradient(loss_plus, loss_minus, perturb)
            # print(f"Gradient {n}: {grad}")
            grads[i, :] = grad

        # elementwise mean of gradients
        mean_grad = np.mean(grads, axis=0)
        # print(f"Mean gradient: {mean_grad}")

        old_W = self.W
        # update weights
        self.W = self.W - self.get_a() * mean_grad

        return old_W, mean_grad

    def gen_fdsa_experiments(self):
        w_dim = self.W_fd.shape[0]
        experiments = []
        for i in range(w_dim):
            e_i = np.zeros(w_dim)
            e_i[i] = 1

            W_fd_plus = self.W_fd + self.get_c() * e_i
            W_fd_minus = self.W_fd - self.get_c() * e_i
            experiments.append((W_fd_plus, W_fd_minus))
        return experiments

    def fdsa_step(self, loss_fn):
        """ FDSA step """

        w_dim = self.W_fd.shape[0]
        grad_fd = np.empty(w_dim)

        experiments = self.gen_fdsa_experiments()

        for i, expt in enumerate(experiments):
            W_fd_plus, W_fd_minus = expt

            x_data = self.np_state.choice((0, 1), size=w_dim)
            result_plus = x_data.dot(W_fd_plus)
            result_minus = x_data.dot(W_fd_minus)

            loss_plus = loss_fn(x_data, result_plus)
            loss_minus = loss_fn(x_data, result_minus)

            grad_fd[i] = compute_fdsa_gradient(loss_plus, loss_minus, self.get_c())

        old_W_fd = self.W_fd
        # update weights
        self.W_fd = self.W_fd - self.get_a() * grad_fd
        return old_W_fd, grad_fd

    def run(self, loss_fn, test_fn, n_grads=5, episilon=1e-5):
        total_samples = 0
        total_samples_fd = 0
        all_mse = []
        all_mse_fd = []
        all_angular_dist = []
        all_rmse = []

        w_dim = self.W.shape[0]
        while True:

            old_W, spsa_grad = self.spsa_step(n_grads, loss_fn)
            total_samples += 2 * n_grads

            old_W_fd, fdsa_grad = self.fdsa_step(loss_fn)
            total_samples_fd += 2 * w_dim

            angular_distance = compute_angle(spsa_grad, fdsa_grad)
            rmse = compute_rmse(spsa_grad, fdsa_grad)
            # if np.linalg.norm(old_W - self.W) / np.linalg.norm(old_W) <= episilon:
            #     print("Less than epsilon! Done!")
            #     break

            ## Output log and graphing
            graph_resolution = 250
            if (
                self.current_iteration % graph_resolution == 0
                and self.current_iteration >= graph_resolution
            ):
                mse = test_fn(self.W)
                mse_fd = test_fn(self.W_fd)

                all_mse.append(mse)
                all_mse_fd.append(mse_fd)
                all_angular_dist.append(angular_distance)
                all_rmse.append(rmse)

                fig, axs = plt.subplots(
                    nrows=2, ncols=1, sharex=False, sharey=False, figsize=(6, 10)
                )

                plt.title("SPSA v. FDSA")
                axs[0].set_title("SPSA v. FDSA")
                axs[0].set_ylabel("MSE")
                x_range = range(
                    0,
                    len(all_mse) * graph_resolution * 2 * n_grads,
                    graph_resolution * 2 * n_grads,
                )
                axs[0].plot(x_range, all_mse, "-r", x_range, all_mse_fd, "-b")
                axs[0].set_xlabel("# loss_fn calls (SPSA)")
                axs[0].legend(["SPSA", "FDSA"])

                ax2 = axs[0].secondary_xaxis(
                    "top",
                    functions=(
                        lambda x: x * w_dim / n_grads,
                        lambda x: x * n_grads / w_dim,
                    ),
                )
                ax2.set_xlabel("# loss_fn calls (FDSA)")

                axs[1].set_title("Stats")
                axs[1].set_ylabel("Angular Distance")
                axs[1].axis(ymin=0, ymax=180)
                axs[1].plot(x_range, all_angular_dist, "-")

                ax3 = axs[1].twinx()
                ax3.set_ylabel("RMSE", color="g")
                for tl in ax3.get_yticklabels():
                    tl.set_color("g")
                ax3.plot(x_range, all_rmse, "g-")

                plt.savefig("SPSA.png", dpi=150)
                plt.close()

                print(
                    f"Iter: {self.current_iteration}\tMSE: {mse}\tMSE_fd: {mse_fd}\t n_samples: {total_samples} - {total_samples_fd}"
                )

            self.current_iteration += 1


def compute_spsa_gradient(result_plus, result_minus, perturbation):
    return (result_plus - result_minus) / (2 * perturbation)


def compute_fdsa_gradient(result_plus, result_minus, c):
    return (result_plus - result_minus) / (2 * c)


def compute_gradients(c, results_csv, perturbations_csv=None, is_fdsa=False):
    if is_fdsa:
        # FDSA Data
        fdsa_results = []
        with open(results_csv, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)
            for row in reader:
                # print(row)
                n_ingredients = int(row[0])
                plus = np.array(row[1 : n_ingredients + 1]).astype(np.float)
                minus = np.array(row[n_ingredients + 1 :]).astype(np.float)
                # print(len(plus), plus)
                # print(len(minus), minus)

                fdsa_results.append((plus, minus))

        all_gradients = []
        for results_plus, results_minus in fdsa_results:
            # for result_p, results_m in zip(results_plus, results_minus):
            grad = (results_plus - results_minus) / (2 * c)
            all_gradients.append(grad)

        return all_gradients

    elif not is_fdsa and perturbations_csv:
        # SPSA Data
        spsa_perturbations = []
        with open(perturbations_csv, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)
            for row in reader:
                # print(row)
                n_ingredients = int(row[0])
                pertubations = []
                for i in range(n_ingredients):
                    p = np.array(row[i + 1].split(" ")).astype(np.float)
                    pertubations.append(p)
                spsa_perturbations.append(pertubations)

        spsa_results = []
        with open(results_csv, "r") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)

            for row in reader:
                # print(row)
                n_ingredients = int(row[0])
                plus = np.array(row[1 : n_ingredients + 1]).astype(np.float)
                minus = np.array(row[n_ingredients + 1 :]).astype(np.float)
                # print(len(plus), plus)
                # print(len(minus), minus)
                spsa_results.append((plus, minus))

        all_gradients = []
        for perturbations, (results_plus, results_minus) in zip(
            spsa_perturbations, spsa_results
        ):
            grads = []
            for p, result_plus, result_minus in zip(
                perturbations, results_plus, results_minus
            ):
                grad = compute_spsa_gradient(result_plus, result_minus, p)
                grads.append(grad)

            mean_grads = []
            for i in range(1, len(perturbations) + 1):
                if i == 1:
                    mean_grads.append(grads[0])
                else:
                    mean_grads.append(np.mean(np.vstack(grads[:i]), axis=0))
                # print("\n", mean_grads[-1])
            all_gradients.append(mean_grads)
        return all_gradients
    else:
        raise Exception("Need perturbations CSV")


def compute_angle(r1, r2):
    "Compute angle between two numpy arrays, r1 and r2"

    if r1.shape[0] != r2.shape[0]:
        raise Exception(f"Shape mismatch: {r1.shape[0]}, {r2.shape[0]}")

    cos_similarity = r1.dot(r2) / (np.sqrt(r1.dot(r1)) * np.sqrt(r2.dot(r2)))
    angular_distance = np.arccos(cos_similarity)
    angular_distance = angular_distance * 180 / np.pi
    return angular_distance


def compute_rmse(r1, r2):
    r1 /= np.sqrt(r1.dot(r1))
    r2 /= np.sqrt(r2.dot(r2))
    return np.sqrt(np.mean((r1 - r2) ** 2))


def compare_gradients(
    c,
    spsa_results_csv="spsa_results.csv",
    spsa_perturbation_csv="spsa_perturbations_R1.csv",
    fdsa_results_csv="fdsa_results.csv",
):
    all_spsa_mean_gradients = compute_gradients(
        c, spsa_results_csv, spsa_perturbation_csv, is_fdsa=False
    )
    fdsa_gradients = compute_gradients(c, fdsa_results_csv, is_fdsa=True)

    all_angles = []
    for f_grad, s_grads in zip(fdsa_gradients, all_spsa_mean_gradients):
        angles = []
        for s_grad in s_grads:
            # for g in mean_grads:
            #     print("\nSPSA MEAN GRAD", g)
            angle = compute_angle(f_grad, s_grad)
            angles.append(angle)
        all_angles.append(angles)
        # angles = list(map(compute_angle, mean_grads, fdsa_gradients))
        # print(f"{idx+1} SPSA Grad angular dist", angles)
    return all_angles


def simulate_with_rule(rule_length, seed=0):
    inputs = np.random.rand(rule_length)
    print(f"Random staring weights: {inputs}")

    spsa = SPSA(inputs)

    rule = dnf.Rule(rule_length, poisson_mu_OR=5, poisson_mu_AND=5, seed=seed,)

    def mse(x, y):
        y_true = rule.evaluate(x)
        err = (y - y_true) ** 2
        # print(f"MSE: {err}"")
        return err

    def test_fn(W):
        x = np.random.choice((0, 1), size=(100, rule_length))
        y_true = rule.evaluate(x)
        y_pred = np.sum(x * W, axis=1)
        # print(x, y_true, y_pred)
        # y_pred[y_pred >= 0.5] = 1
        # y_pred[y_pred < 0.5] = 0

        mse = np.mean(np.power(y_pred - y_true, 2))
        return mse

    spsa.run(mse, test_fn, n_grads=10)


if __name__ == "__main__":
    simulate_with_rule(rule_length=40)
