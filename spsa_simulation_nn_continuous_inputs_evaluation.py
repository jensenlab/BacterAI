

import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import neural_pretrain as neural
import spsa


def load_data(filepath, starting_index=1, max_n=None):
    raw_data = np.genfromtxt(filepath, delimiter=",", dtype=np.float32)[1:, :]
    if max_n:
        raw_data = raw_data[
            np.random.choice(raw_data.shape[0], size=max_n, replace=False), :
        ]
    data = raw_data[starting_index:, :-1]
    data_labels = raw_data[starting_index:, -1]
    return data, data_labels

if __name__ == "__main__":
    train_size = 0.001
    n_perturbations = 20
    perturbation_range = (0.01, 0.20)
    n_trials = 1000
    positive = True

    x, y = load_data(
        filepath=f"models/iSMU-test/data_20_extrapolated.csv",
        # filepath=f"models/iSMU-test/data_20_extrapolated_with_features.csv",
        starting_index=0,
    )
    x_train, _, y_train, _ = train_test_split(
        x, y, train_size=train_size, random_state=0
    )

    growth_model_dir="data/neuralpy_optimization_expts/052220-sparcity-3/no_training"
    model = neural.PredictNet.from_save(growth_model_dir)
    model.train(x_train, y_train)

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False,figsize=(12,10))
    fig.tight_layout(pad=4)
    
    p_range = range(0, n_perturbations+1, 5)
    for positive in range(2):
        errors = []
        for n_pertubation in p_range:
            trial_inputs = np.empty((n_trials * 2, x_train.shape[1]), dtype=np.float32)
            rand_inputs = np.random.choice((0.0, 1.0), size=(n_trials, x_train.shape[1])).astype(np.float32)
            trial_inputs[0:n_trials, :] = rand_inputs

            # perturb_proportional = np.ones((n_trials, x_train.shape[1]))
            begin_sum = np.sum(trial_inputs[0:n_trials, :], axis=1)
            for i, x in enumerate(rand_inputs):
                trial_inputs[i + n_trials, :] = x.copy().astype(np.float32)
                # ones = np.where(x == 1)[0]
                # min_n_p = min(ones.shape[0] , n_pertubation)
                change_idxs = np.random.choice(range(trial_inputs.shape[1]), size=n_pertubation, replace=False)
                p = np.random.uniform(perturbation_range[0], perturbation_range[1])
                if positive == 1:
                    p *= -1
                for j, idx in enumerate(change_idxs):
                    orig_value = trial_inputs[i + n_trials, idx]
                    new_value = trial_inputs[i + n_trials, idx] + p
                    trial_inputs[i + n_trials, idx] = new_value

                    # proportional_change = abs(orig_value - new_value)/orig_value
                    # perturb_proportional[i, idx] = proportional_change
            final_sum = np.sum(trial_inputs[n_trials:, :], axis=1, dtype=np.float32)        
            proportional_change = final_sum/begin_sum

            # print("proportional_change",proportional_change.shape)
            # print("trial_inputs", trial_inputs.shape)
            trial_predictions = np.reshape(model.model.predict(trial_inputs), (trial_inputs.shape[0],))
            # print("trial_predictions:", trial_predictions.shape)
            x_range = range(1, n_trials + 1)

            # mean_perturb_proportional = np.mean(perturb_proportional, axis=1)

            axs[0, positive].plot(trial_predictions[0:n_trials], trial_predictions[n_trials:], ".", markersize=1)
            axs[0, positive].set(xlabel='Binary Input Prediction', ylabel='Perturbed Input Prediction')

            # print("mean perturb_proportional", mean_perturb_proportional)

            error = np.multiply(trial_predictions[0:n_trials], proportional_change)-trial_predictions[n_trials:]
            errors.append(error)
            axs[1, positive].plot(proportional_change, error, ".", markersize=1)
            axs[1, positive].set(xlabel='Perturb input relative change (of sum)', ylabel='Error of scaled results')

        errors = np.vstack(errors)
        y, bins = np.histogram(errors,bins=100)
        bin_centers = 0.5*(bins[1:]+bins[:-1])
        ax2 = axs[1, positive].twiny()

        ax2.set_xlabel('Hist count', color='r')
        for tl in ax2.get_xticklabels():
            tl.set_color('r')

        ax2.plot(y,bin_centers,'r-')


            # axs[1, positive].hist(error, bins=30, alpha=0.5, orientation="horizontal")
    
    axs[0, 0].set_title("SPSA Simulation (+)")
    axs[0, 1].set_title("SPSA Simulation (-)")
    for x in axs.flatten():
        x.legend([f"{i} perturbations" for i in p_range])
        
    plt.savefig("SPSA_Simulation_10.png", dpi=150)
    plt.close()

    # for X data points
        # get binary and continuous growth result from binary input
        # calculate perturbation for N ingredients
        # apply perturbation to the binary input
        # get continuous growth result from continuous input
        # graph both cont. growth results
        # calc error against proportionally scaled input, graph on another separate graph

    # Show graphs



