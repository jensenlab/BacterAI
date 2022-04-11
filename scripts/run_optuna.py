import datetime
import torch.optim as optim
import optuna
import pandas as pd
from torch import nn

from ..net import DatasetAminoAcids, NeuralNetwork, DEVICE


def optuna_objective(trial):
    data_train = pd.read_csv("data/gpr_train_pred_0.20.csv", index_col=None)
    data_test = pd.read_csv("data/gpr_test_pred_0.20.csv", index_col=None)
    data_train = DatasetAminoAcids(
        data_train.to_numpy()[:, :-3], data_train.to_numpy()[:, -3]
    )
    data_test = DatasetAminoAcids(
        data_test.to_numpy()[:, :-3], data_test.to_numpy()[:, -3]
    )

    # Generate the model.
    # model = define_model(trial).to(DEVICE)
    model = NeuralNetwork().to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    wd = trial.suggest_loguniform("weight_decay", 1e-5, 1e-1)
    model.optimizer = getattr(optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=wd
    )

    # loss_name = trial.suggest_categorical(
    #     "loss_criterion", ["BCELoss", "MSELoss", "L1Loss"]
    # )
    # losses = {"BCELoss": nn.BCELoss(), "MSELoss": nn.MSELoss(), "L1Loss": nn.L1Loss()}
    # model.criterion = losses[loss_name]
    batch_size = trial.suggest_int("batch_size", 128, 2048)
    epochs = 250
    # batch_size = 512

    mse = train(
        model,
        data_train,
        data_test,
        epochs,
        batch_size,
        lr,
        print_status=False,
        save_plots=False,
        optuna_trial=trial,
    )
    return mse


def define_model(trial):
    # We optimize the number of layers, hidden untis and dropout ratio in each layer.
    n_layers = trial.suggest_int("n_layers", 3, 8)
    layers = []

    in_features = 20
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 32, 2048)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        # p = trial.suggest_uniform("dropout_l{}".format(i), 0.05, 0.5)
        # layers.append(nn.Dropout(p))
        in_features = out_features

    layers.append(nn.Linear(in_features, 1))
    layers.append(nn.Sigmoid())

    model = NeuralNetwork().to(DEVICE)
    model.linear_relu_stack = nn.Sequential(*layers)

    return model


def run_optuna(n_trials=100):
    start_date = datetime.datetime.now().isoformat().replace(":", ".")
    study = optuna.create_study(
        study_name="SMU_nn_optim_20",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=50),
    )
    study.optimize(optuna_objective, n_trials=n_trials)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("- Number of finished trials: ", len(study.trials))
    print("- Number of pruned trials: ", len(pruned_trials))
    print("- Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("- Value: ", trial.value)

    print("- Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    df_optuna = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df_optuna.to_csv(f"optuna_trial_results_{start_date}.csv")
    print(df_optuna)

    ax = optuna.visualization.matplotlib.plot_intermediate_values(study)
    ax.legend()
    ax.figure.set_figheight(10)
    ax.figure.set_figwidth(20)
    ax.figure.savefig(f"optuna_trial_viz_int_values_{start_date}.png", dpi=400)

    print(f"Finished experiment: {start_date}")


if __name__ == "__main__":
    run_optuna(1000)