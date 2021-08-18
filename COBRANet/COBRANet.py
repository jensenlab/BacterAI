import sys

sys.path.append("../tigerpy")

import cobra
import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from termcolor import colored
import tigerpy as tiger
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import tqdm

import copy
import math
import multiprocessing
import os


class Data(Dataset):
    def __init__(self, X, y=None, mode="train"):
        self.mode = mode

        if self.mode == "train":
            self.X = X
            self.y = y
        else:
            self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.mode == "train":
            X_ = torch.from_numpy(np.array(self.X[idx])).float()
            y_ = torch.from_numpy(np.array(self.y[idx])).float()
            return X_, y_
        else:
            X_ = torch.from_numpy(np.array(self.X[idx])).float()
            return X_


class COBRAnet2(nn.Module):
    def __init__(self, lr=0.1, l2_reg=0.0001):
        super(COBRAnet2, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(47, 256),
            # nn.Linear(136, 256),
            nn.ReLU(),
            # nn.Linear(64, 256),
            # nn.ReLU(),
            # nn.Dropout(p=0.05, inplace=False),
            nn.Linear(256, 512),
            nn.ReLU(),
            # nn.Dropout(p=0.05, inplace=False),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(p=0.05, inplace=False),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Linear(1)
            # nn.Sigmoid(),
        )

        # self.block1 = nn.DataParallel(nn.ReLU(nn.Linear(20, 64)))
        # self.block2 = nn.DataParallel(nn.ReLU(nn.Linear(64, 256)))
        # self.block3 = nn.DataParallel(nn.ReLU(nn.Linear(256, 512)))
        # self.block4 = nn.DataParallel(nn.ReLU(nn.Linear(512, 256)))
        # self.block5 = nn.DataParallel(nn.ReLU(nn.Linear(256, 32)))
        # self.block6 = nn.DataParallel(nn.ReLU(nn.Linear(32, 1)))

        self.criterion = nn.MSELoss()
        self.mse = nn.MSELoss()
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(
        #     self.parameters(), lr=lr, weight_decay=0.001
        # )  # weight_decay=1e-2)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=l2_reg)
        self.threshold = 0.25

    def forward(self, x):
        y = self.linear_relu_stack(x)
        # x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        # y = self.block6(x)
        return y

    def evaluate(self, x):
        x = torch.from_numpy(np.array(x)).float().to(DEVICE)
        y = self.forward(x).to("cpu").detach().numpy().flatten()
        return y


AEROBIC_ONLY = True
MAX_EXCH_BOUND = 10.0
MAX_BOUND = 1000.0
# Exchange reactions to keep open, from Kenan
CDM_EXCH_RXNS = [
    "glc_exch",
    "ala_exch",
    "arg_exch",
    "asp_exch",
    "asn_exch",
    "cys_exch",
    "glu_exch",
    "gln_exch",
    "gly_exch",
    "his_exch",
    "ile_exch",
    "leu_exch",
    "lys_exch",
    "met_exch",
    "phe_exch",
    "pro_exch",
    "ser_exch",
    "thr_exch",
    "trp_exch",
    "tyr_exch",
    "val_exch",
    "ade_exch",
    "gua_exch",
    "ura_exch",
    "4abz_exch",
    "btn_exch",
    "fol_exch",
    "ncam_exch",
    "NADP_exch",
    "pnto_exch",
    "pydx_exch",
    "pydam_exch",
    "pydxn_exch",
    "ribflv_exch",
    "thm_exch",
    "Fe2_exch",
    "Mg2_exch",
    "Mn2_exch",
    "Na_exch",
    "pi_exch",
    "NO3_exch",
    "SO4_exch",
    "H2CO3_exch",
    "H2O_exch",
    "CO2_exch",
    "O2_exch",
    "O2s_exch",
]
AA_EXCH_RXNS = [
    "ala_exch",
    "arg_exch",
    "asp_exch",
    "asn_exch",
    "cys_exch",
    "glu_exch",
    "gln_exch",
    "gly_exch",
    "his_exch",
    "ile_exch",
    "leu_exch",
    "lys_exch",
    "met_exch",
    "phe_exch",
    "pro_exch",
    "ser_exch",
    "thr_exch",
    "trp_exch",
    "tyr_exch",
    "val_exch",
    # "NH4_exch",
]


class FBAModel:
    def __init__(self, model_filepath):
        self.load_gene_names()
        self.model, self.exch_rxns, self.gene_rxns, self.obj_value = self.load_model(
            model_filepath
        )

    def load_gene_names(self, file_path="iSMU_aa_aux_genes.txt"):
        with open(file_path) as f:
            self.gene_names = f.read().splitlines()

    def tiger_setup(self, model):
        model.lb[model.lb > 0] = 0  # remove NGAM
        for i, rxn in enumerate(model.rxns):
            if "_exch" in rxn:
                if rxn in CDM_EXCH_RXNS:
                    model.ub[i] = MAX_EXCH_BOUND
                else:
                    model.ub[i] = 0
            if rxn == "H_exch":
                model.ub[i] = MAX_BOUND
            if rxn == "NH4_exch":
                model.ub[i] = MAX_BOUND

    def base_model_setup(self, model):
        # There are two oxygen exchanges; shut them both off with one variable
        vo = model.addVar(lb=0.0, ub=MAX_EXCH_BOUND, name="oxy_exch")
        vo1 = model.getVarByName("O2_exch")
        vo2 = model.getVarByName("O2s_exch")
        model.addConstr(vo1 <= vo)
        model.addConstr(vo2 <= vo)
        model.Params.OutputFlag = 0  # be quiet

    def load_model(self, model_path):
        model = cobra.io.read_sbml_model(model_path)
        self.tiger_model = tiger.TigerModel.from_cobra(model)
        self.tiger_setup(self.tiger_model)

        # model = self.tiger_model.make_base_model()[0]
        model, v, g = self.tiger_model.make_base_model(self.tiger_model)
        self.base_model_setup(model)

        model.optimize()
        full_obj_val = model.ObjVal  # we'll use this later to rescale the data
        if not AEROBIC_ONLY:
            AA_EXCH_RXNS.append("oxy_exch")

        # The AAs and genes are all the input variables, including oxygen
        exch_rxns = tiger.get_vars_by_name(model, CDM_EXCH_RXNS, "MVar")
        gene_rxns = tiger.get_vars_by_name(model, self.gene_names, "MVar")

        if not AEROBIC_ONLY:
            exch_rxns[-1].ub = 0.0
            model.optimize()
            full_obj_val_anaerobic = model.ObjVal
        else:
            full_obj_val_anaerobic = 0.0

        # print("gene_rxns bounds")
        # print(gene_rxns.ub)

        # print("exch_rxns bounds ")
        # print(exch_rxns.ub)
        # print("Number of targeted genes:", gene_rxns.shape[0])
        # print("Full Objective Value:", full_obj_val)
        # if not AEROBIC_ONLY:
        #     print("Full Objective Value:", full_obj_val_anaerobic, "(anaerobic)")

        return model, exch_rxns, gene_rxns, full_obj_val

    def get_obj(self, model_inputs):
        """Modify FBA model given gene activations and media
        Return model fitness (objective value normalized to unmodified model's objective value)
        """
        # print()
        exch_inputs = model_inputs[: self.exch_rxns.shape[0]]
        # gene_inputs = model_inputs[self.exch_rxns.shape[0] :]
        # print(exch_inputs)
        # print(gene_inputs)
        # store upper bounds
        stored_exch_ub = np.copy(self.exch_rxns.ub)
        # stored_gene_ub = np.copy(self.gene_rxns.ub)

        # set upper bounds to input values
        self.exch_rxns.ub = MAX_EXCH_BOUND * exch_inputs
        # self.gene_rxns.ub = MAX_BOUND * gene_inputs
        self.model.optimize()
        fitness = self.model.ObjVal / self.obj_value
        # print(self.model.ObjVal)

        # restore upper bounds
        self.exch_rxns.ub = stored_exch_ub
        # self.gene_rxns.ub = stored_gene_ub

        return fitness

    # def _mp_helper(self, model_inputs):
    #     with gp.Env() as env, gp.Model(env=env) as model:
    #         temp_tiger = copy.deepcopy(self.tiger_model)
    #         temp_model = temp_tiger.make_base_model(model)[0]
    #         self.base_model_setup(temp_model)

    #         temp_exch_rxns = tiger.get_vars_by_name(temp_model, CDM_EXCH_RXNS, "MVar")
    #         temp_gene_rxns = tiger.get_vars_by_name(temp_model, self.gene_names, "MVar")

    #         exch_inputs = model_inputs[: self.exch_rxns.shape[0]]
    #         gene_inputs = model_inputs[self.exch_rxns.shape[0] :]

    #         # store upper bounds
    #         stored_exch_ub = np.copy(self.exch_rxns.ub)
    #         stored_gene_ub = np.copy(self.gene_rxns.ub)

    #         # set upper bounds to input values
    #         # self.exch_rxns.ub = 0
    #         temp_exch_rxns.ub = MAX_EXCH_BOUND * exch_inputs
    #         temp_gene_rxns.ub = MAX_BOUND * gene_inputs
    #         temp_model.optimize()
    #         fitness = self.model.ObjVal / self.obj_value
    #         print(fitness)
    #         # restore upper bounds
    #         temp_exch_rxns.ub = stored_exch_ub
    #         temp_gene_rxns.ub = stored_gene_ub
    #         return fitness

    # def get_obj_mp(self, inputs):
    #     """Modify FBA model given gene activations and media
    #     Return model fitness (objective value normalized to unmodified model's objective value)
    #     """

    #     with multiprocessing.Pool(processes=30) as pool:
    #         fitness = np.array(pool.map(self._mp_helper, inputs))

    #     return fitness


def simple_train(
    model,
    fba_model,
    save_path,
    save_every,
    batch_size,
    n_batches,
    lr,
    verbose=True,
    save_plots=True,
):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # reaction_bounds = []
    # for rxn in fba_model.media_reaction_names:
    #     reaction_bounds.append(fba_model.model.reactions.get_by_id(rxn).bounds))

    # return
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(
    #     model.optimizer, gamma=0.9, verbose=True
    # )
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     model.optimizer, 100000, gamma=0.98, last_epoch=-1, verbose=False
    # )
    scheduler = torch.optim.lr_scheduler.StepLR(
        model.optimizer,
        5000,
        gamma=0.8,
        last_epoch=-1,
        verbose=False
        # model.optimizer, 5000, gamma=0.98, last_epoch=-1, verbose=False
    )

    train_loss_means = []
    train_mse_means = []
    train_acc_means = []
    test_loss_means = []
    test_mse_means = []
    test_acc_means = []
    test_mse_moving_avg = []

    # data_train = Data(data_inputs, data_labels)
    # for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
    train_loss_epoch = []
    train_mse_epoch = []
    train_acc_epoch = []
    test_loss_epoch = []
    test_mse_epoch = []
    test_acc_epoch = []

    log_probabilities = np.log(np.array([math.comb(47, i) for i in range(1, 47)]))

    scaled_log_probabilities = log_probabilities / log_probabilities.sum()
    # print(f"{scaled_log_probabilities=}")
    # n_inputs = fba_model.exch_rxns.shape[0] + fba_model.gene_rxns.shape[0]
    # sparse_proportion = 0.2
    test_inputs_history = []
    test_labels_history = []
    for batch_n in range(n_batches):
        # train_inputs = np.random.choice([0, 1], size=(batch_size, n_inputs))
        train_inputs = np.random.choice(
            [0, 1], size=(batch_size, fba_model.exch_rxns.shape[0])
        )
        # gene_inputs = np.random.random(size=(batch_size, fba_model.gene_rxns.shape[0]))
        # train_inputs = np.hstack((exch_inputs, gene_inputs))
        # train_inputs = np.random.random(size=(batch_size, n_inputs))

        # print(train_inputs)
        # if batch_n % 2 == 0:
        #     # indices = np.random.choice(
        #     #     train_inputs.size,
        #     #     replace=False,
        #     #     size=int(train_inputs.size * sparse_proportion),
        #     # )
        #     # train_inputs[np.unravel_index(indices, train_inputs.shape)] = 0

        #     # Inflate zeros
        #     num_zero_inflate = np.random.choice(
        #         np.arange(1, 47), size=batch_size, p=scaled_log_probabilities
        #     )

        #     for row_idx in range(batch_size):
        #         n_zeros = num_zero_inflate[row_idx]
        #         indices = np.random.choice(
        #             np.arange(0, 48),
        #             replace=False,
        #             size=n_zeros,
        #         )
        #         train_inputs[row_idx, indices] = 0

        # train_labels = fba_model.get_obj_mp(train_inputs)
        train_labels = np.apply_along_axis(fba_model.get_obj, 1, train_inputs)
        # print(train_labels)
        # with multiprocessing.Pool(processes=30) as pool:
        #     train_labels = np.array(pool.map(fba_model.get_obj, train_inputs))

        # Train
        train_inputs = torch.from_numpy(train_inputs).float().to(DEVICE)
        train_labels = torch.from_numpy(train_labels).float().to(DEVICE).unsqueeze(1)

        # zero the parameter gradients
        model.optimizer.zero_grad()

        # forward + backward + optimize
        train_outputs = model.forward(train_inputs)
        train_loss = model.criterion(train_outputs, train_labels)

        train_loss.backward()
        model.optimizer.step()

        train_mse = model.mse(train_outputs, train_labels)

        test_inputs_history.append(torch.clone(train_inputs.detach()))
        test_labels_history.append(torch.clone(train_labels.detach()))
        if len(test_inputs_history) > 100:
            test_inputs_history.pop(0)
            test_labels_history.pop(0)

        train_outputs[train_outputs >= model.threshold] = 1
        train_outputs[train_outputs < model.threshold] = 0
        train_labels[train_labels >= model.threshold] = 1
        train_labels[train_labels < model.threshold] = 0
        train_acc = (train_outputs == train_labels).sum() / train_outputs.shape[0]

        # Train
        if batch_n > 0:
            # test_inputs = np.random.random(size=(batch_size, 20))
            # test_labels = np.apply_along_axis(fba_model.get_obj, 1, test_inputs)
            # test_inputs = np.random.choice([0, 1], size=(batch_size, 20))
            # test_labels = np.apply_along_axis(fba_model.get_obj, 1, test_inputs)
            # print("test_labels\n", test_labels)
            # with multiprocessing.Pool(processes=30) as pool:
            #     test_labels = np.array(pool.map(fba_model.get_obj, test_inputs))
            # test_inputs = torch.from_numpy(test_inputs).float().to(DEVICE)
            # test_labels = torch.from_numpy(test_labels).float().to(DEVICE).unsqueeze(1)
            test_inputs = torch.cat(test_inputs_history, 0)
            test_labels = torch.cat(test_labels_history, 0)

            test_outputs = model.forward(test_inputs)
            # print("test_outputs\n", test_outputs)
            test_loss = model.criterion(test_outputs, test_labels).item()
            test_mse = model.mse(test_outputs, test_labels).item()

            test_outputs[test_outputs >= model.threshold] = 1
            test_outputs[test_outputs < model.threshold] = 0
            test_labels[test_labels >= model.threshold] = 1
            test_labels[test_labels < model.threshold] = 0
            test_acc = (
                (test_outputs == test_labels).sum() / test_outputs.shape[0]
            ).item()
        else:
            test_loss = 0
            test_mse = 0
            test_acc = 0

        train_loss_epoch.append(train_loss.item())
        train_mse_epoch.append(train_mse.item())
        train_acc_epoch.append(train_acc.item())
        test_loss_epoch.append(test_loss)
        test_mse_epoch.append(test_mse)
        test_acc_epoch.append(test_acc)

        tr_l = sum(train_loss_epoch) / len(train_loss_epoch)
        tr_mse = sum(train_mse_epoch) / len(train_mse_epoch)
        tr_acc = sum(train_acc_epoch) / len(train_acc_epoch)
        te_l = sum(test_loss_epoch) / len(test_loss_epoch)
        te_mse = sum(test_mse_epoch) / len(test_mse_epoch)
        te_acc = sum(test_acc_epoch) / len(test_acc_epoch)

        # train_loss_means.append(tr_l)
        # train_mse_means.append(tr_mse)
        # train_acc_means.append(tr_acc)
        # test_loss_means.append(te_l)
        # test_mse_means.append(te_mse)
        # test_acc_means.append(te_acc)

        # moving_subset = test_mse_means[-10:]
        # moving_avg_mse = sum(moving_subset) / len(moving_subset)
        # test_mse_moving_avg.append(moving_avg_mse)

        # if batch_n > 1000:
        scheduler.step()

        if verbose and batch_n % 100 == 0:
            results = "{} | Train - Acc: {} MSE: {} | Test - Acc: {} MSE: {} | LR: {}"
            print(
                results.format(
                    colored(f"{f'{batch_n}/{n_batches}':>14}", "yellow"),
                    colored(f"{tr_acc:.5f}", "green"),
                    colored(f"{tr_mse:.5f}", "green"),
                    colored(f"{te_acc:.5f}", "green"),
                    colored(f"{te_mse:.5f}", "green"),
                    colored(f"{scheduler.get_last_lr()[0]:.6f}", "blue"),
                )
            )

        if batch_n % save_every == 0:
            model_path = os.path.join(save_path, f"checkpoint_{batch_n}.pkl")
            torch.save(model, model_path)

    model_path = os.path.join(save_path, f"checkpoint_{batch_n+1}.pkl")
    torch.save(model, model_path)
    # if save_plots:
    #     x = range(1, epochs + 1)
    #     fig, axs = plt.subplots(
    #         nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10, 5)
    #     )

    #     axs[0].plot(x, train_loss_means)
    #     axs[0].plot(x, train_mse_means)
    #     axs[0].plot(x, test_loss_means)
    #     axs[0].plot(x, test_mse_means)
    #     axs[0].plot(x, test_mse_moving_avg)

    #     axs[1].plot(x, test_acc_means)
    #     axs[1].plot(x, train_acc_means)
    #     axs[0].legend(
    #         [
    #             "train_loss",
    #             "train_mse",
    #             "test_loss",
    #             "test_mse",
    #             "test_mse_moving_avg-10",
    #         ]
    #     )
    #     axs[1].legend(
    #         [
    #             "train_acc",
    #             "test_acc",
    #         ]
    #     )
    #     axs[0].set_xlabel("Epoch")
    #     axs[1].set_xlabel("Epoch")
    #     plt.title("NN performance")
    #     plt.tight_layout()
    #     plt.savefig("result_training_nn.png")
    #     plt.close()

    # final_moving_avg = test_mse_moving_avg[-1]
    # if print_status:
    #     print(f"Final 10-Moving Avg Epoch MSE: {final_moving_avg}")

    # return final_moving_avg


DEVICE = torch.device("cuda:0")

if __name__ == "__main__":

    # name_mappings_csv = "files/name_mappings_aa.csv"
    # mapped_data_csv = "data/iSMU-012920/initial_data/mapped_data_SMU_combined.csv"
    # experiments, data_growth = utils.parse_data_map(
    #     name_mappings_csv, mapped_data_csv, components
    # )

    # data_growth[data_growth >= 0.25] = 1
    # data_growth[data_growth < 0.25] = 0
    # aerobic_indexes = experiments["aerobic"] == "5% CO2 @ 37 C"
    # experiments.loc[aerobic_indexes, "aerobic"] = 1
    # experiments.loc[~aerobic_indexes, "aerobic"] = 0
    # experiments.to_csv("experiments_df.csv")
    # data_growth.to_csv("growth_data_df.csv")
    # media_names = experiments.columns.to_list()[:-1]  # exclude "aerobic" from names

    # print(experiments)
    # print(data_growth)
    nn_model = COBRAnet2().to(DEVICE)
    fba_model = FBAModel("../models/iSMU_rescaled.xml")

    # nn_model = torch.load("cobranet2/final_model.pkl").to(DEVICE)
    # experiment_name = "cobranet2_genes_sparse_inflate_mod2"
    experiment_name = "models/cobranet2_binary_media"
    simple_train(
        nn_model,
        fba_model,
        save_path=experiment_name,
        save_every=1000,
        lr=0.05,
        batch_size=32,
        n_batches=50000,
    )

    # entire_set = pd.read_csv(
    #     "models/SMU_NN_oracle_extrapolated/data_20_extrapolated.csv", index_col=None
    # ).drop(columns=["grow"])

    # objs = []
    # copy = entire_set.to_numpy()
    # for i in tqdm.trange(len(entire_set)):
    #     objs.append(fba_model.get_obj(copy[i]))
    #     # if i % 1000 == 0:
    #     # print(i)

    # entire_set["cobra_obj"] = objs
    # entire_set.to_csv("entire_aa_set_cobra.csv", index=False)

    # entire_set = pd.read_csv(
    #     "entire_aa_set_cobra.csv", index_col=None
    # )  # .iloc[:10000, :]
    # cobra_obj = entire_set["cobra_obj"].to_numpy()
    # # entire_set = entire_set.iloc[:250000, :-1].to_numpy()

    # n_inputs = fba_model.exch_rxns.shape[0] + fba_model.gene_rxns.shape[0]
    # inputs = np.random.random(size=(10000, n_inputs))
    # labels = np.apply_along_axis(fba_model.get_obj, 1, inputs)
    # for i in range(1, 10):
    #     every = 5000
    #     nn_model = torch.load(f"{experiment_name}/checkpoint_{every*i}.pkl").to(DEVICE)

    #     nn_preds = nn_model.evaluate(inputs)

    #     x = labels
    #     y = nn_preds
    #     xy = np.vstack([x, y])
    #     z = gaussian_kde(xy)(xy)
    #     # Sort the points by density, so that the densest points are plotted last
    #     idx = z.argsort()
    #     x, y, z = x[idx], y[idx], z[idx]
    #     fig, ax = plt.subplots()
    #     ax.scatter(x, y, c=z, s=1)
    #     ax.plot([0, 1], [0, 1], "r-", alpha=0.5)

    #     mse = mean_squared_error(x, y)

    #     ax.set_title(f"Sparse inflate - {every*i} batches MSE={mse}")
    #     ax.set_xlabel("iSMU")
    #     ax.set_ylabel("NN")
    #     # fig.legend()
    #     fig.tight_layout()
    #     plt.savefig(f"results_{experiment_name}_{every*i}.png", dpi=400)

    # nn_model = torch.load(f"{experiment_name}/checkpoint_15000.pkl").to(DEVICE)
    # # 47
    # # 1081
    # # 16215
    # import itertools

    # total = 34686 // 2
    # n_inputs = fba_model.gene_rxns.shape[0]
    # gene_inputs = np.ones((total, n_inputs))

    # # fig, ax = plt.subplots()
    # fig = plt.figure()
    # mse = []
    # for j in range(2):
    #     if j == 0:
    #         cdm_inputs = np.ones((total, fba_model.exch_rxns.shape[0]))
    #     else:
    #         cdm_inputs = np.zeros((total, fba_model.exch_rxns.shape[0]))

    #     input_idx = 0
    #     for n in range(1, 4):
    #         combos = itertools.combinations(range(47), n)
    #         for idxs in combos:
    #             cdm_inputs[input_idx, idxs] = 0 if j == 0 else 1
    #             input_idx += 1

    #     inputs = np.hstack((cdm_inputs, gene_inputs))
    #     labels = np.apply_along_axis(fba_model.get_obj, 1, inputs)
    #     nn_preds = nn_model.evaluate(inputs)
    #     plt.plot(
    #         labels,
    #         nn_preds,
    #         ".",
    #         alpha=0.05,
    #         markersize=2,
    #         label="leave outs" if j == 0 else "leave ins",
    #     )

    #     mse.append(mean_squared_error(labels, nn_preds))

    # plt.plot([0, 1], [0, 1], "r-", alpha=0.5, label="x=y")

    # plt.title(f"Leave outs 30k batches LO-MSE={mse[0]}, LI-MSE={mse[1]}")
    # plt.xlabel("iSMU")
    # plt.ylabel("NN")
    # plt.legend()
    # fig.tight_layout()
    # plt.savefig(f"results_{experiment_name}_leaveoutsins.png", dpi=400)
