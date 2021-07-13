import sys

sys.path.append("../tigerpy")

import cobra
import gurobipy
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tigerpy as tiger
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import tqdm

# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler

import multiprocessing

import utils
import model


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
    def __init__(self, lr=0.01):
        super(COBRAnet2, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
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
        # self.optimizer = Adam(self.parameters(), lr=lr)
        self.optimizer = Adam(
            self.parameters(), lr=lr, weight_decay=0.001
        )  # weight_decay=1e-2)
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
        x = torch.from_numpy(np.array(x)).float().cuda()
        y = self.forward(x).to("cpu").detach().numpy().flatten()
        return y


AEROBIC_ONLY = True
MAX_EXCH_BOUND = 10.0
MAX_BOUND = 1000.0
# Exchange reactions to keep open, from Kenan
# CDM_EXCH_RXNS = [
#     "glc_exch",
#     "ala_exch",
#     "arg_exch",
#     "asp_exch",
#     "asn_exch",
#     "cys_exch",
#     "glu_exch",
#     "gln_exch",
#     "gly_exch",
#     "his_exch",
#     "ile_exch",
#     "leu_exch",
#     "lys_exch",
#     "met_exch",
#     "phe_exch",
#     "pro_exch",
#     "ser_exch",
#     "thr_exch",
#     "trp_exch",
#     "tyr_exch",
#     "val_exch",
#     "ade_exch",
#     "gua_exch",
#     "ura_exch",
#     "4abz_exch",
#     "btn_exch",
#     "fol_exch",
#     "ncam_exch",
#     "NADP_exch",
#     "pnto_exch",
#     "pydx_exch",
#     "pydam_exch",
#     "pydxn_exch",
#     "ribflv_exch",
#     "thm_exch",
#     "Fe2_exch",
#     "Mg2_exch",
#     "Mn2_exch",
#     "Na_exch",
#     "pi_exch",
#     "NO3_exch",
#     "SO4_exch",
#     "H2CO3_exch",
#     "H2O_exch",
#     "CO2_exch",
#     "O2_exch",
#     "O2s_exch",
# "NH4_exch",
# ]
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

        self.model, self.aa_rxns, self.gene_rxns, self.obj_value = self.load_model(
            model_filepath
        )

    def load_model(self, model_path):
        model = cobra.io.read_sbml_model(model_path)
        obj = model.slim_optimize()
        print(obj)
        for rxn in AA_EXCH_RXNS:
            model.reactions.get_by_id(rxn).upper_bound = 0

        print(model.slim_optimize())
        print(model.slim_optimize() / obj)
        print()
        # This is the Error^^^

        return
        # model = tiger.TigerModel.from_cobra(model)

        # model.lb[model.lb > 0] = 0  # remove NGAM
        # for i, rxn in enumerate(model.rxns):
        #     if "_exch" in rxn:
        #         if rxn in CDM_EXCH_RXNS:
        #             model.ub[i] = MAX_EXCH_BOUND
        #         else:
        #             model.ub[i] = 0
        #     if rxn == "H_exch":
        #         model.ub[i] = MAX_BOUND
        #     if rxn == "NH4_exch":
        #         model.ub[i] = MAX_BOUND

        # model, v, g = model.make_base_model(model)

        # # There are two oxygen exchanges; shut them both off with one variable
        # vo = model.addVar(lb=0.0, ub=MAX_EXCH_BOUND, name="oxy_exch")
        # vo1 = model.getVarByName("O2_exch")
        # vo2 = model.getVarByName("O2s_exch")
        # model.addConstr(vo1 <= vo)
        # model.addConstr(vo2 <= vo)

        # gene_list_file = "iSMU_aa_aux_genes.txt"
        # if gene_list_file:
        #     with open(gene_list_file) as f:
        #         genes = f.read().splitlines()
        #         gene_rxns = tiger.get_vars_by_name(model, genes, "MVar")

        # model.Params.OutputFlag = 0  # be quiet
        # model.optimize()
        # full_obj_val = model.ObjVal  # we'll use this later to rescale the data

        # if not AEROBIC_ONLY:
        #     AA_EXCH_RXNS.append("oxy_exch")

        # # The aas are all the input variables, including oxygen
        # aa_rxns = tiger.get_vars_by_name(model, AA_EXCH_RXNS, "MVar")
        # if not AEROBIC_ONLY:
        #     aa_rxns[-1].ub = 0.0
        #     model.optimize()
        #     full_obj_val_anaerobic = model.ObjVal
        # else:
        #     full_obj_val_anaerobic = 0.0

        # print("gene_rxns bounds")
        # print(gene_rxns.ub)

        # print("aa_rxns bounds ")
        # print(aa_rxns.ub)

        # print("Number of targeted genes:", gene_rxns.shape[0])
        # print("Full Objective Value:", full_obj_val)
        # if not AEROBIC_ONLY:
        #     print("Full Objective Value:", full_obj_val_anaerobic, "(anaerobic)")

        # return model, aa_rxns, gene_rxns, full_obj_val

    def get_obj(self, model_inputs):
        """Modify FBA model given gene activations and media
        Return model fitness (objective value normalized to unmodified model's objective value)
        """

        exch_inputs = model_inputs[: len(AA_EXCH_RXNS)]
        # gene_inputs = model_inputs[len(AA_EXCH_RXNS) :]

        # store upper bounds
        stored_exch_ub = np.copy(self.aa_rxns.ub)
        # stored_gene_ub = np.copy(self.gene_rxns.ub)

        # set upper bounds to input values
        # self.aa_rxns.ub = 0
        self.aa_rxns.ub = MAX_EXCH_BOUND * exch_inputs
        # self.gene_rxns.ub = MAX_BOUND * gene_inputs
        self.model.optimize()
        fitness = self.model.ObjVal / self.obj_value
        print(fitness)
        # restore upper bounds
        self.aa_rxns.ub = stored_exch_ub
        # self.gene_rxns.ub = stored_gene_ub

        return fitness


def simple_train(
    model,
    fba_model,
    batch_size,
    n_batches,
    lr=0.01,
    print_status=True,
    save_plots=True,
    compute_test_stats=True,
):

    # reaction_bounds = []
    # for rxn in fba_model.media_reaction_names:
    #     reaction_bounds.append(fba_model.model.reactions.get_by_id(rxn).bounds))

    # return
    scheduler = lr_scheduler.ExponentialLR(model.optimizer, gamma=0.999, verbose=True)

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

    for b in range(n_batches):
        # train_inputs = np.random.choice([0, 1], size=(batch_size, 20))
        train_inputs = np.random.random(size=(batch_size, 20))
        train_labels = np.apply_along_axis(fba_model.get_obj, 1, train_inputs)
        # with multiprocessing.Pool(processes=30) as pool:
        #     train_labels = np.array(pool.map(fba_model.get_obj, train_inputs))

        # Train
        train_inputs = torch.from_numpy(train_inputs).float().cuda()
        train_labels = torch.from_numpy(train_labels).float().cuda().unsqueeze(1)

        # zero the parameter gradients
        model.optimizer.zero_grad()

        # forward + backward + optimize
        train_outputs = model.forward(train_inputs)
        train_loss = model.criterion(train_outputs, train_labels)

        train_loss.backward()
        model.optimizer.step()

        train_mse = model.mse(train_outputs, train_labels)

        train_outputs[train_outputs >= model.threshold] = 1
        train_outputs[train_outputs < model.threshold] = 0
        train_labels[train_labels >= model.threshold] = 1
        train_labels[train_labels < model.threshold] = 0
        train_acc = (train_outputs == train_labels).sum() / train_outputs.shape[0]

        # Train
        if compute_test_stats:
            test_inputs = np.random.random(size=(batch_size, 20))
            test_labels = np.apply_along_axis(fba_model.get_obj, 1, test_inputs)
            # test_inputs = np.random.choice([0, 1], size=(batch_size, 20))
            # test_labels = np.apply_along_axis(fba_model.get_obj, 1, test_inputs)
            # print("test_labels\n", test_labels)
            # with multiprocessing.Pool(processes=30) as pool:
            #     test_labels = np.array(pool.map(fba_model.get_obj, test_inputs))

            test_inputs = torch.from_numpy(test_inputs).float().cuda()
            test_labels = torch.from_numpy(test_labels).float().cuda().unsqueeze(1)

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

        scheduler.step()

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

        if print_status:
            print(
                f"E{b}\t| Train(Loss: {round(tr_l, 4)}, MSE: {round(tr_mse, 4)}, ACC: {round(tr_acc, 4)}) | Test(Loss: {round(te_l, 4)}, MSE: {round(te_mse, 4)}, ACC: {round(te_acc, 4)})"
            )

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


def train(
    model,
    fba_model,
    epochs,
    batch_size,
    n_inputs,
    lr=0.05,
    print_status=True,
    save_plots=True,
    compute_test_stats=True,
):

    scheduler = lr_scheduler.ExponentialLR(model.optimizer, gamma=0.9, verbose=True)

    train_loss_means = []
    train_mse_means = []
    train_acc_means = []
    test_loss_means = []
    test_mse_means = []
    test_acc_means = []
    test_mse_moving_avg = []

    data_inputs = np.random.choice([0, 1], size=(n_inputs, 20))
    # data_labels = np.apply_along_axis(fba_model.get_obj, 1, data_inputs)
    with multiprocessing.Pool(processes=30) as pool:
        data_labels = pool.map(fba_model.get_obj, data_inputs)
        data_labels = np.array(data_labels)

    data_train = Data(data_inputs, data_labels)
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        train_loss_epoch = []
        train_mse_epoch = []
        train_acc_epoch = []
        test_loss_epoch = []
        test_mse_epoch = []
        test_acc_epoch = []

        for train_data in tqdm.tqdm(
            DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
        ):
            train_inputs = train_data[0].cuda()
            train_labels = train_data[1].cuda().unsqueeze(1)
            # train_inputs = np.random.choice([0, 1], size=(batch_size, 20), p=probs)
            # train_labels = np.apply_along_axis(fba_model.get_obj, 1, train_inputs)

            # with multiprocessing.Pool(processes=30) as pool:
            #     train_labels = np.array(
            #         pool.starmap(
            #             fba_model.get_obj,
            #             [(row,) for row in train_inputs],
            #         )
            #     )

            # Train
            # train_inputs = torch.from_numpy(train_inputs).float().cuda()
            # train_labels = (
            #     torch.from_numpy(train_labels).float().cuda().unsqueeze(1)
            # )

            # zero the parameter gradients
            model.optimizer.zero_grad()

            # forward + backward + optimize
            train_outputs = model.forward(train_inputs)
            train_loss = model.criterion(train_outputs, train_labels)

            train_loss.backward()
            model.optimizer.step()

            train_mse = model.mse(train_outputs, train_labels)

            train_outputs[train_outputs >= model.threshold] = 1
            train_outputs[train_outputs < model.threshold] = 0
            train_labels[train_labels >= model.threshold] = 1
            train_labels[train_labels < model.threshold] = 0
            train_acc = (train_outputs == train_labels).sum() / train_outputs.shape[0]

            # Train
            if compute_test_stats:
                test_inputs = np.random.choice([0, 1], size=(batch_size, 20))
                test_labels = np.apply_along_axis(fba_model.get_obj, 1, test_inputs)
                # print("test_labels\n", test_labels)
                # with multiprocessing.Pool(processes=30) as pool:
                #     test_labels = np.array(
                #         pool.starmap(
                #             fba_model.get_obj,
                #             [(row,) for row in test_inputs],
                #         )
                #     )

                test_inputs = torch.from_numpy(test_inputs).float().cuda()
                test_labels = torch.from_numpy(test_labels).float().cuda().unsqueeze(1)

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

        scheduler.step()

        tr_l = sum(train_loss_epoch) / len(train_loss_epoch)
        tr_mse = sum(train_mse_epoch) / len(train_mse_epoch)
        tr_acc = sum(train_acc_epoch) / len(train_acc_epoch)
        te_l = sum(test_loss_epoch) / len(test_loss_epoch)
        te_mse = sum(test_mse_epoch) / len(test_mse_epoch)
        te_acc = sum(test_acc_epoch) / len(test_acc_epoch)

        train_loss_means.append(tr_l)
        train_mse_means.append(tr_mse)
        train_acc_means.append(tr_acc)
        test_loss_means.append(te_l)
        test_mse_means.append(te_mse)
        test_acc_means.append(te_acc)

        moving_subset = test_mse_means[-10:]
        moving_avg_mse = sum(moving_subset) / len(moving_subset)
        test_mse_moving_avg.append(moving_avg_mse)

        if print_status:
            print(
                f"E{epoch}\t| Train(Loss: {round(tr_l, 4)}, MSE: {round(tr_mse, 4)}, ACC: {round(tr_acc, 4)}) | Test(Loss: {round(te_l, 4)}, MSE: {round(te_mse, 4)}, ACC: {round(te_acc, 4)})"
            )

    if save_plots:
        x = range(1, epochs + 1)
        fig, axs = plt.subplots(
            nrows=1, ncols=2, sharex=False, sharey=False, figsize=(10, 5)
        )

        axs[0].plot(x, train_loss_means)
        axs[0].plot(x, train_mse_means)
        axs[0].plot(x, test_loss_means)
        axs[0].plot(x, test_mse_means)
        axs[0].plot(x, test_mse_moving_avg)

        axs[1].plot(x, test_acc_means)
        axs[1].plot(x, train_acc_means)
        axs[0].legend(
            [
                "train_loss",
                "train_mse",
                "test_loss",
                "test_mse",
                "test_mse_moving_avg-10",
            ]
        )
        axs[1].legend(
            [
                "train_acc",
                "test_acc",
            ]
        )
        axs[0].set_xlabel("Epoch")
        axs[1].set_xlabel("Epoch")
        plt.title("NN performance")
        plt.tight_layout()
        plt.savefig("result_training_nn.png")
        plt.close()

    final_moving_avg = test_mse_moving_avg[-1]
    if print_status:
        print(f"Final 10-Moving Avg Epoch MSE: {final_moving_avg}")

    return final_moving_avg


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
    nn_model = COBRAnet2().cuda()
    # fba_model = FBAModel("models/iSMU_rescaled.xml")
    # fba_model = FBAModel("models/iSMU.xml")
    fba_model = FBAModel("models/iSMUv01_CDM.xml")

    # print(fba_model.obj_value)
    # inputs = np.random.random(size=(1000, 20))
    # inputs = np.random.choice([0, 1], size=(1, 20))
    # print(inputs)
    # fitness = np.apply_along_axis(fba_model.get_obj, 1, inputs)
    # print(fitness)

    # # train(
    # #     nn_model,
    # #     fba_model,
    # #     epochs=25,
    # #     lr=0.01,
    # #     batch_size=1024,
    # #     n_inputs=50000,
    # # )

    # simple_train(
    #     nn_model,
    #     fba_model,
    #     lr=0.01,
    #     batch_size=32,
    #     n_batches=2500,
    # )

    # model_path = "cobranet2_alt.pkl"
    # # # model_path = "cobranet2_sm.pkl"

    # # # model_path = "cobranet2.pkl"
    # # # # Final 10-Moving Avg Epoch MSE: 0.0044447677541757
    # torch.save(nn_model, model_path)

    # # entire_set = pd.read_csv(
    # #     "models/SMU_NN_oracle_extrapolated/data_20_extrapolated.csv", index_col=None
    # # ).drop(columns=["grow"])

    # # objs = []
    # # copy = entire_set.to_numpy()
    # # for i in tqdm.trange(len(entire_set)):
    # #     objs.append(fba_model.get_obj(copy[i]))
    # #     # if i % 1000 == 0:
    # #     # print(i)

    # # entire_set["cobra_obj"] = objs
    # # entire_set.to_csv("entire_aa_set_cobra.csv", index=False)

    # entire_set = pd.read_csv(
    #     "entire_aa_set_cobra.csv", index_col=None
    # )  # .iloc[:10000, :]
    # cobra_obj = entire_set["cobra_obj"].to_numpy()
    # entire_set = entire_set.iloc[:, :-1].to_numpy()

    # nn_model = torch.load(model_path).cuda()
    # nn_preds = nn_model.evaluate(entire_set)

    # order = np.argsort(cobra_obj)
    # x = np.arange(len(cobra_obj))

    # fig = plt.figure()

    # plt.plot(
    #     x,
    #     nn_preds[order],
    #     "r.",
    #     label="NN",
    #     alpha=0.01,
    #     markersize=1,
    # )

    # plt.plot(
    #     x,
    #     cobra_obj[order],
    #     "-",
    #     label="iSMU",
    # )

    # fig.legend()
    # fig.tight_layout()
    # plt.savefig("results_nn_v_cobra_lr_simple.png", dpi=400)
