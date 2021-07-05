"""
# GOAL

- [] determine the optimal number of SPSA gradients to avg over, e.g. find the least number of 
samples we need for an (semi-)accurate gradient. 
- [] determine the optimal scaling factor to use

## Plan
- [] perform SPSA using many samples (20 or so)
- [] compare performance to FDSA (collect 2 samples)
- [] determine where we get deminishing returns on gradient accuracy 
- [] repeat using different scaling factors

"""
import argparse
import csv
import logging
import datetime
import math
import os
import sys
import copy

import pandas as pd
import mongoengine
import numpy as np


def set_up_args():
    parser = argparse.ArgumentParser(description="Make CDM for DeepPhenotyping.")

    parser.add_argument(
        "-mi",
        "--mongo_items",
        action="store_true",
        default=False,
        help="Save items (reagents, solutions, stocks, CDM) to MongoDB.",
    )
    parser.add_argument(
        "-me",
        "--make_experiment",
        action="store_false",
        default=True,
        help="Make experiment in MongoDB and export files.",
    )

    parser.add_argument(
        "-n",
        "--make_new",
        action="store_true",
        default=True,
        help="Make new objects instead of assembling them from the database.",
    )

    args = parser.parse_args()


from deepphenotyping import (
    constants,
    ingredients,
    makeids,
    liquid_handlers,
    mapper,
    models,
    scheduling,
    units,
)
from deepphenotyping import utils as dp_utils

from spsa import SPSA


CDM_groups = {
    "amino_acids+NH4": set(
        [
            "dl_alanine",
            "l_arginine",
            "l_aspartic_acid",
            "l_asparagine",
            # "l_cystine",
            "l_cysteine",
            "l_glutamic_acid",
            "l_glutamine",
            "glycine",
            "l_histidine",
            "l_isoleucine",
            "l_leucine",
            "l_lysine",
            "l_methionine",
            "l_phenylalanine",
            "prolines",
            "l_serine",
            "l_threonine",
            "l_tryptophan",
            "l_tyrosine",
            "l_valine",
            "ammoniums",
        ]
    ),
    "vitamins": [
        "paba",
        "biotin",
        "folic_acid",
        "niacinamide",
        "nadp",
        "pantothenate",
        "pyridoxes",
        "riboflavin",
        "thiamine",
        "vitamin_b12",
        "adenine",
        "guanine",
        "uracil",
    ],
    "salts": [
        "iron_nitrate",
        "magnesium_sulfate",
        "manganese_sulfate",
        "sodium_acetate",
        "calcium_chloride",
        "sodium_bicarbonate",
        "potassium_phosphates",
        "sodium_phosphates",
    ],
}


def make_CDM():
    reagents = {}
    stocks = {}
    concentrations = {}
    amino_acid_final_concentrations = {}
    molecular_weights = {}

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
    file_path = os.path.join(
        parent_dir, "files", "CDM_reagents_2x_NH4_tempest_optimized.csv"
    )
    with open(file_path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)

        ## ADDING ALL AMINO ACIDS
        for row in reader:
            # Reading values from CSV
            short_id = row["id"]

            # Skip non-amino acids
            if short_id not in CDM_groups["amino_acids+NH4"]:
                continue
            id_ = f"{short_id}_{row['stock_concentration_x']}x"
            mw = float(row["mw_g_mol"])
            molecular_weights[id_] = mw

            concentrations[id_] = units.parse(row["concentration_g_l"], "g/l")
            amino_acid_final_concentrations[id_] = concentrations[id_].convert(
                "mM", molar_mass=mw
            )

            stock_conc = units.parse(row["stock_concentration_g_l"], "g/l")
            reagent_concentrations = {id_: stock_conc}
            # Creating objects
            # Reagent

            if args.make_new:
                reagents[id_] = ingredients.Reagent(
                    id_=id_, name=row["name"], molecular_weight=mw
                )
            else:
                try:
                    reagent_model = models.ReagentModel.objects.get(
                        id=id_, name=row["name"], molecular_weight=mw
                    )
                    reagents[id_] = ingredients.Reagent.from_mongo(reagent_model)
                except Exception as e:
                    print(
                        "Didn't find {} in reagent database, made new object. Error: {}".format(
                            id_, str(e)
                        )
                    )
                    reagents[id_] = ingredients.Reagent(
                        id_=id_, name=row["name"], molecular_weight=mw
                    )
                    reagents[id_].to_mongo()
                else:
                    print("Retrieved {} from the database.".format(reagents[id_]))

            # Solution
            if args.make_new:
                solution = ingredients.Solution(
                    reagents=reagent_concentrations, id_=id_ + "_stock"
                )
            else:
                try:
                    solution_model = models.SolutionModel.objects.get(id=id_ + "_stock")
                    solution = ingredients.Solution.from_mongo(solution_model)
                except Exception as e:
                    print(
                        "Didn't find {} in solution database, made new object. Error: {}".format(
                            id_, str(e)
                        )
                    )
                    solution = ingredients.Solution(
                        reagents=reagent_concentrations, id_=id_ + "_stock"
                    )
                    solution.to_mongo()
                else:
                    print("Retrieved {} from the database.".format(solution))

            # Stock
            if args.make_new:
                stock = ingredients.Stock(
                    ingredient=solution,
                    type="Solution",
                    date_made=datetime.date.today(),
                    date_expires=datetime.date.today()
                    + datetime.timedelta(6 * 365 / 12),
                    quantity=units.parse("10 ml"),
                    labware=constants.labware[row["labware"]],
                )
            else:
                try:
                    stock_model = models.StockModel.objects.get(
                        ingredient=solution_model, type="Solution"
                    )
                    stock = ingredients.Stock.from_mongo(stock_model)

                except Exception as e:
                    print(
                        "Didn't find {} in stock database, made new object. Error: {}".format(
                            id_, str(e)
                        )
                    )
                    stock = ingredients.Stock(
                        ingredient=solution,
                        type="Solution",
                        date_made=datetime.date.today(),
                        date_expires=datetime.date.today()
                        + datetime.timedelta(6 * 365 / 12),
                        quantity=units.parse("10 ml"),
                        labware=constants.labware[row["labware"]],
                    )
                    stock.to_mongo()
                else:
                    print("Retrieved {} from the database.".format(stock))

            stocks[stock.id] = stock

            if args.mongo_items and args.make_new:
                reagents[id_].to_mongo()
                solution.to_mongo()
                stock.to_mongo()

    with open(file_path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)
        ## MAKING CDM_BASE (CDM without all amino acids)
        reagent_concentrations_CDM_base = {}
        for row in reader:
            # Reading values from CSV
            short_id = row["id"]

            # Skip amino acids
            if short_id in CDM_groups["amino_acids+NH4"]:
                continue
            id_ = f"{short_id}_{row['stock_concentration_x']}x"
            mw = float(row["mw_g_mol"])
            molecular_weights[id_] = mw

            concentrations[id_] = units.parse(row["concentration_g_l"], "g/l")
            stock_conc = units.parse(row["stock_concentration_g_l"], "g/l")
            reagent_concentrations_CDM_base[id_] = stock_conc
            # Creating objects
            # Reagent

        # print(reagent_concentrations_CDM_base, '\n\n')
        mw = 0
        if args.make_new:
            reagents["CDM_base"] = ingredients.Reagent(
                id_="CDM_base", name=row["name"], molecular_weight=mw
            )
        else:
            try:
                reagent_model = models.ReagentModel.objects.get(
                    id="CDM_base", name=row["name"], molecular_weight=mw
                )
                reagents["CDM_base"] = ingredients.Reagent.from_mongo(reagent_model)
            except Exception as e:
                print(
                    "Didn't find {} in reagent database, made new object. Error: {}".format(
                        "CDM_base", str(e)
                    )
                )
                reagents["CDM_base"] = ingredients.Reagent(
                    id_="CDM_base", name=row["name"], molecular_weight=mw
                )
                reagents["CDM_base"].to_mongo()
            else:
                print("Retrieved {} from the database.".format(reagents["CDM_base"]))

        # Solution
        if args.make_new:
            solution = ingredients.Solution(
                reagents=reagent_concentrations_CDM_base, id_="CDM_base" + "_stock"
            )
        else:
            try:
                solution_model = models.SolutionModel.objects.get(
                    id="CDM_base" + "_stock"
                )
                solution = ingredients.Solution.from_mongo(solution_model)
            except Exception as e:
                print(
                    "Didn't find {} in solution database, made new object. Error: {}".format(
                        "CDM_base", str(e)
                    )
                )
                solution = ingredients.Solution(
                    reagents=reagent_concentrations_CDM_base, id_="CDM_base" + "_stock"
                )
                solution.to_mongo()
            else:
                print("Retrieved {} from the database.".format(solution))

        # Stock
        if args.make_new:
            stock = ingredients.Stock(
                ingredient=solution,
                type="Solution",
                date_made=datetime.date.today(),
                date_expires=datetime.date.today() + datetime.timedelta(6 * 365 / 12),
                quantity=units.parse("10 ml"),
                labware=constants.labware[row["labware"]],
            )
        else:
            try:
                stock_model = models.StockModel.objects.get(
                    ingredient=solution_model, type="Solution"
                )
                stock = ingredients.Stock.from_mongo(stock_model)

            except Exception as e:
                print(
                    "Didn't find {} in stock database, made new object. Error: {}".format(
                        "CDM_base", str(e)
                    )
                )
                stock = ingredients.Stock(
                    ingredient=solution,
                    type="Solution",
                    date_made=datetime.date.today(),
                    date_expires=datetime.date.today()
                    + datetime.timedelta(6 * 365 / 12),
                    quantity=units.parse("10 ml"),
                    labware=constants.labware[row["labware"]],
                )
                stock.to_mongo()
            else:
                print("Retrieved {} from the database.".format(stock))

        stocks[stock.id] = stock

        if args.mongo_items and args.make_new:
            reagents["CDM_base"].to_mongo()
            solution.to_mongo()
            stock.to_mongo()

    # for reagent, concentration in concentrations.items():
    #     concentrations.update({reagent : concentration.convert('g/l', molar_mass=molecular_weights[reagent])})
    # CDM
    if args.make_new:
        CDM = ingredients.Solution(concentrations, id_="CDM")
    else:
        try:
            solution_model = models.SolutionModel.objects.get(id="CDM")
            CDM = ingredients.Solution.from_mongo(solution_model)
        except Exception as e:
            print(
                "Didn't find {} in stock database, made new object. Error: {}".format(
                    "CDM", str(e)
                )
            )
            CDM = ingredients.Solution(concentrations, id_="CDM")
            CDM.to_mongo()
        else:
            print("Retrieved {} from the database.".format(CDM))

    if args.mongo_items and args.make_new:
        CDM.to_mongo()

    return reagents, stocks, CDM, amino_acid_final_concentrations, molecular_weights


def scale_NH4_concentrations(original_conc, scale, molecular_weights):
    # scale is a dict (component -> scaling multiplier)
    # returns dict of molar concentrations scaled to compensate for the one removed
    final_conc = pd.DataFrame.from_dict(original_conc, orient="index")
    unit = final_conc.iloc[0, 0].unit_str
    final_conc[0] = final_conc[0].apply(lambda x: x.convert(unit))
    final_conc.insert(1, "value", final_conc[0])
    final_conc.insert(1, "unit", final_conc[0])
    final_conc["value"] = final_conc["value"].apply(lambda x: x.value)
    final_conc["unit"] = final_conc["unit"].apply(lambda x: x.unit_str)
    final_conc = final_conc.drop(0, axis=1)

    sum_conc_change = 0
    for ingredient, s in scale.items():
        orig_conc = final_conc.loc[ingredient].value
        scaled_conc = final_conc.loc[ingredient, "value"] * s
        final_conc.loc[ingredient, "value"] = scaled_conc
        change = scaled_conc - orig_conc
        sum_conc_change += change

    final_conc.loc["ammoniums_50x", "value"] = -sum_conc_change / 2

    zeros = [i for i, val in scale.items() if val == 0]
    final_conc = final_conc.drop(zeros, axis=0)  # DROP IF ANY CONC ARE ZERO

    final_conc.insert(0, "final", pd.Series(dtype=np.str))
    final_conc["final"] = final_conc[["value", "unit"]].apply(
        lambda x: (
            units.parse(x[0], x[1]).convert(
                # "mmol/l",
                # molar_mass=molecular_weights[x.name]
                "g/l",
                molar_mass=molecular_weights[x.name],
            )
        ),
        axis=1,
    )
    new_concentrations = final_conc.to_dict()["final"]

    return new_concentrations


def schedule_FDSA_SPSA(
    cdm,
    stocks,
    amino_acid_final_concentrations,
    mw,
    experiments_csv,
    scaling_factor=0.20,
    output_file="spsa_perturbations.csv",
    use_tempest=False,
):

    experiments = np.genfromtxt(experiments_csv, delimiter=",")

    amino_acid_names = copy.deepcopy(CDM_groups["amino_acids+NH4"])
    amino_acid_names.remove("ammoniums")
    amino_acid_names = list(amino_acid_names)
    n_components = len(amino_acid_names)

    amino_acid_ids = amino_acid_final_concentrations.keys()
    amino_acid_ids = [i for i in amino_acid_ids if i != "ammoniums_50x"]
    solutions = [cdm]

    spsa_perturbations = []
    for e, expt_binary in enumerate(experiments):
        print(f"\n\nExperiments #{e}")

        n_removed = np.where(expt_binary == 0)[0].shape[0]

        ingredients_to_remove = []
        ingredients_to_remove_name = []
        ingredients_remaining = []

        # print(len(expt_binary), len(amino_acid_names))

        for i, include in enumerate(expt_binary):
            if include == 1:
                ingredients_remaining.append(amino_acid_ids[i])
            else:
                ingredients_to_remove.append(amino_acid_ids[i])
                ingredients_to_remove_name.append(amino_acid_names[i])

        media_with_removals = copy.deepcopy(cdm)

        n_remaining = n_components - n_removed
        s = SPSA(W=np.ones(n_remaining))

        fdsa_experiments = s.gen_fdsa_experiments()
        spsa_experiments, perturbations = s.gen_spsa_experiments(n_remaining)
        spsa_perturbations.append((n_remaining, perturbations))
        print("FDSA", len(fdsa_experiments))
        print("SPSA", len(spsa_experiments))
        print("n_removed", n_removed)
        print("n_remaining", n_remaining)
        # print("ingredients_to_remove", ingredients_to_remove)
        # print()
        # print("fdsa_experiments", fdsa_experiments)
        # print("spsa_experiments", spsa_experiments)
        # print(amino_acid_final_concentrations)

        for idx, expt in enumerate(fdsa_experiments):
            expt_plus, expt_minus = expt
            # print(expt_plus, expt_minus)
            # print("before", amino_acid_final_concentrations)

            scaled_conc_plus = {}
            scaled_conc_minus = {}
            for aa, perturb_p, perturb_m in zip(
                ingredients_remaining, expt_plus, expt_minus
            ):
                if perturb_p != 1:
                    scaled_conc_plus[aa] = perturb_p
                if perturb_m != 1:
                    scaled_conc_minus[aa] = perturb_m

            for i in ingredients_to_remove:
                scaled_conc_plus[i] = 0
                scaled_conc_minus[i] = 0

            scaled_conc_plus = scale_NH4_concentrations(
                amino_acid_final_concentrations, scaled_conc_plus, mw
            )
            scaled_conc_minus = scale_NH4_concentrations(
                amino_acid_final_concentrations, scaled_conc_minus, mw
            )

            # print()
            # print("\nafter plus", scaled_conc_plus)
            # print("\nafter minus", scaled_conc_minus)
            # media_concentrations = x  # Calculate
            # print(f"Removing: {ingredients_to_remove}, FDSA remove: {fdsa_remove}")
            # all_remove = fdsa_remove + ingredients_to_remove
            # new_aa_conc = scale_NH4_concentrations(
            #     all_remove, amino_acid_final_concentrations, mw
            # )
            soln_plus = copy.deepcopy(media_with_removals)
            soln_plus = soln_plus.updated(scaled_conc_plus)
            soln_plus = soln_plus.remove_reagents(ingredients_to_remove)
            soln_plus.id = f"EXPT[{e}]_FDSA(+)[{idx}]_CDM -- " + " -- ".join(
                ingredients_to_remove_name
            )
            solutions.append(soln_plus)

            soln_minus = copy.deepcopy(media_with_removals)
            soln_minus = soln_minus.updated(scaled_conc_minus)
            soln_minus = soln_minus.remove_reagents(ingredients_to_remove)
            soln_minus.id = f"EXPT[{e}]_FDSA(-)[{idx}]_CDM -- " + " -- ".join(
                ingredients_to_remove_name
            )
            solutions.append(soln_minus)

        for idx, expt in enumerate(spsa_experiments):
            # length matches number of gradients
            expt_plus, expt_minus = expt

            scaled_conc_plus = {}
            scaled_conc_minus = {}
            for aa, perturb_p, perturb_m in zip(
                ingredients_remaining, expt_plus, expt_minus
            ):
                if perturb_p != 1:
                    scaled_conc_plus[aa] = perturb_p
                if perturb_m != 1:
                    scaled_conc_minus[aa] = perturb_m

            for i in ingredients_to_remove:
                scaled_conc_plus[i] = 0
                scaled_conc_minus[i] = 0

            scaled_conc_plus = scale_NH4_concentrations(
                amino_acid_final_concentrations, scaled_conc_plus, mw
            )
            scaled_conc_minus = scale_NH4_concentrations(
                amino_acid_final_concentrations, scaled_conc_minus, mw
            )

            soln_plus = copy.deepcopy(media_with_removals)
            soln_plus = soln_plus.updated(scaled_conc_plus)
            soln_plus = soln_plus.remove_reagents(ingredients_to_remove)
            soln_plus.id = f"EXPT[{e}]_SPSA(+)[{idx}]_CDM -- " + " -- ".join(
                ingredients_to_remove_name
            )
            solutions.append(soln_plus)

            soln_minus = copy.deepcopy(media_with_removals)
            soln_minus = soln_minus.updated(scaled_conc_minus)
            soln_minus = soln_minus.remove_reagents(ingredients_to_remove)
            soln_minus.id = f"EXPT[{e}]_SPSA(-)[{idx}]_CDM -- " + " -- ".join(
                ingredients_to_remove_name
            )
            solutions.append(soln_minus)

    with open(output_file, "w") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["n_grads", "perturbations"])
        for n, p in spsa_perturbations:
            row = [n]
            row += [" ".join(map(str, x.tolist())) for x in p]
            writer.writerow(row)
    # print(solutions)
    print(len(solutions))

    n_stocks = 6 if use_tempest else 24
    drop_size = "0.2 ul" if use_tempest else "0.1 ul"
    return scheduling.schedule_liquid_handlers(
        solutions,
        constants.strains["SMU"],
        ["aerobic"],
        stocks,
        excess="water",
        plate=constants.labware["WP384"],
        total_volume="80 ul",
        working_volume="78 ul",
        plate_control=("CDM", 1),
        plate_blank=("CDM", 1),
        replicates=3,
        min_drop_size=drop_size,
        max_stocks=n_stocks,
        is_tempest=use_tempest,
    )


def generate_random_experiments(
    n_experiments,
    amino_acid_ids,
    amino_acid_names,
    cdm,
    stocks,
    amino_acid_final_concentrations,
    molecular_weights,
    output_file="experiments_random_R1.csv",
    use_tempest=False,
):
    """
    Generates the base random media with removals to test growth before 
    running SPSA v. FDSA test.
    """

    n_components = len(amino_acid_ids)
    experiments = np.ones((n_experiments, n_components))

    expts_used = set()
    for idx, row in enumerate(experiments):
        while True:
            n_to_remove = np.random.randint(1, n_components + 1)
            print(f"REMOVING: {n_to_remove}")
            indexes_to_remove = np.random.choice(
                range(n_components), size=n_to_remove, replace=False
            )
            index_tup = tuple(sorted(indexes_to_remove.tolist()))
            if index_tup not in expts_used:
                expts_used.add(index_tup)
                break

        experiments[idx, indexes_to_remove] = 0

    with open(output_file, "w") as f:
        np.savetxt(f, experiments, fmt="%i", delimiter=",")

    solutions = [cdm]
    for row in experiments:
        scaled_conc = {}
        removed_ingredient_names = []
        removed_ingredient_ids = []
        for aa_id, name, expt in zip(amino_acid_ids, amino_acid_names, row):
            if expt == 0:
                scaled_conc[aa_id] = 0
                removed_ingredient_names.append(name)
                removed_ingredient_ids.append(aa_id)

        new_conc = scale_NH4_concentrations(
            amino_acid_final_concentrations, scaled_conc, molecular_weights
        )
        new_soln = cdm.without(removed_ingredient_ids)
        new_soln = new_soln.updated(new_conc)
        new_soln.id = f"SPSA(grow_test)_R1_CDM -- " + " -- ".join(
            removed_ingredient_names
        )
        solutions.append(new_soln)

    n_stocks = 6 if use_tempest else 24
    drop_size = "0.2 ul" if use_tempest else "0.1 ul"
    return scheduling.schedule_liquid_handlers(
        solutions,
        constants.strains["SMU"],
        ["aerobic"],
        stocks,
        excess="water",
        plate=constants.labware["WP384"],
        total_volume="80 ul",
        working_volume="78 ul",
        plate_control=("CDM", 1),
        plate_blank=("CDM", 1),
        replicates=3,
        min_drop_size=drop_size,
        max_stocks=n_stocks,
        is_tempest=use_tempest,
    )


def create_experiment(
    is_pre_expt=True,
    expt_name="SPSAvFDSA_1",
    n_experiments=10,
    development=True,
    use_tempest=False,
):
    # use is_pre_expt to test if they grow fine

    if development:
        mongoengine.connect(db="mongo-development", port=27017)
    else:
        mongoengine.connect(db="mongo-production", port=27020)

    (
        CDM_reagents,
        CDM_stocks,
        CDM,
        amino_acid_final_concentrations,
        molecular_weights,
    ) = make_CDM()

    if is_pre_expt:
        expt_name += "_pre_run"
        # Get names and ids of AAs
        amino_acid_names = copy.deepcopy(CDM_groups["amino_acids+NH4"])
        amino_acid_names.remove("ammoniums")
        amino_acid_names = list(amino_acid_names)
        print(amino_acid_names)
        amino_acid_ids = amino_acid_final_concentrations.keys()
        amino_acid_ids = [i for i in amino_acid_ids if i != "ammoniums_50x"]
        print(amino_acid_ids)
        # Generate the pre experiments
        plates, instructions, layout = generate_random_experiments(
            n_experiments,
            amino_acid_ids,
            amino_acid_names,
            CDM,
            list(CDM_stocks.values()),
            amino_acid_final_concentrations,
            molecular_weights,
            output_file="experiments_random_SPSAvFDSA.csv",
            use_tempest=use_tempest,
        )
    else:
        # Generate the experiments
        plates, instructions, layout = schedule_FDSA_SPSA(
            CDM,
            list(CDM_stocks.values()),
            amino_acid_final_concentrations,
            molecular_weights,
            experiments_csv="experiments_random_SPSAvFDSA.csv",
            use_tempest=use_tempest,
        )

    # BioTek read OD manual add
    plate_objects = []
    # biotek_plates => 4 WP384 plates from each DWP96 in plates
    for p in plates:
        # for i in range(1,4):
        plate_objects.append(
            [scheduling.LoadPlate("{}".format(str(p)), constants.labware["WP384"].id_)]
        )
        # bplates.append((scheduling.LoadPlate("{}".format(str(p)), constants.labware['WP384'].id_),
        #                           scheduling.UploadFile(file_id=makeids.unique_id_from_time(length=8))))

    instructions.append(
        scheduling.RunBioTek(
            name="Initial OD",
            plates=plate_objects,
            file=scheduling.UploadFile(file_id="initialOD"),
        )
    )  # initial OD read of WP384

    # Incubate manual add
    # plate_objects = [p[0] for p in plate_objects]
    plate_temps = {}
    for idx, p in enumerate(plates):
        # if idx < len(plates) / 2:
        plate_temps[p] = constants.environments["AE"]
        # else:
        # plate_temps[p] = constants.environments["AN"]

    inc = scheduling.Incubate(
        plates=plate_objects,
        duration=24,
        temperature={plate_id: 37 for plate_id in plates},
        environment=plate_temps,
    )
    instructions.append(inc)

    # Final BioTek read OD manual add
    # instructions.append(scheduling.RunBioTek(plates=biotek_plates, name='Final OD')) # final OD read of WP384
    instructions.append(
        scheduling.RunBioTek(
            name="Final OD",
            plates=plate_objects,
            file=scheduling.UploadFile(file_id="finalOD"),
        )
    )  # initial OD read of WP384

    # create InstructionSet
    instruction_set = scheduling.InstructionSet(
        _id=makeids.unique_id(prefix=expt_name),
        instructions=instructions,
        owner="Adam",
    )

    if args.make_experiment:
        instruction_set.to_mongo()

        if development:
            filepath = os.path.join(
                "/home/lab/Documents/github/DeepPhenotyping/website/data"
            )
        else:
            filepath = os.path.join(
                "/home/lab/DeepPhenotypingServer/DeepPhenotyping/website/data"
            )

        # filepath = os.path.join(settings.BASE_DIR, "data/")
        liquid_handlers.generate_experiment_files(
            instruction_set,
            layout,
            "2 ul",
            path_to_worklists=filepath,
            inoculation_tempest=False,
            dispense_tempest=use_tempest,
        )
        mapper.save_well_map(
            layout,
            instruction_set._id,
            dimensions=constants.labware["WP384"].shape,
            num_split_plates=0,
            path=filepath,
            ot_split_file="files/96_to_384_Mapping_8plate.csv",
        )


if __name__ == "__main__":
    set_up_args()
    development = False

    create_experiment(
        is_pre_expt=True, n_experiments=126, development=development, use_tempest=False
    )

    # create_experiment(is_pre_expt=False, development=development, use_tempest=True)
