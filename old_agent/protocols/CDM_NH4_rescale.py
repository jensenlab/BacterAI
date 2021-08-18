import csv
import logging
import datetime
import argparse
import os
import sys
import copy
import pandas as pd

import mongoengine

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

# parser = argparse.ArgumentParser(description="Make CDM for DeepPhenotyping.")
# parser.add_argument(
#     "-mi",
#     "--mongo_items",
#     action="store_true", default=False,
#     help="Save items (reagents, solutions, stocks, CDM) to MongoDB.",
# )
# parser.add_argument(
#     "-me", "--mongo_experiment", action="store_true", default=True,
#     help="Save experiment to MongoDB."
# )
# parser.add_argument(
#     "-e",
#     "--export",
#     action="store_true", default=True,
#     help="Export all worklist.csv and zip folder of all worklists.",
# )
# parser.add_argument(
#     "-n",
#     "--make_new",
#     action="store_true", default=True,
#     help="Make new objects instead of assembling them from the database.",
# )
# args = parser.parse_args()

mongo_items = False
mongo_experiment = export = False
make_new = True
use_tempest = True

CDM_groups = {
    "amino_acids+NH4": set(
        [
            "dl_alanine",
            "l_arginine",
            "l_aspartic_acid",
            "l_asparagine",
            "l_cystine",
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
            #  'l_proline',
            #  'hydroxy_l_proline',
            "l_serine",
            "l_threonine",
            "l_tryptophan",
            "l_tyrosine",
            "l_valine",
            "ammoniums",
        ]
    ),
    "vitamins": set(
        [
            "paba",
            "biotin",
            "folic_acid",
            "niacinamide",
            "nadp",
            "pantothenate",
            "pyridoxes",
            #  'pyridoxal',
            #  'pyridoxamine',
            "riboflavin",
            "thiamine",
            "vitamin_b12",
            "adenine",
            "guanine",
            "uracil",
        ]
    ),
    "salts": set(
        [
            "iron_nitrate",
            #  'iron_sulfate',
            "magnesium_sulfate",
            "manganese_sulfate",
            "sodium_acetate",
            "calcium_chloride",
            "sodium_bicarbonate",
            "potassium_phosphates",
            "sodium_phosphates",
            #  'potassium_phosphate',
            #  'potassium_phosphate_mono',
            #  'sodium_phosphate',
            #  'sodium_phosphate_mono'
        ]
    ),
}


def make_CDM():
    reagents = {}
    stocks = {}
    concentrations = {}
    amino_acid_final_concentrations = {}
    molecular_weights = {}

    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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

            if make_new:
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
            if make_new:
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
            if make_new:
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

            if mongo_items and make_new:
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
        if make_new:
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
        if make_new:
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
        if make_new:
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

        if mongo_items and make_new:
            reagents["CDM_base"].to_mongo()
            solution.to_mongo()
            stock.to_mongo()

    # for reagent, concentration in concentrations.items():
    #     concentrations.update({reagent : concentration.convert('g/l', molar_mass=molecular_weights[reagent])})
    # CDM
    if make_new:
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

    if mongo_items and make_new:
        CDM.to_mongo()

    return reagents, stocks, CDM, amino_acid_final_concentrations, molecular_weights


def scale_NH4_concentrations(to_remove, final_conc_in, molecular_weights):
    # returns dict of molar concentrations scaled to compensate for the one removed
    final_conc = pd.DataFrame.from_dict(final_conc_in, orient="index")
    unit = final_conc.iloc[0, 0].unit_str
    final_conc[0] = final_conc[0].apply(lambda x: x.convert(unit))
    final_conc.insert(1, "value", final_conc[0])
    final_conc.insert(1, "unit", final_conc[0])
    final_conc["value"] = final_conc["value"].apply(lambda x: x.value)
    final_conc["unit"] = final_conc["unit"].apply(lambda x: x.unit_str)
    final_conc = final_conc.drop(0, axis=1)
    to_remove = utils.assert_list(to_remove)

    # print(to_remove)
    removed_conc = sum([final_conc.loc[r].value for r in to_remove])
    final_conc.loc["ammoniums_50x", "value"] = removed_conc / 2
    final_conc = final_conc.drop(to_remove, axis=0)
    final_conc.insert(0, "final", pd.Series())
    final_conc["final"] = final_conc[["value", "unit"]].apply(
        lambda x: (
            units.parse(x[0], x[1]).convert("g/l", molar_mass=molecular_weights[x.name])
        ),
        axis=1,
    )
    new_concentrations = final_conc.to_dict()["final"]
    return new_concentrations


def schedule_CDM_l2o(
    CDM, stocks, batch_removals, amino_acid_final_concentrations, molecular_weights
):
    solutions = [CDM]

    for components in batch_removals:
        print(f"Removing: {components}")
        new_conc = scale_NH4_concentrations(
            components, amino_acid_final_concentrations, molecular_weights
        )
        new = CDM.without(components)
        new = new.updated(new_conc)
        solutions.append(new)

    print("# of solutions:", len(solutions))

    n_stocks = 6 if use_tempest else 24
    drop_size = "0.2 ul" if use_tempest else "0.1 ul"

    cdm_base_id = [s.id for s in stocks if s.id.startswith("CDM_base_stock")]
    if use_tempest:
        ammoniums_id = [s.id for s in stocks if s.id.startswith("ammoniums")]
        override_sets = [["water"], cdm_base_id, ammoniums_id]
    else:
        override_sets = None

    return scheduling.schedule_liquid_handlers(
        solutions,
        [constants.strains["STH"]],
        ["aerobic", "anaerobic"],
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
        override_stock_sets=override_sets,
        is_tempest=use_tempest,
        manual_fill_stocks=["water", cdm_base_id[0]],
        alphabetical_stocks=True,
    )


def from_batch_list(batch_name, batch_removals, development=True):
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

    scheduling.logger.setLevel(logging.INFO)
    plates, instructions, layout = schedule_CDM_l2o(
        CDM,
        list(CDM_stocks.values()),
        batch_removals,
        amino_acid_final_concentrations,
        molecular_weights,
    )

    # DWP96 pipetting
    # plates, instructions, layout = schedule_CDM_l2o(CDM, list(CDM_stocks.values())) # DWP96 pipetting
    # OpenTrons manual add
    # slots = {
    #     "1": constants.labware["WP384"].id_,
    #     "2": constants.labware["WP384"].id_,
    #     "3": "empty",
    #     "4": "empty",
    #     "5": "empty",
    #     "6": constants.labware["dWP96_2ml"].id_,
    #     "7": "empty",
    #     "8": "empty",
    #     "9": "empty",
    #     "10": "tip",
    #     "11": "empty",
    # }

    # deck = scheduling.LoadDeck(deck_slots=slots, pipette_type={'left': 'p300multi', 'right': None})
    # plates = [[scheduling.LoadPlate(str(p), constants.labware['dWP96_2ml'].id_)] for p in plates]
    # protocol = scheduling.RunProtocol(file_id=makeids.unique_id_from_time(length=8))
    # ot = scheduling.RunOpenTrons(load_deck=deck, plates=plates, run_protocol=protocol)
    # instructions.append(ot)

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
        _id=makeids.unique_id(prefix=batch_name),
        instructions=instructions,
        owner="Adam",
    )

    if mongo_experiment:
        instruction_set.to_mongo()

    if export:
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
            instruction_set, layout, "2 ul", path_to_worklists=filepath
        )
        mapper.save_well_map(
            layout,
            instruction_set._id,
            dimensions=constants.labware["WP384"].shape,
            num_split_plates=0,
            path=filepath,
            ot_split_file="files/96_to_384_Mapping_8plate.csv",
        )
