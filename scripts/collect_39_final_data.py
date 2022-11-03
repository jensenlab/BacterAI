import utils
import constants

if __name__ == "__main__":
    experiment_folder = "experiments/2022-04-18_25"
    round_data = utils.combined_round_data(experiment_folder)
    cols = (
        constants.AA_SHORT
        + constants.BASE_NAMES
        + ["ammoniums_50g/l", "round", "frontier_type", "fitness"]
    )
    round_data = round_data[cols]
    round_data["ammoniums_50g/l"] = round_data["ammoniums_50g/l"].apply(int)
    print(round_data.columns)
    print(round_data.shape)
    round_data.to_csv("bacterai_40_cdm_data_SSA_day_9.csv", index=None)