# Data and Models from Dama, et al., 2023.

Each folder contains the data (`experiment_data.csv`) and folders with the trained neural networks for an experiment in the BacterAI paper.

## Abbreviations

 - SGO = *Streptococcus gordonii*
 - SSA = *Streptococcus sanguinis*
 - TL = Transfer learning
 - CDM = Chemically-defined medium

## File format for `experiment_data.csv`

 - The first 20-39 columns describe the ingredients in the media. A `1` indicates the ingredient was present. For amino acid-only studies, all other ingredients are present.
 - The `type` of policy used to find the experiment, either rollout or a random search.
 - Experiments are selected in pairs at the growth front, with one experiment predicted to grow (`frontier_type=FRONTIER`) and another experiment with one fewer ingredient that is predicted to not grow (`frontier_type=BEYOND`).
 - The predicted growth (`growth_pred`) from the model when the experiment was selected, i.e. using a model trained only with previous data. These are not the "best" growth predictions from the final model trained on all the data.
 - The `round` (day) when the experiment was selected.
 - The mean, media, and sample standard deviation fitness of the experiment. Fitness is relative to controls with all ingredients from the same round.
 - A count of ingredients in the media.
