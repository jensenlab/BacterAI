import copy
import csv
import datetime
import time
import itertools
import math 
import multiprocessing
import os
import random

import cobra
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
from tqdm import tqdm, trange

CDM_RXN_IDS = [
    "glc_exch",
    "ac_exch",
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
    # "pydx_pydam_exch",
    "ribflv_exch",
    "thm_exch",
    # "vitB12_exch",
    # "FeNO3_exch",
    # "MgSO4_exch",
    # "MnSO4_exch",
    # "CaCl2_exch",
    # "NaBic_exch",
    # "KPi_exch",
    # "NaPi_exch",
    # "H2O_exch",
    # "CO2_exch",
    # "H_exch",
    # "O2_exch",
    # "O2s_exch",
]

# CDM_RXN_IDS = [
#     "ac_exch",    
#     "ala_exch",
#     "arg_exch",
#     "asp_exch",
#     "asn_exch",
#     "cys_exch",
#     "glc_exch",
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
#     "ribflv_exch",
#     "thm_exch",
#     "vitB12_exch,
#     "pydx_pydam_exch",
#     "FeNO3_exch",
#     "MgSO4_exch",
#     "MnSO4_exch",
#     "CaCl2_exch",
#     "NaBic_exch",
#     "KPi_exch",
#     "NaPi_exch"
# ]

class Model():
    def __init__(self, path, num_components, new_data=False):
        self.model_path = path
        self.num_components = num_components
        self.model = self.load_cobra(self.model_path)
        self.growth_cutoff = self.get_growth_cutoff()
        self.media_ids = self.get_media_ids(new_data)
        self.minimal_components = self.get_minimal_components()
        
    def reload_model(self):
        self.model = self.load_cobra(self.model_path)
        
    def load_cobra(self, model_path):
        """Load a CobraPy model.
        
        model_path: str
            Path to model.
        """ 
        model = cobra.io.read_sbml_model(model_path)
        return model
    

    def random_reactions(self, num_to_remove=5):
        # num_to_remove - int: number of reactions to remove (set to 0)
        
        remove_indexes = np.random.choice(len(CDM_RXN_IDS), 
                                          num_to_remove, replace=False)
        remove_arr = np.ones(len(CDM_RXN_IDS))
        remove_arr[remove_indexes] = 0
        return reactions_to_knockout(remove_arr, CDM_RXN_IDS), remove_arr


    # def knockout_and_simulate(self, num_to_remove, return_boolean=False):
    #     min_growth = 0.50 * self.model.slim_optimize()
    #     reactions, remove_arr = self.random_reactions(num_to_remove)
    #     grow, objective_value = self.reaction_knockout(reactions, min_growth)
    #     if return_boolean:
    #         return int(grow), remove_arr
    #     return objective_value, remove_arr
        

    # def reactions_to_knockout(self, remove_arr, reactions):
    #     # remove_arr - np.array[int]: binary array (0 = remove, 1 = keep)
    #     # reactions - [str]: list of reactions
        
    #     ones = np.where(remove_arr == 1)[0]
    #     reactions = np.delete(reactions, ones)
    #     return reactions
    
    # def reaction_knockout_cobra(self, reactions, growth_cutoff, dummy=None,
    #                             use_media=False, use_names=True):
    #     # model - cobrapy.Model: model with reactions to knockout
    #     # reactions - [str]: list of reactions to knockout
    #     # growth_cutoff - float: grow/no grow cutoff
        
    #     # model = copy.deepcopy(model)
    #     if dummy:
    #         self.model.add_reactions([dummy_rxn])

    #     if use_media:
    #         with self.model as m:
    #             medium = m.medium
    #             for reaction in reactions:
    #                 medium[reaction] = 0.0
    #             self.medium = medium
    #             objective_value = m.slim_optimize()
    #     else:
    #         reaction_bounds = dict()
    #         for r in reactions:
    #             if use_names:
    #                 reaction_bounds[r] = (
    #                     self.model.reactions.get_by_id(r).bounds)
    #                 self.model.reactions.get_by_id(r).bounds = (0,0)
    #             else:
    #                 reaction_bounds[r] = r.bounds
    #                 r.bounds = (0,0)
    #         objective_value = self.model.slim_optimize()
    #         for r, bounds in reaction_bounds.items():
    #             if use_names:
    #                 self.model.reactions.get_by_id(r).bounds = bounds
    #             else:
    #                 r.bounds = bounds
                    
        
    #     grow = False if objective_value < growth_cutoff else True
        
    #     if dummy:
    #         self.model.reactions.get_by_id(dummy.name).remove_from_model()
        
    #     return objective_value, grow


    def print_compartments(self):
        print(self.model.compartments)
        
        print("############## BOUNDARY ##############")
        for rxn in self.model.boundary:
            print(rxn, rxn.bounds)
        print("\n\n")
        print("############## EXCHANGE ##############")
        for rxn in self.model.exchanges:
            print(rxn, rxn.bounds)
        print("\n\n")
        print("############## SINKS ##############")
        for rxn in self.model.sinks:
            print(rxn, rxn.bounds)
        print("\n\n")
        print("############## DEMANDS ##############")
        for rxn in self.model.demands:
            print(rxn, rxn.bounds)

    def benchmark(self, n=1000):
        start = time.time()
        with self.model as m:
            for _ in trange(n):
                r = random.randint(1, 3)
                reactions = random.sample(m.reactions, r)
                self.reaction_knockout(reactions, 0)
                # m.slim_optimize()
        print(f"Finished in {time.time() - start} seconds.")
        
    def reaction_knockout(self, reactions, min_growth):
        """Knock outs reactions in a given model.
        
        Inputs
        ------
        model: cobrapy.Model
            Model to use when knocking out reactions.
        reactions: list[cobrapy.Reaction]
            Reactions in `model` to be knocked out.
        min_growth: float
            Threshold value for determining growth.
        
        Returns
        -------
        grow: boolean
            Returns `True` if the model's objective value after knocking out 
            reactions is larger that the `min_growth` threshold, and `False`
            otherwise.
        objective_value: float
            The objective value of `model` after knocking out the reactions.
            
        """
        
        with self.model as m:
            for rxn in reactions:
                rxn.knock_out()
                # print(f"KO --> {rxn.id}")
            objective_value = m.slim_optimize()
            # print("OBJ:", objective_value)
        grow = True if objective_value > min_growth else False
        return grow, objective_value


    def get_number_knocked_out(self):
        """Return number of reactions knocked out in a given model
        """
        return sum(
            [1 if rxn.bounds == (0,0) else 0 for rxn in self.model.reactions])


    def knockout_walk(self, valid_reactions, growth_threshold=0.50):
        """Performs a 'knockout walk' in which a random number of reactions are
        removed one at a time. Each cycle of the walk, a list of 'candidate
        reactions' is built, i.e. all of the reactions in `valid_reactions` that 
        are predicted to still lead to growth in `model` when removed. A single 
        random reaction is chosen from the candidates and is knocked out. The 
        cycle repeats.
        
            
        Inputs
        ------
        model: cobrapy.Model
            Model to use when knocking out reactions.
        valid_reactions: list[cobrapy.Reaction]
            Reactions in `model` to be knocked out.
        growth_threshold: float, default: 0.50
            Threshold percentage for determining growth.
        
        Returns
        -------
        valid_reactions: list[cobrapy.Reaction]
            A list of reactions which is a subset of `valid_reactions` and can be 
            knocked out in the future. Reactions that have been knocked out in this
            walk are the ones that have been removed.
        valid_reactions: list[cobrapy.Reaction]
            A list of reactions that were knocked out.
        """
        
        max_objective = self.model.slim_optimize()
        growth_cutoff = growth_threshold * max_objective
        
        num_knockouts = sp.poisson.rvs(5)
        print("Number of KOs:", num_knockouts)
        removed_reactions = list()
        for _ in range(num_knockouts):
            candidate_reactions = list()
            for rxn in valid_reactions:
                does_grow, _ = self.reaction_knockout([rxn], growth_cutoff)
                if does_grow:
                    candidate_reactions.append(rxn)
            
            reaction_to_remove = random.choice(candidate_reactions)
            self.model.reactions.get_by_id(reaction_to_remove.id).knock_out()
            # reaction_to_remove.knock_out()
            valid_reactions.remove(reaction_to_remove)
            removed_reactions.append(reaction_to_remove)
            print(f"\tREMOVED: {reaction_to_remove.id}, # KO'd: {self.get_number_knocked_out()}, # Valid RXNs: {len(valid_reactions)}")
        
        return valid_reactions, removed_reactions


    def get_non_media_reactions(self):
        """Return a list of all reactions in `model` that are not the media 
        (CDM) reactions
        """
        
        with self.model as m:
            all_reactions = set(m.reactions)
            CDM_reactions = list()
            for id_ in CDM_RXN_IDS:
                if m.reactions.has_id(id_):
                    CDM_reactions.append(m.reactions.get_by_id(id_))
            non_media_reactions = all_reactions.difference(CDM_reactions)
        return non_media_reactions
    
    
    def make_minimal_media_models(self, max_n=10):
        """Generate and save derivations of a CobraPy model (at location 
        `model_path`) and their respective minimal medias. 
        
        Each cycle, a 'knockout walk' is performed and a minimal media is calculated. 
        The cycle repeats, knocking out more reactions and saving a new derivation 
        of current model with its minimal media, only when the length of the 
        minimal media increases (as to not have many model derivations of the same
        length minimal medias). When critical reactions are eventually removed, the 
        current model derivation cannot grow. When this happens, then the model is 
        reset to its original state and the cycle continues until `max_n` is 
        reached.
        
        Inputs
        ------
        model_path: str
            Path to cobrapy.Model to use.
        max_n: int, default: 10
            The number of new model derivation to generate and save.
        
        Outputs
        -------
        CobraPy model derivations, along with a CSV listing their minimal media
        components and the reactions removed. They are stored in the parent folder
        of the parent model in a new directory structure:
            > `model_path` parent folder
                > '[model name]_[timestamp]'
                    > length of minimal media
                        > '[model name]_[length of minimal media].xml'
                        > '[model name]_[length of minimal media]_info.csv'
                        
        Returns
        -------
        enclosing_folder: str
            The newly generated folder where all files are saved in.
        """
        
        max_objective = self.model.slim_optimize()
        print("Growth threshold:", self.growth_cutoff)
        
        parent_folder = "/".join(self.model_path.split("/")[:-1])
        timestamp = datetime.datetime.now().isoformat(sep='T', 
                                                    timespec='milliseconds')
        model_name = (self.model_path.split("/")[-1]).split(".")[0]
        enclosing_folder = os.path.join(parent_folder, f"{model_name}_{timestamp}")
        
        
        valid_reactions = self.get_non_media_reactions()
        removed_reactions = list()
        max_length_media = 0
        for _ in range(max_n):
            try:
                valid_reactions, removed = self.knockout_walk(valid_reactions)
                removed_reactions += removed
                minimal_medium = cobra.medium.minimal_medium(
                    self.model, self.growth_cutoff, minimize_components=True)
                current_length_media = len(minimal_medium)
            except Exception as e:
                print("\nModel Failed: {}".format(str(e)))
                print("Resetting model...")
                max_length_media = 0
                self.reload_model()
                valid_reactions = self.get_non_media_reactions()
                removed_reactions = list()
            else:
                if current_length_media > max_length_media:
                    print("\n###########  Found New Minimum!  ############")
                    max_length_media = current_length_media
                    # reactions.append((current_length_media, None))
                    folder = os.path.join(enclosing_folder,
                                          str(max_length_media))
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    new_model_path = (
                        os.path.join(folder, 
                            f"{model_name}_{max_length_media}.xml"))
                    cobra.io.write_sbml_model(self.model, new_model_path)
                
                    new_csv_path = (
                        os.path.join(folder, 
                            f"{model_name}_{max_length_media}_info.csv"))
                    with open(new_csv_path, 'w') as file:
                        writer = csv.writer(file, delimiter=',')
                        writer.writerow(['Minimal Media', 'Removed Reactions'])
                        reaction_ids = [r.id for r in removed_reactions]
                        rows = itertools.zip_longest(
                            list(minimal_medium.index), reaction_ids)
                        writer.writerows(rows)
                print("Minimal media length:", len(minimal_medium))
                print("Media:", list(minimal_medium.index))
                
        return new_model_path
    
    def get_growth_cutoff(self, k=0.30):
        return k*self.model.slim_optimize()

    
    def get_minimal_components(self):
        minimal_medium = cobra.medium.minimal_medium(
            self.model, self.growth_cutoff, minimize_components=True,
            media_components=self.media_ids)
        reaction_ids = [rxn.id for rxn in self.model.exchanges]
        non_media_exchanges = set(reaction_ids) - set(self.media_ids)
        components_of_interest = set(minimal_medium.index) - non_media_exchanges
        return components_of_interest
    
    class Error(Exception):
        """Base class for other exceptions"""
        pass
    
    def _f(self, row):
        # L, row, t = args
        reactions_to_knockout = list()
        for idx, val in enumerate(row):
            if val == 0:
                reactions_to_knockout.append(
                    self.model.reactions.get_by_id(
                        self.media_ids[idx]))
        result = int(
            self.reaction_knockout(reactions_to_knockout, 
                                    self.growth_cutoff)[0])
        return result
        # combined = np.append(row, result)
        # L.append(combined)
        # t.update()
            
    def evaluate(self, inputs, use_bool=True, use_multiprocessing=True):
        """Evaluates the model with a given media input.
        """
        class DataLengthMismatchError(self.Error):
            """Raised when the input data length does not match the rule's
            data length
            
            Inputs
            ------
            l1: int
                Length of input data.
            l2: int
                Length of data the rule can accept.  
            """
            def __init__(self, l1, l2):
                self.message = (
                    f"Input data length ({l1}) does not match with the rule ({l2}).")
        
        class DimensionError(self.Error):
            """Raised when the number of dimensions of the input data is not 1 or 2.
            
            Inputs
            ------
            shape: tuple
                Shape of input data.
            """
            def __init__(self, shape):
                self.message = (f"Input data needs to be 1-D or 2-D: {shape}")

    
        try:
            if inputs.ndim == 1:
                n_rows = 1
            elif inputs.ndim == 2:
                n_rows = inputs.shape[0]          
            else:
                raise DimensionError(inputs.shape)
        except DataLengthMismatchError as e:
            print(f"DataLengthMismatchError: {e.message}")
        else:
            if use_multiprocessing:
                # results = np.array(p_map(self._f, [row for row in inputs]))
                # print(results)
                # return results
                with multiprocessing.Pool(processes=30) as p:
                    results = p.map(self._f, inputs)
                    results = np.array(results)
                    return results
            else: 
                results = np.empty(n_rows)
                for i in trange(n_rows):                
                    reactions_to_knockout = list()
                    for idx, val in enumerate(inputs[i]):
                        if val == 0:
                            reactions_to_knockout.append(
                                self.model.reactions.get_by_id(
                                    self.media_ids[idx]))
                    result = int(
                        self.reaction_knockout(reactions_to_knockout, 
                                            self.growth_cutoff)[0])
                    if not use_bool:
                        result = int(result)
                    results[i] = result
                return results
        
    def get_media_ids(self, new_data):
        data_path = os.path.join(os.path.split(self.model_path)[0], 
                                 f'data_{self.num_components}.csv')
    
        if os.path.exists(data_path) and not new_data:
            with open(data_path, mode='r') as file:
                header = file.readline().split(',')[:-1]
                return header
        else:
            reactions = list()
            for rxn in CDM_RXN_IDS:
                if self.model.reactions.has_id(rxn):
                    reactions.append(rxn)
            # TODO: Check if num_components is <= number of CDM_RXN_IDS
            return random.sample(reactions, self.num_components)
    
    def generate_data_csv(self, use_multiprocessing=False):
        """Automatically generates inputs and evaluates them. Data is saved
        to a CSV file with the last column being the evaluation result,
        and the subsequent columns being a 1 or 0.
        """
        output_filename = os.path.join(os.path.split(self.model_path)[0], 
                                       f'data_{self.num_components}.csv')
        # if os.path.exists(output_filename):
        #     raise FileExistsError
        
        if self.num_components > len(CDM_RXN_IDS):
            raise NotImplementedError

        with open(output_filename, mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar="\'", 
                                quoting=csv.QUOTE_MINIMAL)
            writer.writerow(self.media_ids + ["growth"])
            data = np.array(
                list(itertools.product(
                    (0,1), repeat=self.num_components)), dtype=np.int8)
            results = self.evaluate(data, use_multiprocessing=use_multiprocessing)
            
            print(f"T: {results[results == 1].size}, F: {results[results == 0].size}")
            data = np.hstack((data, np.array([results]).T))
            np.savetxt(file, data, delimiter=',', fmt='%i')

if __name__ == "__main__":
    # model = load_cobra("models/iSMUv01_CDM_LOO_v2.xml")
    
    # for rxn in model.reactions:
    #     if "CDM_exch" in rxn.id
    
    #         print(rxn.id)            
    #         rxn.id = rxn.id[:-8] + "media_exch"
    #     print(rxn.id)
            
    # cobra.io.write_sbml_model(
    #     model, "models/iSMUv01_CDM_LOO_v2.xml")

    # bench()
    m = Model("models/iSMUv01_CDM.xml", 16)
    for rxn in m.model.reactions:
        if "exch" in rxn.id:
            print(f"\"{rxn.id}\",", rxn)
    # m.print_compartments()
    # # m.benchmark(n=1000000)
    # folder_name = m.make_minimal_media_models(max_n=1)
    # print(f"DONE!: {folder_name}")