import random
import csv
import copy 
import itertools
import pickle

import numpy as np
import scipy.stats as sp


class Rule(object):
    """DNF Rule Object using only ANDs and ORs.
    
    definition: list[list[int]]
        A list of lists, where the inside lists correspond to the "AND" \
        groups and the outside list corresponds to the "OR" groups. The \
        length of the groups are genereated randomly and depend on the min \
        or max if specified. "AND" groups sample from `range(data_length)` \
        with replacement.
    data_length: int
        The length of data that the rule can accept.
        
    Example 
    -------    
    >>  Rule(5, 5, 10).definition
    >>  [[1,3,2,0],[2,1,4],[0,1,2,3],[0,3,1,2,4]]
    
        It can be interpreted as:
            (1∧3∧2∧0)∨(2∧1∧4)∨(0∧1∧2∧3)∨(0∧3∧1∧2∧4)
        
    """
    def __init__(self, data_length,
                 poisson_mu_OR=None, poisson_mu_AND=None,
                 definition=None):
        """
    
        Inputs
        ------
        data_length: int
            The length of data that the rule can accept.  
        poisson_mu_OR: int
            Mu to use to get a random number from a Poisson distribution
            which will determine the number of OR groups.
        poisson_mu_AND: int
            Mu to use to get a random number from a Poisson distribution
            which will determine the length of an AND groups.
        definition: list[list[int]]
            A user can directly define a definition here.
        
        """
        
        if not definition:
            if not poisson_mu_OR:
                raise NotImplementedError
            if not poisson_mu_AND:
                raise NotImplementedError
            
            number_of_groups = sp.poisson.rvs(poisson_mu_OR)
            while number_of_groups == 0:
                number_of_groups = sp.poisson.rvs(poisson_mu_OR) 
            definition = list()
            for i in range(number_of_groups):
                n = sp.poisson.rvs(poisson_mu_AND)                
                while n == 0 or n >= data_length:
                    n = sp.poisson.rvs(poisson_mu_AND)
                and_group = random.sample(range(data_length), n)
                definition.append(and_group)
        
        self.definition = definition
        self.data_length = data_length
        
    def __str__(self):
        return f"Rule({self.definition}, {self.data_length})"
    
    class Error(Exception):
        """Base class for other exceptions"""
        pass
    
    def evaluate(self, data, use_bool=True):
        """Evaluates the DNF rule with a given a dataset.
    
        Inputs
        ------
        data: np.array[int]
            A list of ones and zeros.
            
        Returns
        -------
            A boolean value from evaluation.
        
        Example 
        -------    
        self.definition = [[1,3,2,0],[2,1,4],[0,1,2,3]]
        evaluate([1, 0, 1, 1, 0]) will return
        the result: True
        
            Solution:
            (3∧2∧0)∨(2∧1∧4)∨(0∧1∧2∧3)
            (T∧T∧T)∨(T∧F∧F)∨(T∧F∧T∧T) = T∨F∨F = T
            
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
            if data.ndim == 1:
                if data.shape[0] != self.data_length:
                    raise DataLengthMismatchError(len(data), self.data_length)
                n_rows = 1
            elif data.ndim == 2:
                if data.shape[1] != self.data_length:
                    raise DataLengthMismatchError(len(data), self.data_length)
                n_rows = data.shape[0]          
            else:
                raise DimensionError(data.shape)
        except DataLengthMismatchError as e:
            print(f"DataLengthMismatchError: {e.message}")
        else:
            
            definition = self.definition
            results = np.empty(n_rows)
            for d in range(n_rows):
                ors = list()
                for or_index, or_ in enumerate(definition):
                    ands = list()
                    for and_index, and_ in  enumerate(or_):
                        if data.ndim == 1:
                            ands.append(data[and_])
                        else:  
                             ands.append(data[d, and_])
                    and_value = np.all(ands)
                    ors.append(and_value)
                result = np.any(ors)
                if not use_bool:
                    result = int(result)
                results[d] = result
            return results
    
    def generate_data_csv(self, output_filename, save_rule=True,
                          quantity=None, repeat=16):
        """Automatically generates inputs and evaluates them. Data is saved
        to a CSV file with the last column being the evaluation result,
        and the subsequent columns being a 1 or 0.
        """
        
        with open(output_filename + ".csv", mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar="\'", 
                                quoting=csv.QUOTE_NONNUMERIC)
            if save_rule:
                self.to_pickle(output_filename + "_rule.pkl")
            if quantity is not None:
                for i in range(quantity):
                    data = np.random.choice(a=[0, 1], 
                                            size=(self.data_length,)).tolist()
                    result = self.evaluate(data, use_bool=False)[0]
                    data.append(result)
                    writer.writerow(data)
            else:
                data = np.array(list(itertools.product((0,1), repeat=repeat)), dtype=np.int8)
                result = self.evaluate(data, use_bool=False)
                print(f"T: {result[result == 1].size}, F: {result[result == 0].size}")
                data = np.hstack((data, np.array([result]).T))
                print(data)
                np.savetxt(file, data, delimiter=',', fmt='%i')
                
    def to_pickle(self, filename):
        """Saves Rule object to pickle file."""
        
        with open(filename, "wb" ) as f: 
            pickle.dump(self, f)
    
    def minimum_rule(self):
        minimum = float('Inf')
        for i in self.definition:
            length = len(i)
            if length < minimum:
                minimum = length
        return minimum
            
    @classmethod
    def from_pickle(cls, filename):
        """Creates Rule object from a pickled Rule."""
        
        with open(filename, "rb" ) as f: 
            rule = pickle.load(f)
        return cls(data_length=rule.data_length, 
                   definition=rule.definition)


if __name__ == "__main__":
    # Testing
    data = np.random.choice(a=[0, 1], size=(2,8))
    rule = Rule(8, poisson_mu_OR=2, poisson_mu_AND=2)
    print(rule.definition)
    print(data)
    print("Should give T/F:", rule.evaluate(data))
    data = np.random.choice(a=[0, 1], size=(4,))
    print("Should give error and None:", rule.evaluate(data))
    