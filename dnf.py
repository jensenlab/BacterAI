import random
import csv
import copy
import itertools
import os
import pickle

import numpy as np
import scipy.stats as sp

import utils


class Error(Exception):
    """Base class for other exceptions"""

    pass


class DataLengthMismatchError(Error):
    """
    Raise when the input dimension does not match the rule's
    dimension
    
    Inputs
    ------
    l1: int
        Length of input data.
    l2: int
        Length of data the rule can accept.  
    """

    def __init__(self, l1, l2):
        self.message = f"Input dimension ({l1}) does not match with the rule ({l2})."


class DimensionError(Error):
    """
    Raise when the number of dimensions of the input data is not 1 or 2.
    
    Inputs
    ------
    shape: tuple
        Shape of input data.
    """

    def __init__(self, shape):
        self.message = f"Input data needs to be 1-D or 2-D: {shape}"


class Rule(object):
    """
    DNF Rule using only conjucitions (ANDs) and disjuctions (ORs).
    
    definition: list[list[int]]
        A list of lists, where the inside lists correspond to the "AND"
        groups and the outside list corresponds to the "OR" groups. The
        length of the groups are genereated randomly and depend on the min
        or max if specified. "AND" groups sample from [1, dimension].
        with replacement.
    dimension: int
        The length of data that the rule can accept.
        
    Example 
    -------    
    >>  Rule(5, 5, 10).definition
    >>  [[1,3,2,0],[2,1,4],[0,1,2,3],[0,3,1,2,4]]
    
        It can be interpreted as:
            (1∧3∧2∧0)∨(2∧1∧4)∨(0∧1∧2∧3)∨(0∧3∧1∧2∧4)
        
    """

    def __init__(
        self,
        dimension,
        poisson_mu_OR=None,
        poisson_mu_AND=None,
        definition=None,
        seed=None,
    ):
        """
    
        Inputs
        ------
        dimension: int
            The dimension of the data rule's input.  
        poisson_mu_OR: int, default=None
            Mu to use to get a random variate from a Poisson distribution
            which will determine the number of disjunctions.
        poisson_mu_AND: int, default=None
            Mu to use to get a random variate from a Poisson distribution
            which will determine the length of a conjunction group. If a 
            random variate, N, is chosen <0, N=1. Similarly, if N is chosen 
            and >dimension, N=dimension.
        definition: list[list[int]], default=None
            A user can directly define a definition here.
        seed: Int, default=None
            Seed for generating a numpy.random.RandomState.
        
        """

        self.np_state = np.random.RandomState(
            np.random.MT19937(np.random.SeedSequence(seed))
        )

        if not definition:
            if not poisson_mu_OR:
                raise NotImplementedError
            if not poisson_mu_AND:
                raise NotImplementedError

            definition = []

            # Pick number of disjuctions from a poisson distribution
            number_of_groups = 0
            while number_of_groups == 0:
                number_of_groups = sp.poisson.rvs(
                    poisson_mu_OR, random_state=utils.numpy_state_int(self.np_state)
                )

            # Generate conjuctions from a poisson distribution
            for i in range(number_of_groups):
                n = sp.poisson.rvs(
                    poisson_mu_AND, random_state=utils.numpy_state_int(self.np_state),
                )
                n = min(max(0, n), dimension)  # restrict conjunctions to [1, dimension]

                and_group = self.np_state.choice(
                    range(dimension), size=n, replace=False
                ).tolist()
                definition.append(and_group)

        self.definition = definition
        self.dimension = dimension
        self.minimum_cardinality = self.get_minimum_cardinality()

    def get_definition(self, pretty=True):
        if pretty:
            return f"({')∨('.join(['∧'.join(map(str, ands)) for ands in self.definition])})"
        else:
            return str(self.definition)

    def __str__(self):
        return f"Rule({self.get_definition()}, {self.dimension})"

    def evaluate(self, data, use_bool=False):
        """
        Evaluates the DNF rule with a given a dataset.
    
        Inputs
        ------
        data: np.array[int]
            A list of ones and zeros.
        use_bool: Bool, default=False
            Flag to return boolean result if true, else return float.

        Returns
        -------
            A boolean or float value from evaluation.
        
        Example 
        -------    
        self.definition = [[3,2,0],[2,1,4],[0,1,2,3]]
        evaluate([1, 0, 1, 1, 0]) will return
        the result: True
        
            Solution:
            (3∧2∧0)∨(2∧1∧4)∨(0∧1∧2∧3)
            (T∧T∧T)∨(T∧F∧F)∨(T∧F∧T∧T) = T∨F∨F = T
            
        """

        try:
            if data.ndim == 1:
                if data.shape[0] != self.dimension:
                    raise DataLengthMismatchError(len(data), self.dimension)
                n_rows = 1
            elif data.ndim == 2:
                if data.shape[1] != self.dimension:
                    raise DataLengthMismatchError(len(data), self.dimension)
                n_rows = data.shape[0]
            else:
                raise DimensionError(data.shape)
        except DataLengthMismatchError as e:
            print(f"DataLengthMismatchError: {e.message}")
        else:

            definition = self.definition
            results = np.empty(n_rows)
            for d in range(n_rows):
                ors = []
                for or_index, or_ in enumerate(definition):
                    ands = []
                    for and_index, and_ in enumerate(or_):
                        if data.ndim == 1:
                            ands.append(data[and_])
                        else:
                            ands.append(data[d, and_])
                    and_value = np.all(ands)
                    ors.append(and_value)
                result = np.any(ors)
                if not use_bool:
                    result = float(result)

                results[d] = result

            return results

    def generate_data_csv(
        self, output_path_csv, quantity=None, output_path_rule=None  # , repeat=16
    ):
        """
        Automatically generates inputs and evaluates them. Data is saved
        to a CSV file with the last column being the evaluation result,
        and the other columns being a 1 or 0.

        Returns column names of new data.
        """

        try:
            if quantity is 0:
                raise Exception("quantity must be > 0")
        except Exception as e:
            print("ERROR:", str(e))
        else:
            with open(os.path.join(output_path_csv), mode="a") as file:
                if output_path_rule:
                    self.to_pickle(os.path.join(output_path_rule))
                # writer = csv.writer(
                #     file, delimiter=",", quotechar="'", quoting=csv.QUOTE_NONNUMERIC
                # )
                if quantity is not None:
                    # Ensure no duplicates
                    inputs = np.array([])
                    needed = quantity
                    while needed > 0:
                        new_inputs = self.np_state.choice(
                            a=[0, 1], size=(needed, self.dimension)
                        )
                        stack = (
                            (inputs, new_inputs) if inputs.size != 0 else (new_inputs)
                        )
                        inputs = np.unique(np.vstack(stack), axis=0)
                        needed = quantity - inputs.shape[0]
                else:
                    inputs = np.array(
                        list(itertools.product((0, 1), repeat=self.dimension)),
                        dtype=np.int8,
                    )

                results = self.evaluate(inputs, use_bool=False)
                data_out = np.hstack((inputs, np.array([results]).T))
                column_names = [f"input_{n}" for n in range(self.dimension)] + ["grow"]

                writer = csv.writer(file, delimiter=",")
                writer.writerow(column_names)
                writer.writerows(data_out)

            return column_names

    def to_pickle(self, filename):
        """Saves Rule object to pickle file."""

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def get_minimum_cardinality(self):
        # Minimum is the length of the shortest "and" section
        minimum = min([len(i) for i in self.definition])
        return minimum

    @classmethod
    def from_pickle(cls, filename):
        """Creates Rule object from a pickled Rule."""

        with open(filename, "rb") as f:
            rule = pickle.load(f)
        return cls(dimension=rule.dimension, definition=rule.definition)


if __name__ == "__main__":
    # Testing
    data = np.random.choice(a=[0, 1], size=(2, 8))
    rule = Rule(8, poisson_mu_OR=2, poisson_mu_AND=2)
    print(rule.definition)
    print(data)
    print("Should give T/F:", rule.evaluate(data))
    data = np.random.choice(a=[0, 1], size=(4,))
    print("Should give error and None:", rule.evaluate(data))
