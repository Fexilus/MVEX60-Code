"""Implementation of jet spaces and related functions for sympy."""
from itertools import combinations_with_replacement
import operator
from copy import copy, deepcopy

from sympy import Symbol


class JetSpace:
    """A local coordinate representation of a jet space."""
    degree = None
    base_space = []
    fibres = {}

    def __init__(self, base_coord, fibre_coord, degree):
        self.degree = degree

        try:
            self.base_space = list(base_coord)
        except TypeError:
            self.base_space = [base_coord]

        base_size = len(self.base_space)

        try:
            for coordinate in fibre_coord:
                self.fibres[coordinate] = {(0,) * base_size: coordinate}
        except TypeError:
            self.fibres[fibre_coord] = {(0,) * base_size: fibre_coord}

        deriv_tuples = [(0,) * i + (1,) + (0,) * (base_size - i - 1)
                        for i in range(base_size)]

        for d in range(1, degree + 1):
            deriv_index_combs = combinations_with_replacement(deriv_tuples, d)
            deriv_indices = [tuple(map(sum, zip(*tuples)))
                             for tuples in deriv_index_combs]

            deriv_symbols = list(combinations_with_replacement(self.base_space, d))

            for dependent in self.fibres:
                for deriv_index, deriv_symbol in zip(deriv_indices, deriv_symbols):
                    deriv_string = "".join(map(str, deriv_symbol))
                    symbol_name = dependent.name + "_{" + deriv_string + "}"
                    self.fibres[dependent][deriv_index] = Symbol(symbol_name)

    def __deepcopy__(self, memo):
        dcopy = copy(self)
        dcopy.base_space = deepcopy(self.base_space, memo)
        dcopy.fibres = deepcopy(self.fibres, memo)

        return dcopy

    def base_index(self, base_symbol):
        """Returns the derivative index for a coordinate in the base space."""
        base_num = self.base_space.index(base_symbol)
        base_size = len(self.base_space)

        return (0,) * base_num + (1,) + (0,) * (base_size - base_num - 1)

    def extension(self, new_degree):
        """Creates a jet space on the same total space of a higher degree."""
        new_space = deepcopy(self)

        if new_degree > new_space.degree:
            base_size = len(new_space.base_space)

            deriv_tuples = [(0,) * i + (1,) + (0,) * (base_size - i - 1)
                        for i in range(base_size)]

            for d in range(new_space.degree + 1, new_degree + 1):
                deriv_index_combs = combinations_with_replacement(deriv_tuples, d)
                deriv_indices = [tuple(map(sum, zip(*tuples)))
                                for tuples in deriv_index_combs]

                deriv_symbols = list(combinations_with_replacement(new_space.base_space, d))

                for dependent in new_space.fibres:
                    for deriv_index, deriv_symbol in zip(deriv_indices, deriv_symbols):
                        deriv_string = "".join(map(str, deriv_symbol))
                        symbol_name = dependent.name + "_{" + deriv_string + "}"
                        new_space.fibres[dependent][deriv_index] = Symbol(symbol_name)

            new_space.degree = new_degree

        return new_space

    def get_dependents(self):
        """Return the dependent coordinates of the jet space."""
        
        return list(self.fibres)


def total_derivative(jet_exp, coordinate, domain):
    """The total derivative of an expression in a coordinate."""
    codomain = domain.extension(domain.degree + 1)

    diff_jet_exp = jet_exp.diff(coordinate)

    coord_index = domain.base_index(coordinate)

    for dependent, dependent_fibres in domain.fibres.items():
        for fibre_index, fibre_coord in dependent_fibres.items():
            deriv_index = tuple(map(operator.add, fibre_index, coord_index))
            deriv_symbol = codomain.fibres[dependent][deriv_index]
            diff_jet_exp += deriv_symbol * jet_exp.diff(fibre_coord)

    return diff_jet_exp
