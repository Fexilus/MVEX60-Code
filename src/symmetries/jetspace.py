"""Implementation of jet spaces and related functions for sympy."""
from itertools import combinations_with_replacement
import operator
from copy import copy, deepcopy

from sympy import Symbol

from .utils import iter_wrapper


class JetSpace:
    """A local coordinate representation of a jet space."""

    def __init__(self, base_coord, fibre_coord, degree):

        self.degree = degree

        self.base_space = list(iter_wrapper(base_coord))

        base_size = len(self.base_space)

        self.fibres = {}
        for coordinate in iter_wrapper(fibre_coord):
            self.fibres[coordinate] = {(0,) * base_size: coordinate}

        self._add_jet_fibers(degree)


    def __deepcopy__(self, memo):

        dcopy = copy(self)
        dcopy.base_space = deepcopy(self.base_space, memo)
        dcopy.fibres = deepcopy(self.fibres, memo)

        return dcopy


    def base_index(self, base_symbol):
        """Returns the derivative index for a coordinate in the base
        space.
        """
        base_num = self.base_space.index(base_symbol)
        base_size = len(self.base_space)

        return (0,) * base_num + (1,) + (0,) * (base_size - base_num - 1)


    def extension(self, new_degree):
        """Creates a jet space on the same total space of a higher
        degree.
        """
        new_space = deepcopy(self)

        if new_degree > new_space.degree:
            new_space._add_jet_fibers(new_space.degree + 1, new_degree)
            new_space.degree = new_degree
        else:
            raise ValueError("Extension must be to higher degree")

        return new_space


    def get_dependents(self):
        """Return the dependent coordinates of the jet space."""

        return list(self.fibres)


    def _add_jet_fibers(self, upper_degree, lower_degree=1):
        """Internal function for creating the jet coordinates."""

        base_size = len(self.base_space)

        deriv_tuples = [(0,) * i + (1,) + (0,) * (base_size - i - 1)
                        for i in range(base_size)]

        for d in range(lower_degree, upper_degree + 1):
            deriv_index_combs = combinations_with_replacement(deriv_tuples, d)
            deriv_indices = [tuple(map(sum, zip(*tuples)))
                             for tuples in deriv_index_combs]

            deriv_symbols = list(combinations_with_replacement(self.base_space,
                                                               d))

            for dependent in self.fibres:
                for deriv_index, deriv_symbol in zip(deriv_indices,
                                                     deriv_symbols):
                    deriv_string = "".join(map(str, deriv_symbol))
                    symbol_name = dependent.name + "_{" + deriv_string + "}"
                    self.fibres[dependent][deriv_index] = Symbol(symbol_name)

    @property
    def original_total_space(self):
        """Return the coordinates of the total space on which the jet 
        space is built

        Returns:
            A 2-tuple of lists of the coordinates of the base space and
            fibre respectively.
        """

        return self.base_space, self.get_dependents()


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
