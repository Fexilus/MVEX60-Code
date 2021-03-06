"""Implementation of jet spaces and related functions for sympy."""
from itertools import combinations_with_replacement
import operator
from copy import copy, deepcopy

from sympy import Symbol

from .utils import iter_wrapper


class JetSpace:
    """A local coordinate representation of a jet space.

    :param base_coord: The basis vectors of the base space.
    :type base_coord: list[:class:`sympy.Expr`]

    :param fiber_coord: The basis vectors of the fibers.
    :type fiber_coord: list[:class:`sympy.Expr`]

    :param degree: The degree of the created jet space. A degree of 0
        corresponds to the total space.
    :type degree: int
    """

    def __init__(self, base_coord, fiber_coord, degree):

        self.degree = degree

        self.base_space = list(iter_wrapper(base_coord))

        base_size = len(self.base_space)

        self.fibers = {}
        for coordinate in iter_wrapper(fiber_coord):
            self.fibers[coordinate] = {(0,) * base_size: coordinate}

        self._add_jet_fibers(degree)


    def __deepcopy__(self, memo):

        dcopy = copy(self)
        dcopy.base_space = deepcopy(self.base_space, memo)
        dcopy.fibers = deepcopy(self.fibers, memo)

        return dcopy


    def base_index(self, base_symbol):
        """Returns the derivative index for a coordinate in the base
        space.

        :param base_symbol: The symbol to find the base index of.
        :type base_symbol: :class:`sympy.Expr`

        :return: The multiindex of the desired symbol in the base space.
        :rtype: tuple[int, ...]
        """
        base_num = self.base_space.index(base_symbol)
        base_size = len(self.base_space)

        return (0,) * base_num + (1,) + (0,) * (base_size - base_num - 1)


    def extension(self, new_degree):
        """Creates a jet space on the same total space of a higher
        degree.

        :param new_degree: The degree of the extended jet space. Must
            be higher than the current degree.
        :type new_degree: int

        :return: A deep copy of the jet space, with additional jet
            fibers of higher degree.
        :rtype: :class:`~JetSpace`
        """
        new_space = deepcopy(self)

        if new_degree > new_space.degree:
            new_space._add_jet_fibers(new_space.degree + 1, new_degree)
            new_space.degree = new_degree
        else:
            raise ValueError("Extension must be to higher degree")

        return new_space


    @property
    def dependents(self):
        """The fibers of the total space on which the jet space is
        built.

        :return: The symbols of the original fibers.
        :rtype: list[:class:`sympy.Expr`]
        """

        return list(self.fibers)


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

            for dependent in self.fibers:
                for deriv_index, deriv_symbol in zip(deriv_indices,
                                                     deriv_symbols):
                    deriv_string = "".join(map(str, deriv_symbol))
                    symbol_name = dependent.name + "_{" + deriv_string + "}"
                    self.fibers[dependent][deriv_index] = Symbol(symbol_name)

    @property
    def original_total_space(self):
        """The coordinates of the total space on which the jet space is
        built.

        :return: The coordinates of the base space and fiber.
        :rtype: tuple[list[:class:`sympy.Expr`],
            list[:class:`sympy.Expr`]]
        """

        return self.base_space, self.dependents


def total_derivative(jet_exp, coordinate, domain):
    """The total derivative of an expression in a coordinate.

    :param jet_exp: The expression to be derived.
    :type jet_exp: :class:`sympy.Expr`

    :param coordinate: The coordinate to derive in. Should be one of the
        symbols of the base space.
    :type coordinate: :class:`sympy.Expr`

    :param domain: The jet space in which ``jet_exp`` exists.
    :type domain: :class:`~JetSpace`

    :return: The derived expression. This expression exists in the
        the (once) extended jet space compared to the domain of the
        derivative.
    :rtype: :class:`sympy.Expr`
    """
    codomain = domain.extension(domain.degree + 1)

    diff_jet_exp = jet_exp.diff(coordinate)

    coord_index = domain.base_index(coordinate)

    for dependent, dependent_fibers in domain.fibers.items():
        for fiber_index, fiber_coord in dependent_fibers.items():
            deriv_index = tuple(map(operator.add, fiber_index, coord_index))
            deriv_symbol = codomain.fibers[dependent][deriv_index]
            diff_jet_exp += deriv_symbol * jet_exp.diff(fiber_coord)

    return diff_jet_exp
