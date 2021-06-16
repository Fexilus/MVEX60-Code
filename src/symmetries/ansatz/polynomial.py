"""Automatic generation of polynomial ansätze for infinitesimal
generators.
"""
from itertools import count
import operator

from sympy import Symbol
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key

from ..generator import Generator


def create_poly_ansatz(jet_space, degree=1):
    """Create an infinitesimal generator that is polynomial in the
    components of a given jet space.

    :param jet_space: The jet space on which the generator can act.
    :type jet_space: :class:`~symmetries.jetspace.JetSpace`

    :param degree: The degree of the polynomials in all generator
        components.
    :type degree: int, optional

    :return: The generator and arbitrary constants defined by the
        ansatz.
    :rtype: tuple[:class:`~symmetries.generator.Generator`,
        list[:class:`sympy.Expr`]]
    """
    independents = jet_space.base_space
    dependents = jet_space.dependents

    key = monomial_key("grlex", tuple(reversed(independents + dependents)))

    monoids = sorted(itermonomials(independents + dependents, degree), key=key)

    all_constants = []
    constant_rows = count(1)

    xis = []
    for _, i in zip(independents, constant_rows):
        constants = [Symbol(f"c_{{{i},{j}}}")
                     for j in range(1, len(monoids) + 1)]
        xis += [sum(map(operator.mul, monoids, constants))]

        all_constants += constants

    etas = []
    for _, i in zip(dependents, constant_rows):
        constants = [Symbol(f"c_{{{i},{j}}}")
                     for j in range(1, len(monoids) + 1)]
        etas += [sum(map(operator.mul, monoids, constants))]

        all_constants += constants

    generator = Generator(xis, etas, jet_space.original_total_space)

    return generator, all_constants
