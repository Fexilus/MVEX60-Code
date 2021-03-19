from itertools import count
import operator

from sympy import symbols
from sympy.polys.monomials import itermonomials
from sympy.polys.orderings import monomial_key

from ..generator import Generator


def create_poly_ansatz(jet_space, degree=1):
    independents = jet_space.base_space
    dependents = jet_space.get_dependents()

    key = monomial_key("grlex", tuple(reversed(independents + dependents)))

    monoids = sorted(itermonomials(independents + dependents, degree), key=key)

    all_constants = []
    constant_rows = count(1)

    xis = []
    for _, i in zip(independents, constant_rows):
        constants = symbols(f"c_{i}_{{(1:{len(monoids) + 1})}}")
        xis += [sum(map(operator.mul, monoids, constants))]

        all_constants += constants

    etas = []
    for _, i in zip(dependents, constant_rows):
        constants = symbols(f"c_{i}_{{(1:{len(monoids) + 1})}}")
        etas += [sum(map(operator.mul, monoids, constants))]

        all_constants += constants

    generator = Generator(xis, etas)

    return generator, all_constants
