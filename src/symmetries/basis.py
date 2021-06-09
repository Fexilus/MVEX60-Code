"""Operations to determine generator bases for ansätze."""
from itertools import chain

from sympy import poly

from .generator import Generator


def decompose_generator(generator, basis):
    """Decompose a generator by a basis of arbitrary constants or functions.

    Only generators linear in the basis is implemented.
    """

    xi_polys = [poly(xi, basis) for xi in generator.xis]
    eta_polys = [poly(eta, basis) for eta in generator.etas]

    generator_basis = []

    for base in chain((1,), basis):
        nonzero_generator = False

        base_xis = []
        base_etas = []

        for xi_poly in xi_polys:
            coeff = xi_poly.coeff_monomial(base)

            if coeff:
                nonzero_generator = True
                base_xis += [coeff]
            else:
                base_xis += [0]

        for eta_poly in eta_polys:
            coeff = eta_poly.coeff_monomial(base)

            if coeff:
                nonzero_generator = True
                base_etas += [coeff]
            else:
                base_etas += [0]

        if nonzero_generator:
            generator_basis += [Generator(base_xis, base_etas,
                                          generator.total_space)]

    return generator_basis
