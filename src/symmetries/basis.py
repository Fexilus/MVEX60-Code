"""A function to split a generator into base components."""
from itertools import chain

from sympy import poly

from .generator import Generator


def decompose_generator(generator, basis):
    """Decompose a generator by a basis of arbitrary constants or
    functions.

    Only decomposition of generators linear in the basis is implemented.

    :param generator: The generator to decompose.
    :type generator: :class:`~generator.Generator`

    :param basis: The arbitrary constants or functions in which the
        generator can be decomposed.
    :type basis: list[:class:`sympy.Expr`]

    :return: The generators that span the space the input generator was
        in.
    :rtype: list[:class:`~generator.Generator`]
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
