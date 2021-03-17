"""Implementation of infinitesimal generators for sympy."""
import operator

from sympy import Array, sympify

from .jetspace import total_derivative


def create_generator(xis, etas):
    """Create an operator on the form of an infinitesimal generator."""

    try:
        iter(xis)
    except TypeError:
        xis = [xis]

    xis = sympify(xis)

    try:
        iter(etas)
    except TypeError:
        etas = [etas]

    etas = sympify(etas)

    def apply_generator(expr, jet_space):
        eta_prolongations = {}
        base_size = len(jet_space.base_space)

        for dependent, eta in zip(jet_space.fibres, etas):
            eta_prolongations[dependent] = {(0,) * base_size: eta}

            multiindex_iter = iter(jet_space.fibres[dependent])
            next(multiindex_iter)

            for multiindex in multiindex_iter:
                index_class = next((i for i, x in enumerate(multiindex) if x), None)
                leading_deriv_index = (0,) * index_class + (1,) + (0,) * (base_size - index_class - 1)
                leading_deriv_symbol = jet_space.base_space[index_class]

                prev_index = tuple(map(operator.sub, multiindex, leading_deriv_index))
                prev_prolongation = eta_prolongations[dependent][prev_index]

                eta_prolongations[dependent][multiindex] = total_derivative(prev_prolongation, leading_deriv_symbol, jet_space)

                for base_coord, xi in zip(jet_space.base_space, xis):
                    base_index = jet_space.base_index(base_coord)
                    derivative_index = tuple(map(operator.add, prev_index, base_index))

                    eta_prolongations[dependent][multiindex] -= jet_space.fibres[dependent][derivative_index] * total_derivative(xi, base_coord, jet_space)

        out_expr = 0

        for base_coord, xi in zip(jet_space.base_space, xis):
            out_expr += xi * expr.diff(base_coord)

        for dependent in jet_space.fibres:
            for multiindex in jet_space.fibres[dependent]:
                out_expr += eta_prolongations[dependent][multiindex] * expr.diff(jet_space.fibres[dependent][multiindex])

        return out_expr

    return apply_generator
