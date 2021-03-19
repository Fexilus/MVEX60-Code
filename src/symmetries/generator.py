"""Implementation of infinitesimal generators for sympy."""
import operator

from sympy import sympify

from .jetspace import total_derivative
from .utils import iter_wrapper


class Generator:
    """A local coordinate representation of an infinitesimal generator."""

    xis = []
    etas = []

    def __init__(self, xis, etas):

        self.xis = [sympify(xi) for xi in iter_wrapper(xis)]

        self.etas = [sympify(eta) for eta in iter_wrapper(etas)]

    def __call__(self, expr, jet_space):
        """Apply the generator on an expression on a jet space."""

        eta_prolongations = get_prolongations(self.xis, self.etas, jet_space)

        # Apply differential operation in each jet space coordinate
        out_expr = 0

        for base_coord, xi in zip(jet_space.base_space, self.xis):
            out_expr += xi * expr.diff(base_coord)

        for dependent in jet_space.fibres:
            for multiindex in jet_space.fibres[dependent]:
                derivative = expr.diff(jet_space.fibres[dependent][multiindex])
                eta_prolongation = eta_prolongations[dependent][multiindex]

                out_expr += eta_prolongation * derivative

        return out_expr

    def __repr__(self):
        return f"Generator({self.xis}, {self.etas})"

    def __str__(self):
        xi_str = "  " + "\n  ".join(str(xi) for xi in self.xis)
        eta_str = "  " + "\n  ".join(str(eta) for eta in self.etas)

        return f"Generator with xis:\n{xi_str}\nand etas:\n{eta_str}"


def get_prolongations(xis, etas, jet_space):
    """Calculate the coefficients of a vector field prolonged over a jet space.

    The vector field is characterized by the coefficients of derivatives in
    the base space (xis) and the coefficients of derivatives in the fiber of
    the original fiber bundle from which the jet space is created.
    """

    eta_prolongations = {}
    base_size = len(jet_space.base_space)

    for dependent, eta in zip(jet_space.fibres, etas):

        multiindex_iter = iter(jet_space.fibres[dependent])

        eta_prolongations[dependent] = {(0,) * base_size: eta}
        next(multiindex_iter)

        for multiindex in multiindex_iter:
            # Calculate the index class, ie. the "first" derivative's number
            index_class = next(i for i, x in enumerate(multiindex) if x)

            leading_deriv_index = ((0,) * index_class +
                                   (1,) +
                                   (0,) * (base_size - index_class - 1))
            leading_deriv_symbol = jet_space.base_space[index_class]

            # Calculate a lower order deivative from which the current
            # derivative can be taken.
            prev_index = tuple(map(operator.sub, multiindex,
                                   leading_deriv_index))

            prev_prolongation = eta_prolongations[dependent][prev_index]

            # The D(eta_(n-1)) component of the prolongation formula
            eta_component = total_derivative(prev_prolongation,
                                             leading_deriv_symbol,
                                             jet_space)
            eta_prolongations[dependent][multiindex] = eta_component

            # The omega_(n-1)*D(xi) components of the prolongation formula
            for base_coord, xi in zip(jet_space.base_space, xis):
                base_index = jet_space.base_index(base_coord)
                derivative_index = tuple(map(operator.add, prev_index,
                                             base_index))

                deriv_coord = jet_space.fibres[dependent][derivative_index]
                xi_term = deriv_coord * total_derivative(xi, base_coord,
                                                         jet_space)

                eta_prolongations[dependent][multiindex] -=  xi_term

    return eta_prolongations
