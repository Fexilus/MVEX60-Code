"""Implementation of infinitesimal generators for sympy."""
import operator

from sympy import sympify

from .jetspace import JetSpace, total_derivative
from .utils import iter_wrapper, zip_strict


class Generator:
    """A local coordinate representation of an infinitesimal generator."""

    def __init__(self, xis, etas, total_space):

        self.xis = [sympify(xi) for xi in iter_wrapper(xis)]

        self.etas = [sympify(eta) for eta in iter_wrapper(etas)]

        self.total_space = total_space

    def __call__(self, expr, jet_space=None):
        """Apply the generator on an expression on a jet space."""

        if not jet_space:
            jet_space = JetSpace(*self.total_space, 0)

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

        return f"Generator({self.xis}, {self.etas} on {self.total_space})"

    def __str__(self):

        xi_str = "  " + "\n  ".join(str(xi) for xi in self.xis)
        eta_str = "  " + "\n  ".join(str(eta) for eta in self.etas)

        return (f"Generator on {self.total_space}\nwith xis:\n{xi_str}\n"
                f"and etas:\n{eta_str}")

    def __eq__(self, other):

        if isinstance(other, Generator):
            if self.total_space != other.total_space:
                return False

            for this_xi, other_xi in zip_strict(self.xis, other.xis):
                if this_xi.expand() != other_xi.expand():
                    return False

            for this_eta, other_eta in zip_strict(self.etas, other.etas):
                if this_eta.expand() != other_eta.expand():
                    return False

            return True

        return False


def generator_on(total_space):
    """Returns a initiallization method for generators on the total space."""

    class _Generator(Generator):
        def __init__(self, xis, etas):
            super(_Generator, self).__init__(xis, etas, total_space)

    return _Generator


def get_prolongations(xis, etas, jet_space):
    """Calculate the coefficients of a vector field prolonged over a jet space.

    The vector field is characterized by the coefficients of derivatives in
    the base space (xis) and the coefficients of derivatives in the fiber of
    the original fiber bundle (etas) from which the jet space is created.
    """

    eta_prolongations = {}
    base_size = len(jet_space.base_space)

    for dependent, eta in zip_strict(jet_space.fibres, etas):

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
            for base_coord, xi in zip_strict(jet_space.base_space, xis):
                base_index = jet_space.base_index(base_coord)
                derivative_index = tuple(map(operator.add, prev_index,
                                             base_index))

                deriv_coord = jet_space.fibres[dependent][derivative_index]
                xi_term = deriv_coord * total_derivative(xi, base_coord,
                                                         jet_space)

                eta_prolongations[dependent][multiindex] -=  xi_term

    return eta_prolongations


def lie_bracket(generator1, generator2):
    """The Lie bracket of two generators in the same coordinate system."""

    if generator1.total_space != generator2.total_space:
        raise NotImplementedError("Generators have to be in same coordinates")

    bracket_xis = []
    for xi1, xi2 in zip(generator1.xis, generator2.xis):
        bracket_xis += [(generator1(xi2) - generator2(xi1)).expand()]

    bracket_etas = []
    for eta1, eta2 in zip(generator1.etas, generator2.etas):
        bracket_etas += [(generator1(eta2) - generator2(eta1)).expand()]

    return Generator(bracket_xis, bracket_etas, generator1.total_space)
