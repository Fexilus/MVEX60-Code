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

    def __add__(self, other):

        if not isinstance(other, Generator):
            return NotImplemented

        if self.total_space != other.total_space:
            raise NotImplementedError("Generators have to be in same coordinates")

        sum_xis = []
        for xi1, xi2 in zip(self.xis, other.xis):
            sum_xis += [(xi1 + xi2).expand()]

        sum_etas = []
        for eta1, eta2 in zip(self.etas, other.etas):
            sum_etas += [(eta1 + eta2).expand()]

        return Generator(sum_xis, sum_etas, self.total_space)

    def __sub__(self, other):

        if not isinstance(other, Generator):
            return NotImplemented

        if self.total_space != other.total_space:
            raise NotImplementedError("Generators have to be in same coordinates")

        sum_xis = []
        for xi1, xi2 in zip(self.xis, other.xis):
            sum_xis += [(xi1 - xi2).expand()]

        sum_etas = []
        for eta1, eta2 in zip(self.etas, other.etas):
            sum_etas += [(eta1 - eta2).expand()]

        return Generator(sum_xis, sum_etas, self.total_space)
    
    def __truediv__(self, other):

        quot_xis = []
        for xi in self.xis:
            quot_xis += [(xi / other).expand()]

        quot_etas = []
        for eta in self.etas:
            quot_etas += [(eta / other).expand()]

        return Generator(quot_xis, quot_etas, self.total_space)

    def __rmul__(self, other):
        # Only right multiplication is implemented, as left multiplication of
        # the differential operator might be interpreted as application.

        prod_xis = []
        for xi in self.xis:
            prod_xis += [(other * xi).expand()]

        prod_etas = []
        for eta in self.etas:
            prod_etas += [(other * eta).expand()]

        return Generator(prod_xis, prod_etas, self.total_space)
    
    def __neg__(self):

        neg_xis = [(-xi).expand() for xi in self.xis]

        neg_etas = [(-eta).expand() for eta in self.etas]

        return Generator(neg_xis, neg_etas, self.total_space)

    def __pos__(self):

        pos_xis = [(+xi).expand() for xi in self.xis]

        pos_etas = [(+eta).expand() for eta in self.etas]

        return Generator(pos_xis, pos_etas, self.total_space)


def generator_on(total_space):
    """Returns a initiallization method for generators on the total space.

    Is meant to be used in code where several generators on the same space are
    used to reduce visual clutter.
    """

    class _Generator(Generator):
        def __init__(self, xis, etas):
            super().__init__(xis, etas, total_space)

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
