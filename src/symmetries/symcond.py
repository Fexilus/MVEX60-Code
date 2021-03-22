"""Testing of symmetry conditions."""
from sympy import linsolve

from .utils import optional_iter, iter_wrapper, zip_strict


@optional_iter
def get_lin_symmetry_cond(diff_eqs, generator, jet_space,
                          derivative_hints=None):
    """Test if the linearized symmetry conditions hold differential equations.

    Args:
        diff_eqs: A single or iterable of differential equations formulated in
            a jet space.

        generator: An infinitesimal generator that takes an expression and a
            jet space and applies the prolongation on that jet space to the
            expression.

        jet_space: The jet space on which the differential equations exist.

        derivative_hints: If given, contains a single item or list (depending
            on the form of the differential equations) of the highest order
            derivatives to be solved for.

    Returns:
        A single or list (depending on the form of the differential equations)
        of differential equations that must hold for the infinitesimal
        generator to generate a group of symmetries that constitute a symmetry.
    """

    # Ensure that the iterable is reusable
    diff_eqs = list(diff_eqs)

    eqs_halfway = (generator(eq, jet_space) for eq in diff_eqs)

    submanifold_subs = find_submanifold_subs(diff_eqs, jet_space,
                                             derivative_hints=derivative_hints)
    sym_cond = list(eq.subs(submanifold_subs) for eq in eqs_halfway)

    return sym_cond


def find_submanifold_subs(diff_eqs, jet_space, derivative_hints=None):
    """Find the substitutions for evaluation on differential equations."""

    if derivative_hints:
        subs_coords = list(iter_wrapper(derivative_hints))

        if len(subs_coords) > len(set(subs_coords)):
            ValueError("Multiple instances of same hint")
    else:
        raise NotImplementedError("Hint-free substitutions not implemented")

    def substitutions():
        for diff_eq, subs_coord in zip_strict(iter_wrapper(diff_eqs),
                                              subs_coords):
            rhs = list(linsolve([diff_eq], subs_coord))

            if len(rhs) != 1:
                if len(rhs) == 0:
                    raise ValueError(f"Hint {subs_coord} is not in equation")
                else:
                    raise ValueError(f"Hint {subs_coord} has multiple"
                                     "solutions")

            yield rhs[0][0]

    return list(zip(subs_coords, substitutions()))
