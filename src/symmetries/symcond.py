"""Testing of symmetry conditions."""
from sympy import linsolve

from .utils import optional_iter, iter_wrapper, zip_strict


@optional_iter
def get_lin_symmetry_cond(diff_eqs, generator, jet_space,
                          derivative_hints=None):
    """Test if the linearized symmetry conditions hold differential
    equations.

    :param diff_eqs: The differential equation(s) expressed in jet space
        notation.
    :type diff_eqs: :class:`sympy.Expr` or list[:class:`sympy.Expr`]

    :param generator: The generator corresponding to the Lie group of
        transformations to be tested.
    :type generator: :class:`~generator.Generator`

    :param jet_space: The jet space on which the differential equations
        exist.
    :type jet_space: :class:`~jetspace.JetSpace`

    :param derivative_hints: The highest order derivative(s) that the
        differential equation(s) can be solved for.
    :type derivative_hints: :class:`sympy.Expr` or
        list[:class:`sympy.Expr`]

    :return: The differential equation(s) that must hold for the
        infinitesimal generator to generate a Lie group of symmetries.
        The differential equations are expressed in jet space notation.
    :rtype: :class:`sympy.Expr` or list[:class:`sympy.Expr`]
    """
    # Ensure that the iterable is reusable
    diff_eqs = list(diff_eqs)

    eqs_halfway = (generator(eq, jet_space) for eq in diff_eqs)

    submanifold_subs = find_submanifold_subs(diff_eqs, jet_space,
                                             derivative_hints=derivative_hints)
    sym_cond = list(eq.subs(submanifold_subs) for eq in eqs_halfway)

    return sym_cond


@optional_iter
def find_submanifold_subs(diff_eqs, jet_space, derivative_hints=None):
    """Find substitutions that can be used to evaluate a jet space
    expression on the surface of a differential equation.

    :param diff_eqs: The differential equation(s) expressed in jet space
        notation.
    :type diff_eqs: :class:`sympy.Expr` or list[:class:`sympy.Expr`]

    :param jet_space: The jet space on which the differential equations
        exist. (Is currently not used)
    :type jet_space: :class:`~jetspace.JetSpace`

    :param derivative_hints: Highest order derivative(s) that the
        differential equation(s) can be solved for. (Is currently
        required)
    :type derivative_hints: :class:`sympy.Expr` or
        list[:class:`sympy.Expr`]
    """
    if derivative_hints:
        subs_coords = list(iter_wrapper(derivative_hints))

        if len(subs_coords) > len(set(subs_coords)):
            ValueError("Multiple instances of same hint")
    else:
        raise NotImplementedError("Hint-free substitutions not implemented")

    def substitutions():
        for diff_eq, subs_coord in zip_strict(diff_eqs, subs_coords):
            rhs = list(linsolve([diff_eq], subs_coord))

            if len(rhs) != 1:
                if len(rhs) == 0:
                    raise ValueError(f"Hint {subs_coord} is not in equation")
                else:
                    raise ValueError(f"Hint {subs_coord} has multiple"
                                     "solutions")

            yield rhs[0][0]

    return list(zip(subs_coords, substitutions()))
