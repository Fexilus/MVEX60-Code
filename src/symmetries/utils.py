"""Utilities for the main symmetries package."""
from itertools import zip_longest
from functools import wraps

from sympy import symbols


def optional_iter(func):
    """Decorator that allows the first argument and return value to be
    optionally treated as iterables.

    :param func: The function should take an iterable as first argument
        and return an iterable.

    :return: The decorated function will accept either a single element
        or an iterable as first argument, and will return a single
        element or an iterable depending on the input type.
    """
    def as_iter(arg):
        yield arg

    @wraps(func)
    def wrapped_func(first_arg, *args, **kwargs):
        try:
            # Test if iterable or not
            iter(first_arg)
        except TypeError:
            return next(iter(func(as_iter(first_arg), *args, **kwargs)))
        else:
            return func(first_arg, *args, **kwargs)

    return wrapped_func


def iter_wrapper(possible_iter):
    """Ensures that the argument can be treated as an iterable.

    :param possible_iter: A possible iterable.

    :return: A guaranteed iterable.
    """
    try:
        yield from possible_iter
    except TypeError:
        yield possible_iter


def zip_strict(*iters):
    """Zip two iterables and ensure that they are equal.
    Will be replaced with ``strict=True`` argument in Python 3.10.

    :param \\*iters: Any number of iterables.

    :return: An iterable through tuples of the ingoing iterables.

    :raises ValueError: if the iterables have different lengths.
    """
    fill = object()

    for ziped in zip_longest(*iters, fillvalue=fill):
        if fill in ziped:
            raise ValueError("Iterables have different lengths")

        yield ziped


def derivatives_sort_key(function_order=None, argument_order=None):
    """Ad hoc sort key for derivatives.

    :param function_order: An ordering of the functions.
    :type function_order: list[:class:`sympy.Derivative`], optional

    :param argument_order: An ordering of the arguments.
    :type argument_order: list[:class:`sympy.Expr`], optional

    :return: A sorting key for :class:`sympy.Derivative` that sorts
        firstly by function, secondly by derivative degree and thirdly
        by derivative arguments.
    """
    if function_order:
        num_funcs = len(function_order)
    else:
        num_funcs = 1

    def _key(derivative):
        nonlocal function_order, argument_order

        if not function_order:
            function_order = [derivative.expr]

        if not argument_order:
            argument_order = list(derivative.expr_free_symbols)

        num_dep = len(argument_order)

        num_der = derivative.derivative_count

        # This is a very overkill upper bound of
        # 1) All combinations with replacement of up to num_der
        #    derivative
        # 2) The value of dep_val
        comb_bound = num_der * num_dep ** num_der * num_der ** num_dep
        count_val = num_funcs * comb_bound

        func_index = list(function_order).index(derivative.expr)
        func_val = func_index * comb_bound

        variable_dict = dict(derivative.variable_count)
        dep_val = 0
        for index, dep in enumerate(argument_order):
            dep_val += variable_dict.get(dep, 0) * num_der ** index

        return count_val + func_val + dep_val

    return _key


def replace_consts(exprs, new_const_name):
    """Replace arbitrary constant from eg. :func:`sympy.dsolve`.

    :param exprs: The expression(s) to replace constants in.
    :type exprs: :class:`sympy.Expr` or list[:class:`sympy.Expr`]

    :param new_const_name: The new base name to give to the constants.
        E.g. ``new_const_name=r"\\alpha"`` will result in constants
        ``r"\\alpha_{1}"``, ``r"\\alpha_{2}"`` etc.
    :type new_const_name: str

    :return: The new expressions and all the new constants.
    :rtype: tuple[:class:`sympy.Expr` or list[:class:`sympy.Expr`],
        list[:class:`sympy.Expr`]]
    """
    exprs = iter_wrapper(exprs)

    new_consts = []

    i = 1
    while True:
        old_const = symbols(f"C{i}")
        new_const = symbols(f"{new_const_name}_{{{i}}}")

        new_exprs = [expr.subs(old_const, new_const) for expr in exprs]
        if new_exprs == exprs:
            return exprs, new_consts

        new_consts.append(new_const)
        exprs = new_exprs
        i += 1
