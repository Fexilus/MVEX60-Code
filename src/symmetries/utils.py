"""Utilities."""
from itertools import zip_longest


def optional_iter(func):
    """Decorator that allows the first argument to be treated as iterable."""

    def as_iter(arg):
        yield arg

    def inner(first_arg, *args, **kwargs):
        try:
            # Test if iterable or not
            iter(first_arg)
        except TypeError:
            return next(func(as_iter(first_arg), *args, **kwargs))
        else:
            return func(first_arg, *args, **kwargs)

    return inner


def iter_wrapper(possible_iter):
    """Ensures that the argument can be treated as an iterator."""
    try:
        # Test if iterable or not
        iter(possible_iter)
    except TypeError:
        yield possible_iter
    else:
        yield from possible_iter


def zip_strict(*iters):
    """Zip two iterables and ensure that they are equal.
    Will be replaced with strict=True argument in python 3.10.
    """

    fill = object()

    for ziped in zip_longest(*iters, fillvalue=fill):
        if fill in ziped:
            raise ValueError("Iterables have different lengths")

        yield ziped
