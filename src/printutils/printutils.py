"""Class for reproducing most of the printing style of my reports."""
from sympy.printing.latex import LatexPrinter
from sympy.core.function import AppliedUndef, Derivative, Function, Subs
from sympy import Symbol, S


class CustomLatexPrinter(LatexPrinter):
    """Latex printer that prints derivatives with f_t notation.

    Redefines several of the built in methods of the default LaTeX
    printer.
    """

    def __init__(self, *args, short_functions=True, **kwargs):
        self._short_functions=short_functions

        super().__init__(*args, **kwargs)

    def _print_Derivative(self, expr):
        if not isinstance(expr.expr, AppliedUndef):
            return super()._print_Derivative(expr)
        func_expr = expr.expr

        if len(func_expr.args) == 1:
            tex = func_expr.func.name + "\'" * expr.derivative_count
        else:
            tex = func_expr.func.name + "_{"

            for derivative in expr.variables:
                tex += str(derivative)

            tex += "}"

        deriv_as_func_expr = Function(tex)(*func_expr.args)

        return self._print_Function(deriv_as_func_expr)

    def _print_Function(self, expr, exp=None):
        if ((not self._short_functions)
            or (not isinstance(expr, AppliedUndef))
            or (not all(isinstance(arg, Symbol) for arg in expr.args))):

            return super()._print_Function(expr, exp)

        return self._hprint_Function(expr.func.name)

    def _print_Subs(self, subs):
        expr, old, new = subs.args

        if isinstance(expr, AppliedUndef):
            raise NotImplementedError("Subs of argument in function")
        if isinstance(expr, Derivative):
            args = expr.expr.args

            if len(args) > 1:
                raise NotImplementedError("Only single arg implemented")

            deriv_func_name = self._print_Derivative(expr)
            as_func = Function(deriv_func_name)(*args)

            new_expr = Subs(as_func, old, new).doit()

            return super()._print_Function(new_expr)
        else:
            return super()._print_Subs(subs)

    def _hprint_Function(self, func):
        main, first_tick, more_ticks = func.partition("\'")

        return super()._hprint_Function(main) + first_tick + more_ticks

    def _print_Poly(self, poly):

        terms = []
        for monom, coeff in poly.terms():
            s_monom = ''
            for i, exp in enumerate(monom):
                if exp > 0:
                    if exp == 1:
                        s_monom += self._print(poly.gens[i])
                    else:
                        s_monom += self._print(pow(poly.gens[i], exp))

            if coeff.is_Add:
                if s_monom:
                    s_coeff = f"\\left({self._print(coeff)}\\right)"
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue

                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue

                s_coeff = self._print(coeff)

            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + " " + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ['-', '+']:
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        tex = ' '.join(terms)

        return tex
