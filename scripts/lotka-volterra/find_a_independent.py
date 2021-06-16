"""Calculate generators independent of parameter a."""

from math import prod

from sympy import symbols, Function, poly, Derivative, pdsolve, Wild, Eq
from sympy.core.function import AppliedUndef
from sympy.solvers.solveset import linsolve

from symmetries import JetSpace, generator_on, get_lin_symmetry_cond
from symmetries.utils import derivatives_sort_key
from printutils import CustomLatexPrinter


t = symbols("t", real=True)
u1, u2 = states = symbols("N P", positive=True)
jet_space = JetSpace(t, states, 1)

u1t = jet_space.fibers[u1][(1,)]
u2t = jet_space.fibers[u2][(1,)]

a, b, c, d = parameters = symbols("a b c d")
diff_eqs = [u1t - a*u1 + b*u1*u2, u2t - c*u1*u2 + d*u2]

Generator = generator_on(jet_space.original_total_space)

xi = Function("\\xi")(t, *states)
eta1 = Function("\\eta^1")(t, *states)
eta2 = Function("\\eta^2")(t, *states)

# Printing tools
latex = CustomLatexPrinter({"ln_notation": True})

# Solve the symmetry conditions

sym_conds = get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)),
                                  jet_space, derivative_hints=(u1t, u2t))

param_ind_det_eq1_dict = sym_conds[0].expand().collect(a, evaluate=False)
param_ind_det_eq2_dict = sym_conds[1].expand().collect(a, evaluate=False)

print("Original parameter independence determining equations:")
print(latex.doprint(param_ind_det_eq1_dict[1].collect((xi, eta1, eta2))))
print(latex.doprint(param_ind_det_eq1_dict[a].collect((xi, eta1, eta2))))
print(latex.doprint(param_ind_det_eq1_dict[a**2].collect((xi, eta1, eta2))))
print(latex.doprint(param_ind_det_eq2_dict[1].collect((xi, eta1, eta2))))
print(latex.doprint(param_ind_det_eq2_dict[a].collect((xi, eta1, eta2))))

xi = Function("\\xi")(t, u2)
eta2 = Function("\\eta^2")(t, u2)

sym_conds = get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)),
                                  jet_space, derivative_hints=(u1t, u2t))

param_ind_det_eq2_dict = sym_conds[1].expand().collect(a, evaluate=False)
eq_2_1_dict = param_ind_det_eq2_dict[1].collect(eta1, evaluate=False)

old_eta1 = eta1
eta1 = - eq_2_1_dict[1] / eq_2_1_dict[eta1]

print("dPdt, 1 can be written as:")
print(latex.doprint(Eq(old_eta1, eta1)))

# The second symmetry condition is \equiv 0, so only first is dealt with
sym_cond = get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)),
                                 jet_space, derivative_hints=(u1t, u2t))[0]
function_monoids = list(sym_cond.expand().atoms(AppliedUndef))
function_monoids += sorted(sym_cond.expand().atoms(Derivative),
                           key=derivatives_sort_key([xi, eta2], [t, u2]))

a_u1_separated_equations = poly(sym_cond.expand(), (a, u1)).as_dict()

print("dNdt eq. a-N-separation:")
for key, eq in a_u1_separated_equations.items():
    key_prod = prod(expr**n for expr, n in zip((a, u1), key))
    print(f"{latex.doprint(key_prod)} & : & "
          f"{latex.doprint(eq.expand().collect(function_monoids))}")

algebraic_sols = linsolve(list(a_u1_separated_equations.values()),
                          function_monoids)
algebraic_sols_dict = dict(zip(function_monoids, list(algebraic_sols)[0]))

print("dNdt eq. a-N-separation and linsolve:")
print(latex.doprint(Eq(xi.diff(t), algebraic_sols_dict[xi.diff(t)])))
print(latex.doprint(Eq(xi.diff(u2), algebraic_sols_dict[xi.diff(u2)])))

c1 = symbols("c_1", real=True)
old_xi = xi
xi = c1
eta1 = eta1.replace(old_xi, xi).doit()

sym_cond = get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)),
                                 jet_space, derivative_hints=(u1t, u2t))[0]
function_monoids = list(sym_cond.expand().atoms(AppliedUndef))
function_monoids += sorted(sym_cond.expand().atoms(Derivative),
                           key=derivatives_sort_key([xi, eta2], [t, u2]))

a_u1_separated_equations = poly(sym_cond.expand(), (a, u1)).as_dict()

print("Simplification gives the dNdt eq. a-N-separation:")
for key, eq in a_u1_separated_equations.items():
    key_prod = prod(expr**n for expr, n in zip((a, u1), key))
    print(f"{latex.doprint(key_prod)} & : & "
          f"{latex.doprint(eq.expand().collect(function_monoids))}")

eta2_solution = pdsolve(a_u1_separated_equations[(1, 0)])

print("With solution:")
print(latex.doprint(eta2_solution))
old_eta2 = eta2
eta2 = eta2_solution.rhs
eta1 = eta1.replace(old_eta2, eta2).doit()

sym_cond = get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)),
                                 jet_space, derivative_hints=(u1t, u2t))[0]
function_monoids = list(sym_cond.expand().atoms(AppliedUndef))
function_monoids += sorted(sym_cond.expand().atoms(Derivative),
                           key=derivatives_sort_key())

a_u1_separated_equations = poly(sym_cond.expand(), (a, u1)).as_dict()

print("Further simplified dNdt eq. a-N-separation:")
for key, eq in a_u1_separated_equations.items():
    key_prod = prod(expr**n for expr, n in zip((a, u1), key))
    print(f"{latex.doprint(key_prod)} & : & "
          f"{latex.doprint(eq.expand().collect(function_monoids))}")

relevant_eq = a_u1_separated_equations[(0, 1)]
arbitrary_func_name = list(relevant_eq.atoms(AppliedUndef))[0].name
eta2 = eta2.replace(Function(arbitrary_func_name)(Wild("x")), 0)
eta1 = eta1.replace(Function(arbitrary_func_name)(Wild("x")), 0).doit()

print("Final result:")
print(Generator(xi, (eta1, eta2)))
