from sympy import symbols, Function, poly, Derivative, pdsolve, Wild
from sympy.core.function import AppliedUndef
from sympy.solvers.solveset import linsolve

from symmetries.jetspace import JetSpace
from symmetries.generator import generator_on
from symmetries.symcond import get_lin_symmetry_cond

t = symbols("t", real=True)
u1, u2 = states = symbols("u^1 u^2", positive=True)
jet_space = JetSpace(t, states, 1)

u1t = jet_space.fibres[u1][(1,)]
u2t = jet_space.fibres[u2][(1,)]

a, b, c, d = parameters = symbols("a b c d")
diff_eqs = [u1t - a*u1 + b*u1*u2, u2t - c*u1*u2 + d*u2]

Generator = generator_on((t, states))

xi = Function("xi")(t, *states)
eta1 = Function("eta^1")(t, *states)
eta2 = Function("eta^2")(t, *states)

# Solve the symmetry conditions

#print(get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)), jet_space, derivative_hints=(u1t, u2t))[0].expand().collect(a, evaluate=False)[a**2])
xi = Function("xi")(t, u2)

#print(get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)), jet_space, derivative_hints=(u1t, u2t))[1].expand().collect(a, evaluate=False)[a])
eta2 = Function("eta^2")(t, u2)

eq_2_1_dict = get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)), jet_space, derivative_hints=(u1t, u2t))[1].expand().collect(a, evaluate=False)[1].collect(eta1, evaluate=False)
eta1 = - eq_2_1_dict[1] / eq_2_1_dict[eta1]

function_monoids = list(get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)), jet_space, derivative_hints=(u1t, u2t))[0].expand().atoms(AppliedUndef))
function_monoids += list(get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)), jet_space, derivative_hints=(u1t, u2t))[0].expand().atoms(Derivative))

a_u1_separated_equations = poly(get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)), jet_space, derivative_hints=(u1t, u2t))[0].expand(), (a, u1)).coeffs()

#print(dict(zip(function_monoids, list(linsolve(a_u1_separated_equations, function_monoids))[0])))
c1 = symbols("c_1", real=True)
old_xi = xi
xi = c1
eta1 = eta1.replace(old_xi, xi).doit()

eta2_solution = pdsolve(poly(get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)), jet_space, derivative_hints=(u1t, u2t))[0].expand(), (a, u1)).coeffs()[0])

old_eta2 = eta2
eta2 = eta2_solution.rhs
eta1 = eta1.replace(old_eta2, eta2).doit()

#print(poly(get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)), jet_space, derivative_hints=(u1t, u2t))[0].expand(), (a, u1)).coeffs()[1])

arbitrary_func_name = list(poly(get_lin_symmetry_cond(diff_eqs, Generator(xi, (eta1, eta2)), jet_space, derivative_hints=(u1t, u2t))[0].expand(), (a, u1)).coeffs()[1].atoms(AppliedUndef))[0].name

eta2 = eta2.replace(Function(arbitrary_func_name)(Wild("x")), 0)
#eta1 = eta1.replace(Function(arbitrary_func_name)(Wild("x")), 0).doit()
eta1 = 0 # Temporary fix because of bug fixed after release of sympy 1.7.1

print(Generator(xi, (eta1, eta2)))
