"""
Calculate generators linear in the state for the classical Gompertz
model.
"""

from sympy import symbols, poly, Function, Eq, exp, dsolve

from symmetries.jetspace import JetSpace
from symmetries.symcond import get_lin_symmetry_cond
from symmetries.generator import Generator
from symmetries.ansatz.basis import decompose_generator
from symmetries.utils import replace_consts

from printutils import CustomLatexPrinter

latex = CustomLatexPrinter({"ln_notation": True}, short_functions=False)

# Time
t = time = symbols("t", real=True)
# States
W = state = symbols("W", nonnegative=True)

jet_space = JetSpace(time, state, 1)

Wt = jet_space.fibres[W][(1,)]

# Parameters
kG, Ti = symbols("k_G T_i")

# Right hand differential equation sides
right_hand_side = omega = kG * exp(-kG * (t - Ti)) * W

diff_function = Wt - omega

# Ansatz formulation
f1, f2, f3, f4 = arbitrary_functions = [Function(f"f_{i}")(t)
                                        for i in range(1, 5)]
inf_generator = Generator(f1 + f2 * W, f3 + f4 * W, jet_space.base_space)

print("Ansatz:")
print(latex.doprint(Eq(Function(f"\\xi")(t, state), inf_generator.xis[0])))
print(latex.doprint(Eq(Function(f"\\eta")(t, state), inf_generator.etas[0])))

lin_symmetry_cond = get_lin_symmetry_cond(diff_function, inf_generator,
                                          jet_space, derivative_hints=Wt)

print("Linearized symmetry condition:")
print(latex.doprint(Eq(lin_symmetry_cond.expand(), 0)))

num_eqs = 1
num_decomposed_eqs = 0

print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
      end="", flush=True)

odeqs = poly(lin_symmetry_cond, W).coeffs()

num_decomposed_eqs += 1
print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
        end="", flush=True)

print(f"\nThe equation system has {str(len(odeqs))} ODE:s:",
      flush=True)

for odeq in odeqs:
    print(latex.doprint(Eq(odeq, 0)))

solution_raw = dsolve(odeqs, arbitrary_functions[0:len(odeqs)])
solution_new_const, arbitrary_const = replace_consts(solution_raw, "c")
solution = dict([(sol.lhs, sol.rhs) for sol in solution_new_const])

xis = [xi.subs(solution) for xi in inf_generator.xis]
etas = [eta.subs(solution) for eta in inf_generator.etas]

solution_generator = Generator(xis, etas, jet_space.original_total_space)

print("Solution:")
print(latex.doprint(Eq(Function(f"\\xi")(t, state),
                       solution_generator.xis[0])))
print(latex.doprint(Eq(Function(f"\\eta")(t, state),
                       solution_generator.etas[0])))

solution_basis = decompose_generator(solution_generator, arbitrary_const)

print("Solution basis:")
for i, generator in enumerate(solution_basis):
    print(f"Generator {i}:")
    print(latex.doprint(Eq(Function(f"\\xi")(t, state),
                           generator.xis[0].expand().powsimp())))
    print(latex.doprint(Eq(Function(f"\\eta")(t, state),
                           generator.etas[0].expand().powsimp())))
