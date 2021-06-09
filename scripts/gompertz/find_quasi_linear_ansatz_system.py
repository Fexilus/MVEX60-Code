"""Calculate generators linear in the state for the system Gompertz
model.
"""
from sympy import symbols, poly, Function, Eq, dsolve, Derivative, linsolve

from symmetries import (JetSpace, get_lin_symmetry_cond, Generator,
                        decompose_generator)
from symmetries.utils import replace_consts
from printutils import CustomLatexPrinter


latex = CustomLatexPrinter({"ln_notation": True}, short_functions=False)

# Time
t = time = symbols("t", real=True)
# States
W, G = states = symbols("W G", nonnegative=True)

jet_space = JetSpace(time, states, 1)

Wt = jet_space.fibres[W][(1,)]
Gt = jet_space.fibres[G][(1,)]

# Parameters
kG = symbols("k_G")

# Right hand differential equation sides
omega_W = G * W
omega_G = -kG * G

diff_functions = [Wt - omega_W, Gt - omega_G]

# Ansatz formulation
arbitrary_functions = [Function(f"f_{i}")(t) for i in range(1, 10)]
f1, f2, f3, f4, f5, f6, f7, f8, f9 = arbitrary_functions
inf_generator = Generator(f1 + f2 * W + f3 * G,
                          [f4 + f5 * W + f6 * G,
                           f7 + f8 * W + f9 * G],
                          jet_space.base_space)

print("Ansatz:")
print(latex.doprint(Eq(Function(f"\\xi")(t, *states), inf_generator.xis[0])))
print(latex.doprint(Eq(Function(f"\\eta^1")(t, *states),
                       inf_generator.etas[0])))
print(latex.doprint(Eq(Function(f"\\eta^2")(t, *states),
                       inf_generator.etas[1])))

lin_symmetry_cond = get_lin_symmetry_cond(diff_functions, inf_generator,
                                          jet_space, derivative_hints=[Wt, Gt])

print("Linearized symmetry condition:")
for eq in lin_symmetry_cond:
    print(latex.doprint(Eq(eq.expand(), 0)))

num_eqs = len(lin_symmetry_cond)
num_decomposed_eqs = 0

print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
      end="", flush=True)

solvable_eqs = []
for eq in lin_symmetry_cond:
    solvable_eqs += poly(eq, states).coeffs()

    num_decomposed_eqs += 1
    print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
          end="", flush=True)

print(f"\nThe equation system has {str(len(solvable_eqs))} equations",
      flush=True)

for eq in solvable_eqs:
    print(latex.doprint(Eq(eq, 0)))

non_odeqs = [eq for eq in solvable_eqs if not eq.atoms(Derivative)]
odeqs = [eq for eq in solvable_eqs if eq.atoms(Derivative)]

solved_eqs = []
while non_odeqs:
    func_solutions = linsolve(solved_eqs + non_odeqs, arbitrary_functions)
    solution = dict(zip(arbitrary_functions, tuple(func_solutions)[0]))

    new_solvable_eqs = [eq.subs(solution).doit() for eq in odeqs]

    solved_eqs += non_odeqs

    non_odeqs = [eq for eq in new_solvable_eqs
                 if not eq.atoms(Derivative) and not eq.is_zero]
    odeqs = [eq for eq in new_solvable_eqs if eq.atoms(Derivative)]

# Index quick fix for algebraicly dependent equations
solution_raw = dsolve(odeqs[1:], [f1, f3, f5, f7, f9])
solution_new_const, arbitrary_const = replace_consts(solution_raw, "c")
solution.update((sol.lhs, sol.rhs) for sol in solution_new_const)

xis = [xi.subs(solution) for xi in inf_generator.xis]
etas = [eta.subs(solution) for eta in inf_generator.etas]

solution_generator = Generator(xis, etas, jet_space.original_total_space)

print("Solution:")
print(latex.doprint(Eq(Function(f"\\xi")(t, *states),
                       solution_generator.xis[0].expand())))
print(latex.doprint(Eq(Function(f"\\eta^1")(t, *states),
                       solution_generator.etas[0].expand())))
print(latex.doprint(Eq(Function(f"\\eta^2")(t, *states),
                       solution_generator.etas[1].expand())))

solution_basis = decompose_generator(solution_generator, arbitrary_const)

print("Solution basis:")
for i, generator in enumerate(solution_basis):
    print(f"Generator {i}:")
    print(latex.doprint(Eq(Function(f"\\xi")(t, *states), generator.xis[0])))
    print(latex.doprint(Eq(Function(f"\\eta^1")(t, *states),
                           generator.etas[0])))
    print(latex.doprint(Eq(Function(f"\\eta^2")(t, *states),
                           generator.etas[1])))
