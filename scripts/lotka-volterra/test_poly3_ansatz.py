from sympy import symbols, poly, linsolve

from symmetries.jetspace import JetSpace
from symmetries.ansatz.polynomial import create_poly_ansatz
from symmetries.symcond import get_lin_symmetry_cond


# Time
t = time = symbols("t", real=True)
# States
u1, u2 = states = symbols("u^1 u^2", nonnegative=True)

jet_space = JetSpace(time, states, 1)

# Parameters
r, a = symbols('r a')
b, m = symbols('b m')

# Right hand differential equation sides
omega_1 = r * u1 - a * u1 * u2
omega_2 = b * u1 * u2 - m * u2

right_hand_sides = [omega_1, omega_2]

diff_functions = [jet_space.fibres[u1][(1,)] - omega_1,
                  jet_space.fibres[u2][(1,)] - omega_2]

inf_generator, ansatz_consts = create_poly_ansatz(jet_space, 3)

first_derivatives = (fibres[(1,)] for _, fibres in jet_space.fibres.items())
lin_symmetry_cond = get_lin_symmetry_cond(diff_functions, inf_generator,
                                          jet_space,
                                          derivative_hints=first_derivatives)

num_eqs = len(lin_symmetry_cond)
num_decomposed_eqs = 0

print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
      end="", flush=True)

solvable_eqs = []
for eq in lin_symmetry_cond:
    solvable_eqs += poly(eq, (time,) + states).coeffs()

    num_decomposed_eqs += 1
    print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
          end="", flush=True)

print(f"\nThe equation system has {str(len(solvable_eqs))} equations",
      flush=True)

consts_sol = linsolve(solvable_eqs, ansatz_consts)
solution = dict(zip(ansatz_consts, tuple(consts_sol)[0]))

print(solution)
