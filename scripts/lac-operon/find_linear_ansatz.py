"""Calculate generators linear in time and states for lac operon model."""

from sympy import symbols, together, numer, poly, linsolve, Function, Eq

from symmetries.jetspace import JetSpace
from symmetries.ansatz.polynomial import create_poly_ansatz
from symmetries.symcond import get_lin_symmetry_cond
from symmetries.generator import Generator
from symmetries.ansatz.basis import decompose_generator

from printutils import CustomLatexPrinter

latex = CustomLatexPrinter({"ln_notation": True})

# Time
t = time = symbols("t", real=True)
# States
M, B, L, A, P = states = symbols("M B L A P", nonnegative=True)

jet_space = JetSpace(time, states, 1)

# Parameters
alphaM, K1, K, Gamma0, gammaM = symbols("alpha_M, K_1, K, Gamma_0, gamma_M")
n = symbols("n", positive=True)
alphaB, gammaB = symbols("alpha_B, gamma_B")
alphaL, Le, KLe, betaL1, KL1, betaL2, KL2, gammaL = symbols("alpha_L, L_e, K_{L_e}, beta_{L_1}, K_{L_1}, beta_{L_2}, K_{L_2}, gamma_L")
alphaA, KL, betaA, KA, gammaA = symbols("alpha_A, K_L, beta_A, K_A, gamma_A")
alphaP, gammaP = symbols("alpha_P, gamma_P")

# Right hand differential equation sides
omega_M = alphaM * (1 + K1 * A ** n) / (K + K1 * A ** n) + Gamma0 - gammaM * M
omega_B = alphaB * M - gammaB * B
omega_L = alphaL * P * Le / (KLe + Le) - betaL1 * P * L / (KL1 + L) - betaL2 * B * L / (KL2 + L) - gammaL * L
omega_A = alphaA * B * L / (KL + L) - betaA * B * A / (KA + A) - gammaA * A
omega_P = alphaP * M - gammaP * P

right_hand_sides = [omega_M, omega_B, omega_L, omega_A, omega_P]

diff_functions = [jet_space.fibres[M][(1,)] - omega_M,
                  jet_space.fibres[B][(1,)] - omega_B,
                  jet_space.fibres[L][(1,)] - omega_L,
                  jet_space.fibres[A][(1,)] - omega_A,
                  jet_space.fibres[P][(1,)] - omega_P]

inf_generator, ansatz_consts = create_poly_ansatz(jet_space, 1)

print("Ansatz:")

print_xi = poly(inf_generator.xis[0], ansatz_consts + [t] + list(states))
print(latex.doprint(Eq(Function(f"\\xi")(t, *states), print_xi)))

for i, eta in enumerate(inf_generator.etas, start=1):
    print_eta = poly(eta, ansatz_consts + [t] + list(states))
    print(latex.doprint(Eq(Function(f"\\eta^{i}")(t, *states), print_eta)))

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
    eq_numer = numer(together(eq))
    solvable_eqs += poly(eq_numer, (time,) + states + (A ** n,)).coeffs()

    num_decomposed_eqs += 1
    print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
          end="", flush=True)

print(f"\nThe equation system has {str(len(solvable_eqs))} equations",
      flush=True)

consts_sol = linsolve(solvable_eqs, ansatz_consts)
solution = dict(zip(ansatz_consts, tuple(consts_sol)[0]))
xis = [xi.subs(solution) for xi in inf_generator.xis]
etas = [eta.subs(solution) for eta in inf_generator.etas]

solution_generator = Generator(xis, etas, jet_space.original_total_space)

solution_basis = decompose_generator(solution_generator, ansatz_consts)

print("Solution basis:")
for generator in solution_basis:
    print(generator)
