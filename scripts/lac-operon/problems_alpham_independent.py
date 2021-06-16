"""Show the problems with finding :math:`\\alpha_M`-independent
symmetries.
"""
from sympy import symbols, Function, Derivative, Union

from symmetries import JetSpace, generator_on, get_lin_symmetry_cond
from symmetries.utils import derivatives_sort_key
from printutils import CustomLatexPrinter

# Time
t = time = symbols("t", real=True)
# States
M, B, L, A, P = states = symbols("M B L A P", nonnegative=True)

jet_space = JetSpace(time, states, 1)

Mt = jet_space.fibers[M][(1,)]
Bt = jet_space.fibers[B][(1,)]
Lt = jet_space.fibers[L][(1,)]
At = jet_space.fibers[A][(1,)]
Pt = jet_space.fibers[P][(1,)]

derivatives = [Mt, Bt, Lt, At, Pt]

# Parameters
alphaM, K1, K, Gamma0, gammaM = symbols("\\alpha_M, K_1, K, \\Gamma_0, "
                                        "\\gamma_M", positive=True)
n = symbols("n", positive=True)
alphaB, gammaB = symbols("\\alpha_B, \\gamma_B", positive=True)
alphaL, Le, KLe, betaL1, KL1, betaL2, KL2, gammaL = symbols("\\alpha_L, L_e, "
                                                            "K_{L_e}, "
                                                            "\\beta_{L_1}, "
                                                            "K_{L_1}, "
                                                            "\\beta_{L_2}, "
                                                            "K_{L_2}, "
                                                            "\\gamma_L",
                                                            positive=True)
alphaA, KL, betaA, KA, gammaA = symbols("\\alpha_A, K_L, \\beta_A, K_A, "
                                        "\\gamma_A", positive=True)
alphaP, gammaP = symbols("\\alpha_P, \\gamma_P", positive=True)

# Right hand differential equation sides
omega_M = alphaM * (1 + K1 * A ** n) / (K + K1 * A ** n) + Gamma0 - gammaM * M
omega_B = alphaB * M - gammaB * B
omega_L = (alphaL * P * Le / (KLe + Le) - betaL1 * P * L / (KL1 + L)
           - betaL2 * B * L / (KL2 + L) - gammaL * L)
omega_A = alphaA * B * L / (KL + L) - betaA * B * A / (KA + A) - gammaA * A
omega_P = alphaP * M - gammaP * P

right_hand_sides = [omega_M, omega_B, omega_L, omega_A, omega_P]

diff_eqs = [Mt - omega_M,
            Bt - omega_B,
            Lt - omega_L,
            At - omega_A,
            Pt - omega_P]

Generator = generator_on(jet_space.original_total_space)

xi = Function("\\xi")(t, *states)
etas = []
for i in range(1, 6):
    etas.append(Function(f"\\eta^{i}")(t, *states))

# Printing tools
latex = CustomLatexPrinter({"ln_notation": True})

# Solve the symmetry conditions

sym_conds = get_lin_symmetry_cond(diff_eqs, Generator(xi, etas),
                                  jet_space, derivative_hints=derivatives)
function_monoids = [xi, *etas]
all_derivs = Union(*(sym_cond.expand().atoms(Derivative)
                     for sym_cond in sym_conds))
function_monoids += sorted(all_derivs,
                           key=derivatives_sort_key([xi, *etas], [t, *states]))

num_eqs = len(sym_conds)
num_decomposed_eqs = 0

print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
      end="", flush=True)

param_ind_det_eq_dicts = []
for sym_cond in sym_conds:
    param_ind_det_eq_dicts.append(sym_cond.expand().collect(alphaM,
                                                            evaluate=False))

    num_decomposed_eqs += 1
    print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
          end="", flush=True)

tot_num_eqs = sum(len(eq_dict) for eq_dict in param_ind_det_eq_dicts)
print(f"\nThe equation system has {str(tot_num_eqs)} equations",
      flush=True)

print("Number of terms in the parameter independence determining equations:")
for eq_num, eq_dict in enumerate(param_ind_det_eq_dicts):
    for key, eq in eq_dict.items():
        print(f"{derivatives[eq_num]}, {key}: "
              f"{len(eq.expand().args)} terms")

# Since (1 + K1 * A ** n) / (K + K1 * A ** n) is not zero for all A > 0
print("Eliminate M-derivatives from all functions but eta1:")

xi = Function("\\xi")(t, B, L, A, P)
for i in range(2, 6):
    etas[i - 1] = Function(f"\\eta^{i}")(t, B, L, A, P)

sym_conds = get_lin_symmetry_cond(diff_eqs, Generator(xi, etas),
                                  jet_space, derivative_hints=derivatives)

num_eqs = len(sym_conds)
num_decomposed_eqs = 0

print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
      end="", flush=True)

param_ind_det_eq_dicts = []
for sym_cond in sym_conds:
    param_ind_det_eq_dicts.append(sym_cond.expand().collect(alphaM,
                                                            evaluate=False))

    num_decomposed_eqs += 1
    print(f"\r{num_decomposed_eqs}/{num_eqs} equations decomposed",
          end="", flush=True)

tot_num_eqs = sum(len(eq_dict) for eq_dict in param_ind_det_eq_dicts)
print(f"\nThe equation system has {str(tot_num_eqs)} equations",
      flush=True)

print("Number of terms in the parameter independence determining equations:")
for eq_num, eq_dict in enumerate(param_ind_det_eq_dicts):
    for key, eq in eq_dict.items():
        print(f"{derivatives[eq_num]}, {key}: "
              f"{len(eq.expand().args)} terms")
