"""Confirm that symmetries of the classical Gompertz model calculated by
hand are correct.
"""
from sympy import symbols, ln, exp

from symmetries import JetSpace, generator_on, get_lin_symmetry_cond

# Time
t = time = symbols("t", real=True)
# States
W = state = symbols("W", nonnegative=True)

# Jet space and derivative coordinate
jet_space = JetSpace(time, state, 1)

Wt = jet_space.fibres[W][(1,)]

# Parameters
kG, Ti = symbols("k_G T_i")

# Differential equations
classical_equation = Wt - kG * exp(-kG * (t - Ti)) * W

Generator = generator_on(jet_space.original_total_space)
# Generators
X_cla1 = Generator(exp(kG * t), 0)
X_cla2 = Generator(W*exp(kG * t + exp(-kG * (t - Ti))), 0)
X_cla3 = Generator(0, exp(-exp(Ti * kG) * exp(-kG * t)))
X_cla4 = Generator(0, W)
X_cla5 = Generator(1, - kG * W * ln(W))

generators = [X_cla1, X_cla2, X_cla3, X_cla4, X_cla5]

all_classical_confirmed = True
for generator in generators:
    sym_cond = get_lin_symmetry_cond(classical_equation, generator, jet_space,
                                     derivative_hints=Wt)
    if not sym_cond.expand().together().is_zero:
        all_classical_confirmed = False

        print(f"The generator {generator} is not a symmetry of the classical "
              "Gompertz model")

if all_classical_confirmed:
    print("All expected generators are generators of the classical Gompertz "
          "model")
