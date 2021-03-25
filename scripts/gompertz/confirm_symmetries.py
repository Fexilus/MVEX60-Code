from sympy import symbols, ln, exp

from symmetries.jetspace import JetSpace
from symmetries.generator import generator_on
from symmetries.symcond import get_lin_symmetry_cond

# Time
t = time = symbols("t", real=True)
# States
W = state = symbols("W", nonnegative=True)

# Jet space and derivative coordinate
jet_space = JetSpace(time, state, 1)

Wt = jet_space.fibres[W][(1,)]

# Parameters
kG, Ti, A = symbols('k_G T_i A')

# Differential equations
autonomous_equation = Wt + kG * W * ln(W / A)
classical_equation = Wt - kG * exp(-kG * (t - Ti)) * W

Generator = generator_on((time, state))
# Generators
X_aut1 = Generator(1, 0)
X_aut2 = Generator(t, W * ln(W / A) * ln(ln(W / A)))
X_aut3 = Generator(0, W * ln(W / A))
X_aut4 = Generator(exp(-kG * t), -kG * exp(-kG * t) * W * ln(W / A))
X_aut6 = Generator(0, exp(-kG * t) * W)

autonomous_generators = [X_aut1, X_aut2, X_aut3, X_aut4, X_aut6]

X_cla1 = Generator(0, W)
X_cla2 = Generator(1, - kG * W * ln(W))
X_cla3 = Generator(exp(kG * t), 0)

classical_generators = [X_cla1, X_cla2, X_cla3]

# Confirm generators for respective equations
all_autonomous_confirmed = True
for generator in autonomous_generators:
    sym_cond = get_lin_symmetry_cond(autonomous_equation, generator, jet_space,
                                     derivative_hints=Wt)
    if not sym_cond.expand().is_zero:
        all_autonomous_confirmed = False

        print(f"The generator {generator} is not a symmetry of the autonomous "
              "Gompertz model")

if all_autonomous_confirmed:
    print("All expected generators are generators of the autonomous Gompertz "
          "model")

all_classical_confirmed = True
for generator in classical_generators:
    sym_cond = get_lin_symmetry_cond(classical_equation, generator, jet_space,
                                     derivative_hints=Wt)
    if not sym_cond.expand().is_zero:
        all_classical_confirmed = False

        print(f"The generator {generator} is not a symmetry of the classical "
              "Gompertz model")

if all_classical_confirmed:
    print("All expected generators are generators of the classical Gompertz "
          "model")

# Try using generators of the equations on the other equation
unique_classical_generators = (generator for generator in classical_generators
                               if generator not in autonomous_generators)
any_classical_overlapping = False
for generator in unique_classical_generators:
    sym_cond = get_lin_symmetry_cond(autonomous_equation, generator, jet_space,
                                     derivative_hints=Wt)
    if sym_cond.expand().is_zero:
        any_classical_overlapping = True

        print(f"{generator} is also a symmetry of the autonomous Gompertz "
              "model")

if not any_classical_overlapping:
    print("None of the unique classical symmetry generators are symmetries "
          "of the autonomous Gompertz model")

unique_autonomous_generators = (generator for generator
                                in autonomous_generators
                                if generator not in classical_generators)
any_autonomous_overlapping = False
for generator in unique_autonomous_generators:
    sym_cond = get_lin_symmetry_cond(classical_equation, generator, jet_space,
                                     derivative_hints=Wt)
    if sym_cond.expand().is_zero:
        any_autonomous_overlapping = True

        print(f"{generator} is also a symmetry of the classical Gompertz "
              "model")

if not any_autonomous_overlapping:
    print("None of the unique autonomous symmetry generators are symmetries "
          "of the classical Gompertz model")
