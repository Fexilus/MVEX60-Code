from sympy import symbols, ln

from symmetries.jetspace import JetSpace
from symmetries.generator import generator_on, lie_bracket
from symmetries.symcond import get_lin_symmetry_cond

# Time
t = time = symbols("t", real=True)
# States
W = symbols("W", positive=True)
G = symbols("G", real=True)
states = (W, G)

# Jet space and derivative coordinate
jet_space = JetSpace(time, states, 1)

Wt = jet_space.fibres[W][(1,)]
Gt = jet_space.fibres[G][(1,)]

# Parameters
kG = symbols("k_G")

# Differential equations
diff_equation = [Wt - G * W,
                 Gt + kG * G]

Generator = generator_on((time, states))
# Generators
X1 = Generator(1, (0, 0))
X2 = Generator(0, (ln(W) * W, G))
X3 = Generator(0, (W, 0))

generators = [X1, X2, X3]

# Confirm generators for respective equations
all_confirmed = True
for generator in generators:
    sym_conds = get_lin_symmetry_cond(diff_equation, generator, jet_space,
                                     derivative_hints=(Wt, Gt))
    if not all(sym_cond.expand().together().is_zero for sym_cond in sym_conds):
        all_confirmed = False

        print(f"The generator {generator} is not a symmetry of the system "
              "Gompertz model")

if all_confirmed:
    print("All expected generators are generators of the system Gompertz "
          "model")
