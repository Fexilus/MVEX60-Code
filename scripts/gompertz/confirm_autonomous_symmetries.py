"""Confirm that symmetries of the autonomous Gompertz model calculated
by hand are correct.
"""
from sympy import symbols, ln, exp, Piecewise, solve, Equality, expand_log

from symmetries import JetSpace, generator_on, get_lin_symmetry_cond

# Time
t = time = symbols("t", real=True)
# States
W = state = symbols("W", positive=True)

# Jet space and derivative coordinate
jet_space = JetSpace(time, state, 1)

Wt = jet_space.fibres[W][(1,)]

# Parameters
kG, A = symbols("k_G A", positive=True)

# Differential equations
autonomous_equation = Wt + kG * W * ln(W / A)

Generator = generator_on(jet_space.original_total_space)
# Generators
X_aut1 = Generator(exp(kG * t) * ln(W / A), 0)
X_aut2 = Generator(0, exp(-kG * t) * W)
X_aut3 = Generator(0, W * ln(W / A))
X_aut4 = Generator(1, 0)
X_aut5 = Generator(t, W * ln(W / A) * ln(abs(ln(W / A))))
X_aut6 = Generator(exp(-kG * t), -kG * exp(-kG * t) * W * ln(W))

generators = [X_aut1, X_aut2, X_aut3, X_aut4, X_aut5, X_aut6]

# Confirm generators for respective equations
all_autonomous_confirmed = True
for generator in generators:
    sym_cond = get_lin_symmetry_cond(autonomous_equation, generator, jet_space,
                                     derivative_hints=Wt)
    if not sym_cond.expand().together().is_zero:
        # Test if the symmetry condition holds piecewise
        if isinstance(sym_cond.expand().together().simplify(), Piecewise):
            piecewise_holds = True
            for expr, cond in sym_cond.expand().together().simplify().args:

                if isinstance(cond, Equality):
                    # If the condition can never happen, skip checking
                    # the expression
                    if not solve(cond):
                        continue

                    expanded_expr = expand_log(expr)
                    expanded_lhs = expand_log(cond.lhs)
                    cond_expr = expanded_expr.subs(expanded_lhs, cond.rhs)

                else:
                    cond_expr = expr

                if not cond_expr.expand().together().is_zero:
                    piecewise_holds = False

            if piecewise_holds:
                print(f"The generator {generator} is piecewise a "
                      "symmetry of the autonomous Gompertz model")
                continue

        all_autonomous_confirmed = False

        print(f"The generator {generator} is not a symmetry of the autonomous "
              "Gompertz model")

if all_autonomous_confirmed:
    print("All expected generators are generators of the autonomous Gompertz "
          "model")
