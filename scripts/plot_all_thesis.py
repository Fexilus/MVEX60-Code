"""Script that plots everything to a desired location."""
import os.path
import argparse
import importlib

import matplotlib.pyplot as plt


plt.rc("font", family="serif")
plt.rc("mathtext", fontset="cm")
plt.rc("axes", titlesize=16)

SOLUTION_PLOT_FILES = [("gompertz.plot_classical", {}),
                       ("gompertz.plot_autonomous", {}),
                       ("gompertz.plot_system", {}),
                       ("lotka-volterra.plot_solutions", {})]
GENERATOR_PLOT_FILES = [("gompertz.plot_classical_generators", {}),
                        ("gompertz.plot_autonomous_generators", {}),
                        ("gompertz.plot_system_generators", {})]
OTHER_PLOT_FILES = [("generic.plot_triangle_symmetries", {}),
                    ("generic.plot_jet_surface", {}),
                    ("gompertz.plot_jet_surfaces", {})]

# Import the files containing the plot functions as modules
solution_modules = []
for src_file, kwargs in SOLUTION_PLOT_FILES:
    solution_modules.append((importlib.import_module(src_file), kwargs))

generator_modules = []
for src_file, kwargs in GENERATOR_PLOT_FILES:
    generator_modules.append((importlib.import_module(src_file), kwargs))

other_modules = []
for src_file, kwargs in OTHER_PLOT_FILES:
    other_modules.append((importlib.import_module(src_file), kwargs))

# Add parser functionality
parser = argparse.ArgumentParser(description="Generate all plots in the"
                                             "project.")

parser.add_argument("-p", "--path", default="./",
                    help="Path that plots are saved to")

args = parser.parse_args()

path = os.path.realpath(args.path)

# Plot solution curves
for module, kwargs in solution_modules:
    module.plot(save_path=path, **kwargs)

# Plot generator representatives
generator_kwargs = {"transformation_kw_args": {"arrow_stroke_arguments": {"scaling": 2.8}}}

for module, kwargs in generator_modules:
    module.plot(save_path=path, **generator_kwargs, **kwargs)

# Plot other plots
for module, kwargs in other_modules:
    module.plot(save_path=path, **kwargs)
