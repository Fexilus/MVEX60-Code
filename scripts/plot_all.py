"""Script that plots everything to a desired location."""
import os.path
import argparse
import importlib


SOLUTION_PLOT_FILES = ["gompertz.plot_classical",
                       "gompertz.plot_autonomous",
                       "gompertz.plot_system",
                       "lotka-volterra.plot_solutions"]
GENERATOR_PLOT_FILES = ["gompertz.plot_classical_generators",
                        "gompertz.plot_autonomous_generators",
                        "gompertz.plot_system_generators"]
OTHER_PLOT_FILES = ["generic.plot_triangle_symmetries",
                    "generic.plot_jet_surface"]

# Import the files containing the plot functions as modules
solution_modules = []
for src_file in SOLUTION_PLOT_FILES:
    solution_modules.append(importlib.import_module(src_file))

generator_modules = []
for src_file in GENERATOR_PLOT_FILES:
    generator_modules.append(importlib.import_module(src_file))

other_modules = []
for src_file in OTHER_PLOT_FILES:
    other_modules.append(importlib.import_module(src_file))

# Add parser functionality
parser = argparse.ArgumentParser(description="Generate all plots in the"
                                             "project.")

parser.add_argument("-p", "--path", default="./",
                    help="Path that plots are saved to")

args = parser.parse_args()

path = os.path.realpath(args.path)

for module in solution_modules:
    module.plot(save_path=path)

for module in generator_modules:
    module.plot(save_path=path)

for module in other_modules:
    module.plot(save_path=path)
