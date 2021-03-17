# Symmetry calculations for systems of first oreder ODE:s
This repository contains python code for calculating and validating symmetries of differential equations, created for my Master's thesis project on symmetries in biochemical and ecological ODE:s.
The code is built on the `sympy` package.

The code is formated as a combination of a package and scripts.
This means that project-specific scripts can easily be run requiring minimal knowledge of python packages, while a user intending to use the code to do their own calculations can do this by installing the package.

## Using the the scripts

The dependencies of the scripts, including the developed package, are contatined in `requirements.txt`.
Some examples of how to install dependencies from the file are shown below; the user can choose among these or other methods to their own preference.
```
  $ pip install -r requirements.txt
  $ pipenv install
  $ conda install --file requirements.txt
```
When the dependencies are installed, the scripts can then be run as any other python script by
```
  $ python scripts/path/to/script.py
```
while in the correct environment, or any other method to the users liking.

## Using the package
TODO
