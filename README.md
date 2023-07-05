# CO bandhead disk model

This code calculates parametrized LTE models of CO (overtone) bandhead emission, from a flat Keplarian disk with a radial temperature and column density structure. Detailed description of the model and its scientific motivation can be found in the paper "Massive pre-main-sequence stars in M17: First and second overtone CO bandhead emission and the thermal infrared." by [J. Poorta et al.](https://ui.adsabs.harvard.edu/abs/2023arXiv230501436P/abstract) (DOI: https://doi.org/10.1051/0004-6361/202245658). The exact version of the code used for the paper can be found at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7774529.svg)](https://doi.org/10.5281/zenodo.7774529)

## Table of Contents
- [Installation and dependencies](#installation-and-dependencies)
- [Documentation](#documentation)
- [Issues, bugs and problems](#issues-bugs-and-problems)
- [Acknowledging the code](#acknowledging-the-code)

## Installation and dependencies
To run the code and calculate (grids) of models the repository can be downloaded. 

The exact environment (which was created under Python 3.7.5) used to develop the code can be found in `requirements.txt`. However, the code will likely work under any Python 3 distribution, once the essential packages `numpy`, `matplotlib`, `pandas`, and `scipy` are installed . 

All relevant modules are in the ```model``` directory.  To get started run ``example.py`` which calculates a few models and plots them. From within the folder where the project is stored, run:
```bash
python -m model.example
```
To calculate a full grid of models (WARNING: this takes a long time!), run the ``co_bandhead_grid.py``. Before doing so, make sure to follow the instructions at the beginning of that module (see [Documentation](#Documentation)). Then, from within the folder where the project is stored, run:
```bash
python -m model.co_bandhead_grid
```
The (grid) parameters in ``example`` or ``co_bandhead_grid`` can easily be adjusted in the source files.

## Documentation

For detailed documentation per module and method see the GitHub [pages](https://jpoorta.github.io/co_bandhead_disk_model/index.html) documentation of this project.

## Issues, bugs and problems

Some known issues can be found in the project [Issues](https://github.com/JPoorta/co_bandhead_disk_model/issues). Feel free to raise any new issues. 

## Acknowledging the code

When making use of this code in a scientific publication please cite the paper [J. Poorta et al.](https://doi.org/10.1051/0004-6361/202245658).
