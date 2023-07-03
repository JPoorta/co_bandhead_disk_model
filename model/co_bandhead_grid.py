"""
Run this module to calculate a grid of CO-bandhead spectra. Check a few things:

1. In ``config.py`` set the correct project folder:

.. code-block:: python

    project_folder = "<full/path/to/>co_bandhead_disk_model/"

2. For running from the command line, make sure the path to the project folder is inserted in the python path:

.. code-block:: python

    sys.path.insert(0, "<full/path/to/>co_bandhead_disk_model/")

3. That the ``results`` folder exists in the project folder with the star folders.
    (e.g. `<full/path/to>/co_bandhead_disk_model/results/B268`)

4. Check on local machine if all the imports work (if all necessary python packages are installed).

Run this module from command line as follows:

.. code-block:: bash

    $ python model/co_bandhead_grid

"""

import os
import sys

import numpy as np

sys.path.insert(0, "/home/johanna/PhD/github_codes/co_bandhead_disk_model/")
import model.config as cfg
import model.flat_disk_log_grid as fld


# 90000
Ti = np.array([2000, 3000, 4000, 4500, 5000, 5500, 6000, 7000, 8000])  # 9
p = np.array([0.5, 0.75, 1, 2, 3]) * -1  # 5
Ni = np.geomspace(5.e23, 5.e27, 10)  # 10
q = np.array([-1, 0.5, 1, 1.5]) * -1  # 4
Ri = np.geomspace(1.1, 30, 10)  # 10
dv0 = np.array([1, 2, 3, 4, 5])  # 5

inc_deg = np.array([10, 20, 30, 40, 50, 60, 70, 80])

stars = ['B268', 'B243', 'B275', 'B163', 'B331']  #

vupper = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 2, 3, 4, 5, 6, 7]
vlower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5]
nJ = 150
dust = True
saved_list = None

sed_best_fit = True
grid_name = "grid_21_5_2021_dv"
wvl_file = "wvl_re.npy"

tiv, pv, niv, qv, riv = np.meshgrid(Ti, p, Ni, q, Ri, sparse=False)
grid = [tiv, pv, niv, qv, riv]


def calculate_grid(thread_no):
    """
    Calculates the full grid specified in this module (or 1/16th of it, see ``thread_no``)  for all the objects under
    ``stars``; i.e., one grid per star. Results will be stored in the ``results`` folder with structure
    **results/<star>/<grid_name>/*.npy**.

    :param thread_no: If `int`, it should be between 0 and 15; in which case for each integer 1/16th of the
        grid will be calculated, so that the grid can be calculated parallel on 16 cores, by calling this  function
        16 times with ``thread_no`` = [0, ... ,15].

        If `None`, the full grid for all the stars will be calculated serially.
    :type thread_no: int or None
    :return:
    """
    for star in stars:
        # Create the folders where the results are to be stored.
        print(cfg.results_folder + star + '/' + grid_name)
        try:
            os.mkdir(cfg.results_folder + star + '/' + grid_name)
        except FileNotFoundError:
            print("Make sure a *results* folder exists in co_bandhead_disk_model containing one folder for each star.")
            return
        except FileExistsError:
            pass

    fld.run_grid_log_r(grid=grid, inc_deg=inc_deg, stars=stars, dv0=dv0, vupper=vupper, vlower=vlower, nJ=nJ,
                       dust=dust, sed_best_fit=sed_best_fit, save=grid_name, lisa_it=thread_no, saved_list=saved_list)

    return


if __name__ == "__main__":

    try:
        thread_no = int(sys.argv[-1])
    except ValueError:
        thread_no = None
    calculate_grid(thread_no)
