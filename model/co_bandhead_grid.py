"""
Run this code to calculate a grid of CO-bandhead spectra.
Check a few things:
1. In ```config.py``` set the correct project folder:
   >> project_folder = "<full/path/to/flat_disk_model_for_sharing/>"
2. Make sure the path to the project folder is inserted in the python path:
   >> sys.path.insert(0, "<full/path/to/flat_disk_model_for_sharing/>")
   As long as this script is run from within the flat_disk_model_for_sharing folder, this step should be redundant.
3. That the results folder exists in the project folder with the star folders.
(e.g. <full/path/to/flat_disk_model_for_sharing/results/B268>)
4. Check on local machine if all the imports work (if all necessary python packages are installed).
"""

import os
import sys

import numpy as np

sys.path.insert(0, "/home/johanna/PhD/github_codes/co_bandhead_disk_model/")
import model.config as cfg
import model.flat_disk_log_grid as fld

thread_no = int(sys.argv[-1])

cfg.get_and_convert_mag_naira()

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
    for star in stars:
        # Create the folders where the results are to be stored.
        print(cfg.results_folder + star + '/' + grid_name)
        try:
            os.mkdir(cfg.results_folder + star + '/' + grid_name)
        except:
            pass

    fld.run_grid_log_r(grid=grid, inc_deg=inc_deg, stars=stars, dv0=dv0, vupper=vupper, vlower=vlower, nJ=nJ,
                       dust=dust, sed_best_fit=sed_best_fit, save=grid_name, lisa_it=thread_no, saved_list=saved_list)

    return


calculate_grid(thread_no)
