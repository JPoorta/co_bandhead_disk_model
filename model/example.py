"""
Run this module to quickly calculate and plot a set of models with default settings and the option to test several
    values for one parameter. NOTE: it will calculate the model(s) only for the first star specified in
    ``all_params[stars]``, unless `test_param` is "stars".
Different parameter settings can be tested by changing input in :meth:`example.run` and defaults can be adapted in
``grid_params`` or ``all_params``.
``test_param`` (str) the name of the input parameter which for which a (set of) model(s) is to be calculated. This
        can be any parameter listed in the dictionaries ``grid_params`` or ``all_params``.
``test_param_array`` (array-like) list or array of values (can be only one) that test_param should take. In general
        parameters cannot take just any value, so make sure to provide correct types and (more or less) physical values
        where relevant. More information can be found in :meth:`flat_disk_log_grid.run_grid_log_r`.
"""

import matplotlib.pyplot as plt
import numpy as np

import processing_and_plotting.plotting_routines as pltr
from model.flat_disk_log_grid import run_grid_log_r
import model.config as cfg


def run():

    star = "B275"
    grid_params, all_params = cfg.get_default_params(star)

    # set the parameter to be tested (optional).
    test_param = "p"  # "Ti"
    test_param_array = [-2, ]

    # Adjust defaults if wanted (optional).
    all_params["vupper"] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    all_params["vlower"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    all_params["dust"] = True
    grid_params["p"] = -2
    all_params["Rmax_in"] = 100
    all_params["dF"] = None
    all_params["num_CO"] = 100

    run_test(test_param, test_param_array, grid_params, all_params)

    plt.show()

    return


def make_grid(ti, p, ni, q, ri):
    """
    Takes the grid arrays and converts it to a numpy iterable grid. For explanation on the grid arrays check
    :meth:`flat_disk_log_grid.run_grid_log_r`.

    :param ti:
    :param p:
    :param ni:
    :param q:
    :param ri:
    :return: numpy grid
    """
    tiv, pv, niv, qv, riv = np.meshgrid(ti, p, ni, q, ri, sparse=False)
    return [tiv, pv, niv, qv, riv]


def run_test(test_param, test_param_array, grid_params, all_params):
    """
    For each value in `test_param_array` calculate and plot a model with otherwise default values. For more
    information on the input see :meth:`example.run`

    :param test_param:
    :param test_param_array:
    :param grid_params:
    :param all_params:
    :return:
    """
    star = all_params.get("stars")[0]

    for value in test_param_array:

        # If the parameter is a grid parameter:
        if test_param in grid_params.keys():
            grid_params[test_param] = value
        # If it is one of the parameters which take an iterable as input.
        elif test_param in ["inc_deg", "stars", "dv0"]:
            all_params[test_param] = [value]
        # If it is one of the other passable parameters:
        elif test_param in all_params.keys():
            all_params[test_param] = value
        else:
            print(
                "Please pass a valid test parameter. "
                "This is one of the parameters in the dictionaries <grid_params> and <all_params>.")
            return

        grid = make_grid(**grid_params)
        wvl, flux_tot_ext, flux_norm_ext, flux_norm_intp, conv_flux, conv_flux_norm = \
            run_grid_log_r(grid, **all_params)

        continuum_flux = flux_tot_ext / flux_norm_ext

        pltr.quick_plot_results(star, wvl, flux_tot_ext, flux_norm_ext, conv_flux_norm, continuum_flux,
                                label=test_param + " = " + str(value))

    plt.loglog(wvl, continuum_flux, '--', label="total continuum flux")
    pltr.plot_star(all_params.get("stars")[0])

    return


if __name__ == "__main__":
    run()
