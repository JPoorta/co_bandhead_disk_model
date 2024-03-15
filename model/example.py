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
from processing_and_plotting.Gridpoint import Gridpoint
from grids.list_of_grids import common_test_grid
import grids.run_grids as run_gr


def run():
    run_gr.run_quick_test_grid(common_test_grid(), quick_plots=True, plot_B275_checks=True)

    return


def make_grid(ti, t1, a, p, ni, q, ri):
    """
    Takes the grid arrays and converts it to a numpy iterable grid. For explanation on the grid arrays check
    :meth:`flat_disk_log_grid.run_grid_log_r`.

    :param ti:
    :param t1:
    :param a:
    :param p:
    :param ni:
    :param q:
    :param ri:
    :return: numpy grid
    """
    tiv, t1v, av, pv, niv, qv, riv = np.meshgrid(ti, t1, a, p, ni, q, ri, sparse=False)
    return [tiv, t1v, av, pv, niv, qv, riv]


def run_test(test_param, test_param_array, grid_params, all_params, figax, quick_plots=False):
    """
    For each value in `test_param_array` calculate and plot a model with otherwise default values. For more
    information on the input see :meth:`example.run`

    :param test_param:
    :param test_param_array:
    :param grid_params:
    :param all_params:
    :param quick_plots:
    :return:
    """
    star = all_params.get("stars")[0]

    fig, ax = figax
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
        gp = Gridpoint(all_params=all_params, **grid_params)

        wvl, flux_tot_ext, flux_norm_ext, flux_norm_intp, conv_flux, conv_flux_norm = \
            run_grid_log_r(grid, **all_params)

        continuum_flux = flux_tot_ext / flux_norm_ext

        pltr.plot_t_structure_gas(gp)

        pltr.quick_plot_norm_convolved_flux(star, wvl, conv_flux_norm, label=pltr.label_dict[test_param]
                                                                             + " = " + str(value),
                                            fig_ax=(fig, ax))
        if quick_plots:
            pltr.quick_plot_results(star, wvl, flux_tot_ext, flux_norm_ext, continuum_flux,
                                    label=test_param + " = " + str(value))

    return gp, star, wvl, continuum_flux


def finish_plots(gp, star, wvl, continuum_flux, figax, plot_B275_checks=False, quick_plots=False):

    fig, ax = figax

    if quick_plots:
        plt.figure(2)
        plt.loglog(wvl, continuum_flux, '--', label="total continuum flux")
        pltr.plot_star(star)

    pltr.plot_t_structure_dust(gp)
    pltr.plot_t_structure_original(star)
    plt.figure(3)
    pltr.plot_obs_spectrum(star, fig_ax=(fig, ax))
    if plot_B275_checks:
        pltr.plot_275_checks(wvl, fig_ax=(fig, ax), rmax_in=False)

    return


if __name__ == "__main__":
    run()
