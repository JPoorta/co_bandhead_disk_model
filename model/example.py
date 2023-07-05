"""
Run this module to quickly calculate and plot a set of models.
Different parameter settings can be tested by changing input in :meth:`example.run` and defaults can be adapted in
``grid_params`` or ``all_params`` (see source code).
"""

import matplotlib.pyplot as plt
import numpy as np

import model.config as cfg
import model.sed_calculations as seds
from model.flat_disk_log_grid import run_grid_log_r


def run(test_param=None, test_param_array=None):
    """
    This method will be run when the module is called from the command line. It will calculate and plot a (set of)
    model(s) with default settings and the option to test several values for one parameter. NOTE: it will calculate
    the model(s) only for the first star specified in ``all_params[stars]``, unless `test_param` is "stars".

    :param test_param: the name of the input parameter which for which a (set of) model(s) is to be calculated. This
        can be any parameter listed in the dictionaries ``grid_params`` or ``all_params``.
    :type test_param: str
    :param test_param_array: list or array of values (can be only one) that test_param should take. In general
        parameters cannot take just any value, so make sure to provide correct types and (more or less) physical values
        where relevant. More information can be found in :meth:`flat_disk_log_grid.run_grid_log_r`.
    :type test_param_array: array-like
    :return:
    """
    if test_param is None:
        test_param = "dv0"  # "p"
    if test_param_array is None:
        test_param_array = [1, 2, 3]  # [-0.5, -0.75, -2]

    run_test(test_param, test_param_array)

    plt.show()

    return


# DEFAULTS

grid_params = {"Ri": 4.8,
               "Ti": 4000,  # [500,1000,2500,4000] ,
               "p": -0.75,  # [-0.5, -0.75, -2]
               "Ni": 3.e25,
               "q": -1.5,
               }

all_params = {"inc_deg": [40],  # 10,20,30,40,50,60,70,80
              "stars": ['B275'],  # , 'B243', 'B275', 'B163', 'B331']
              "dv0": [1],
              "vupper": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 2, 3, 4, 5, 6, 7],
              "vlower": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5],
              "nJ": 150,
              "dust": True,
              "sed_best_fit": True,
              # From here on only optional parameters.
              "num_CO": 100,
              "num_dust": 200,
              "Rmin_in": None,
              "Rmax_in": None,
              "print_Rs": True,
              "convolve": True,
              "save": None,
              "maxmin": (1.3, 1.02),
              "lisa_it": None,
              "saved_list": None
              }


def plot_star():
    """
    Function to plot the SED and photometry of the first star specified under ``all_params[stars]``.

    :return:
    """
    star = all_params.get("stars")[0]

    Mstar, T_eff, log_g, Rstar, Res, SNR, R_v, A_v, RV = cfg.stel_parameter_dict.get(star)
    photometry = cfg.photometry_dict.get(star)
    flux_phot, sigma_phot = np.stack(photometry[1]).T
    modelname, Ti_d, p_d, Ni_d, q_d, i_d, Ri_d_AU, gass_mass, chi_sq, chi_sq_red = cfg.best_dust_fit.get(star)
    wvl_sed, flux_sed, fl_star_ext, fl_dust_ext, flux_sed_not_ext = seds.SED_full(Ri_d_AU, Ti_d, p_d, Ni_d, q_d, i_d,
                                                                                  star,
                                                                                  A_v, R_v, num_r=200)
    plt.loglog(wvl_sed, flux_sed, label="SED best fit")
    plt.loglog(wvl_sed, fl_star_ext, label="star")
    plt.errorbar(photometry[0], flux_phot, yerr=sigma_phot, fmt='o', color='k', label='photometry')
    plt.xlim(0.4, 100)
    plt.ylim(1.e-15, 1.e-9)
    plt.legend()

    return


def make_grid(Ti, p, Ni, q, Ri):
    """
    Takes the grid arrays and converts it to a numpy iterable grid. For explanation on the grid arrays check
    :meth:`flat_disk_log_grid.run_grid_log_r`.

    :param Ti:
    :param p:
    :param Ni:
    :param q:
    :param Ri:
    :return: numpy grid
    """
    tiv, pv, niv, qv, riv = np.meshgrid(Ti, p, Ni, q, Ri, sparse=False)
    return [tiv, pv, niv, qv, riv]


def run_test(test_param, test_param_array):
    """
    For each value in `test_param_array` calculate and plot a model with otherwise default values. For more
    information on the input see :meth:`example.run`

    :param test_param:
    :param test_param_array:
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

        plt.figure(0)
        plt.title(star + " Continuum subtracted, extincted flux")
        plt.plot(wvl, flux_tot_ext - continuum_flux, label=test_param + " = " + str(value))
        plt.legend()

        plt.figure(1)
        plt.title(star + " Normalized flux")
        plt.plot(wvl, flux_norm_ext, label=test_param + " = " + str(value))
        plt.vlines([1.558, 1.5779, 1.5982], 0, 2, linestyles='dashed')
        # plt.ylim(0.99,1.12)
        plt.legend()

        plt.figure(3)
        plt.title(star + " Normalized, convolved flux")
        plt.plot(wvl, conv_flux_norm, label=test_param + " = " + str(value))
        # plt.ylim(0.99, 1.12)
        plt.legend()

        plt.figure(2)
        plt.title(star)
        plt.loglog(wvl, flux_tot_ext, label='total extincted flux; ' + test_param + " = " + str(value))

    plt.loglog(wvl, continuum_flux, '--', label="total continuum flux")
    plot_star()

    return


if __name__ == "__main__":
    run()
