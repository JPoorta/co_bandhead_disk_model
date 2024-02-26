import matplotlib.pyplot as plt
import numpy as np

import model.config as cfg
import model.sed_calculations as seds


def plot_star(star):
    """
    Function to plot the SED and photometry of the first star specified under ``all_params[stars]``.

    :return:
    """

    Mstar, T_eff, log_g, Rstar, Res, SNR, R_v, A_v, RV = cfg.stel_parameter_dict.get(star)
    photometry = cfg.photometry_dict.get(star)
    flux_phot, sigma_phot = np.stack(photometry[1]).T
    modelname, Ti_d, p_d, Ni_d, q_d, i_d, Ri_d_AU, gas_mass, chi_sq, chi_sq_red = cfg.best_dust_fit.get(star)
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


def plot_obs_spectrum(star):
    """
    Function to read the observed and fitted spectra of CO paper I (Poorta et al., 2023)

    :param star: (str)
    :return:
    """
    wvl_obj, flux_obj, wvl, conv_flux, fit_mask = np.load(str(cfg.spectra_dir) + "/" + star + ".npy", allow_pickle=True)
    plt.plot(wvl_obj, flux_obj, label="data")
    plt.plot(wvl, conv_flux, label="best fit")
    plt.legend()

    return


def quick_plot_results(star, wvl, flux_tot_ext, flux_norm_ext, conv_flux_norm, continuum_flux, label):
    """

    :param star:
    :param wvl:
    :param flux_tot_ext:
    :param flux_norm_ext:
    :param conv_flux_norm:
    :param continuum_flux:
    :param label:
    :return:
    """

    plt.figure(0)
    plt.title(star + " Continuum subtracted, extincted flux")
    plt.plot(wvl, flux_tot_ext - continuum_flux, label=label)
    plt.legend()

    plt.figure(1)
    plt.title(star + " Normalized flux")
    plt.plot(wvl, flux_norm_ext, label=label)
    plt.vlines([1.558, 1.5779, 1.5982], 0, 2, linestyles='dashed')
    # plt.ylim(0.99,1.12)
    plt.legend()

    plt.figure(3)
    plt.title(star + " Normalized, convolved flux")
    plt.plot(wvl, conv_flux_norm, label=label)
    # plt.ylim(0.99, 1.12)
    plt.legend()

    plt.figure(2)
    plt.title(star)
    plt.loglog(wvl, flux_tot_ext, label='total extincted flux; ' + label)

    return


def plot_t_structure_gas():

    plt.figure(5)
    r_tot = np.concatenate((R_CO_only, R_dust[mix]))
    t_tot = np.concatenate((T_gas, T_gas_mix))
    plt.loglog(r_tot / cfg.AU, t_tot, label=p)
    plt.loglog(R_dust[mix] / cfg.AU, T_dust, label="dust")
    plt.legend()


def plot_t_structure_dust(r_dust, t_dust):



    plt.figure(5)
    plt.loglog(R_dust[mix] / cfg.AU, T_dust, label="dust")
    plt.legend()
