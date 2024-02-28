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


def plot_275_checks(wvl, no_dust=False, rmax_in=False):
    """
    Plots the original best fit model for B275 after ALMA dust SED implementation.

    :param wvl: model wavelength array
    :param no_dust: (Bool) if True plot the original dust=False option, in which the spectrum was only normalized
    to stellar continuum.
    :param rmax_in: (Bool) if True plot the best fit model after ALMA dust SED, after R_dust was removed from
    flat_disk_log_grid, with extended disk upto 60 AU (Rmax_in=100; defaults to ALMA outer disk radius). This model
    gives some absorption, and otherwise differs minimally from alma_dust because the 100 radial points (num_CO)
    now run upto 60 AU instead of ~0.67 AU.
    :return:
    """

    alma_dust = np.load(cfg.spectra_dir / "B275_alma_dust.npy")
    plt.plot(wvl, alma_dust, label="alma_dust", zorder=-1)
    if no_dust:
        no_dust = np.load(cfg.spectra_dir / "B275_no_dust.npy")
        plt.plot(wvl, no_dust, label="no dust", zorder=-1)
    if rmax_in:
        rmax_in100 = np.load(cfg.spectra_dir / "B275_Rmax_in_100.npy")
        plt.plot(wvl, rmax_in100, label="Rmax_in=100 (Rmax=60)", zorder=-1)
    plt.legend()


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


def plot_t_structure_gas(r_co, **grid_params):


    plt.loglog(r_co / cfg.AU, T_gas, label=p)

    plt.legend()


def plot_t_structure_dust(r_co, mix, t_dust):

    plt.loglog(R_CO[mix] / cfg.AU, T_dust, label="dust")
    p_plot = cfg.best_fit_params[st][2]
    plt.loglog(R_CO / cfg.AU, cfg.t_simple_power_law(R_CO, ti, ri, p_plot), label="p=" + str(p_plot))

    plt.figure(5)
    plt.loglog(R_dust[mix] / cfg.AU, T_dust, label="dust")
    plt.legend()
