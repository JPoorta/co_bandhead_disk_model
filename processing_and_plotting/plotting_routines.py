import matplotlib.pyplot as plt
import numpy as np

import model.config as cfg
import model.sed_calculations as seds
import processing_and_plotting.Gridpoint as Gridpoint
from model.flat_disk_log_grid import create_t_gas_array, create_radial_array


label_dict = {"ri": r"$R_i$",
              "ti": r"$(T_i)_{\rm ex}$",
              "p": r"$p$",
              "ni": r"$(N_i)_{\rm H_2}$",
              "q": r"$q$",
              "t1": r"$T_1$",
              "a": r"$a$",
              "inc_deg": r"$i$",
              "tex": r"$T_{\rm ex}$",
              "R": r"$R$ (AU)"}


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


def plot_obs_spectrum(star, fig_ax=None):
    """
    Function to read and plot the observed and fitted spectra of CO paper I (Poorta et al., 2023)

    :param star: (str)
    :param fig_ax: Optional fig and ax objects to plot on.
    :return:
    """
    wvl_obj, flux_obj, wvl, conv_flux, fit_mask = np.load(str(cfg.spectra_dir) + "/" + star + ".npy", allow_pickle=True)
    plot_on_divided_axes(wvl_obj, flux_obj, fig_ax=fig_ax,
                         **{"c": 'k', "label": "data", "zorder": -2, "alpha": 0.9, "lw": 0.6})
    plot_on_divided_axes(wvl[:-9], conv_flux[:-9], fig_ax=fig_ax, **{"c": 'r', "label": "best fit", "lw": 0.6, "alpha": 1})

    return


def plot_275_checks(fig_ax=None, no_dust=False, rmax_in=False):
    """
    Plots the original best fit model for B275 after ALMA dust SED implementation.

    :param wvl: model wavelength array
    :param fig_ax: optional matplotlib figure and ax objects to plot on.
    :param no_dust: (Bool) if True plot the original dust=False option, in which the spectrum was only normalized
    to stellar continuum.
    :param rmax_in: (Bool) if True plot the best fit model after ALMA dust SED, after R_dust was removed from
    flat_disk_log_grid, with extended disk upto 60 AU (Rmax_in=100; defaults to ALMA outer disk radius). This model
    gives some absorption, and otherwise differs minimally from alma_dust because the 100 radial points (num_CO)
    now run upto 60 AU instead of ~0.67 AU.
    :return:
    """
    wvl = np.load(cfg.spectra_dir / "wvl.npy")
    alma_dust = np.load(cfg.spectra_dir / "B275_alma_dust.npy")
    plot_on_divided_axes(wvl, alma_dust, fig_ax=fig_ax, **{"label": r"original $T_{\rm ex}$", "zorder": -1})
    if no_dust:
        no_dust = np.load(cfg.spectra_dir / "B275_no_dust.npy")
        plot_on_divided_axes(wvl, no_dust, fig_ax=fig_ax, **{"label": "no dust", "zorder": -1})
    if rmax_in:
        rmax_in100 = np.load(cfg.spectra_dir / "B275_Rmax_in_100.npy")
        plot_on_divided_axes(wvl, rmax_in100, fig_ax=fig_ax, **{"label": "Rmax_in=100 (Rmax=60)", "zorder": -1})


def quick_plot_results(star, wvl, flux_tot_ext, flux_norm_ext, continuum_flux, label):
    """
    Make some quick diagnostic plots of a model just run.

    :param star:
    :param wvl:
    :param flux_tot_ext:
    :param flux_norm_ext:
    :param continuum_flux:
    :param label:
    :return:
    """
    quick_plot_cont_subtr_ext_flux(star, wvl, flux_tot_ext, continuum_flux, label)
    quick_plot_norm_flux(star, wvl, flux_norm_ext, label)
    quick_plot_total_ext_flux(star, wvl, flux_tot_ext, label)

    return


def quick_plot_cont_subtr_ext_flux(star, wvl, flux_tot_ext, continuum_flux, label):
    plt.figure(0)
    plt.title(star + " Continuum subtracted, extincted flux")
    plt.plot(wvl, flux_tot_ext - continuum_flux, label=label)
    plt.legend()
    return


def quick_plot_norm_flux(star, wvl, flux_norm_ext, label):
    plt.figure(1)
    plt.title(star + " Normalized flux")
    plt.plot(wvl, flux_norm_ext, label=label)
    plt.legend()
    return


def quick_plot_total_ext_flux(star, wvl, flux_tot_ext, label):
    plt.figure(2)
    plt.title(star)
    plt.loglog(wvl, flux_tot_ext, label='total extincted flux; ' + label)
    return


def quick_plot_norm_convolved_flux(star, wvl, conv_flux_norm, label, fig_ax=None):
    title = star + " Normalized, convolved flux"
    plot_on_divided_axes(wvl, conv_flux_norm, fig_ax=fig_ax, title=title, **{"label": label, "zorder": 0,
                                                                             "lw": 0.7})

    return


def create_3_in_1_figure(num):
    """
    Return a figure with figure number `num`, with 3 subplots to plot the three parts of the CO spectrum on.

    :return:
    """
    return plt.subplots(1, 3, figsize=(11, 2.6), num=num, sharey=True,
                        gridspec_kw={"width_ratios": [1.5, 3, 3],
                                     "top": 0.975,
                                     "bottom": 0.20,
                                     "left": 0.035,
                                     "right": 0.965,
                                     "hspace": 0.2,
                                     "wspace": 0.045})


def plot_on_divided_axes(x, y, num=3, fig_ax=None, title=None, legend_specs=None, **kwargs):
    """
    Plot the given x,y on a plot showing only the wavelength regions of interest, that is, second, first overtone and
    fundamental. Also plots the legend.

    :param x: (array) x data, should be wavelength in micron.
    :param y: (array) y data, should be normalized flux.
    :param num (int) figure number. Defaults to three.
    :param fig_ax: (fig, ax) objects. If not provided will be created using 'create_3_in_1_figure'.
    :param title: (str) title of the figure.
    :param kwargs: arguments to pass on to plot, color, zorder etc.
    :return: The fig and ax objects for further plotting.
    """

    if fig_ax is None:
        fig, ax = create_3_in_1_figure(num)
    else:
        fig, ax = fig_ax

    if legend_specs is None:
        legend_specs=dict(loc='upper right')

    fig.suptitle(title)
    for axi in ax:
        axi.plot(x, y, **kwargs)
        axi.set_xlabel(r"$\lambda (\mu$m)")

    ax[0].set_xlim(1.55, 1.70)  # 1.75
    ax[1].set_xlim(2.285, 2.6)  # 2.9
    ax[2].set_xlim(4.28, 5.3)
    ax[0].set_ylim(0.95, 1.5)

    ax[0].spines['right'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(labelright=False, right=False, left=False, labelleft=False)
    ax[1].spines['left'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[2].tick_params(labelright=True, right=True, left=False, labelleft=False)

    ax[1].legend(**legend_specs)   #  bbox_to_anchor=(-0.05, 1)

    return fig, ax


def plot_t_structure_gas(gp, ):
    """
    Using the 'obtain_model_arrays_from_params' function plot the CO temperature array.

    :param gp: Gridpoint object
    :return:
    """
    r_co, co_only = gp.obtain_radial_model_array()
    t_gas = gp.obtain_model_t_gas()

    plt.figure(5, figsize=(5, 4))
    label_base = label_dict["ti"] + "=" + str(gp.ti) + " " + label_dict["ri"] + "=" + str(gp.ri) + " "
    if None in [gp.t1, gp.a]:
        label = label_base + label_dict["p"] + "=" + str(gp.p)
    else:
        label = label_base + label_dict["t1"] + "=" + str(gp.t1) + " " + label_dict["a"] + "=" + str(gp.a)
    plt.loglog(r_co / cfg.AU, t_gas, label=label)

    plt.ylabel(r"$T$ (K)", fontsize=11)
    plt.xlabel(label_dict["R"], fontsize=11)
    plt.legend()
    plt.tight_layout()

    return


def plot_t_structure_dust(gp):
    """
    Using the 'obtain_model_arrays_from_params' function plot the dust temperature array.

    :param gp: Gridpoint object
    :return:
    """
    r_co, co_only = gp.obtain_radial_model_array()
    t_dust = gp.obtain_model_t_dust()
    plt.figure(5)
    plt.loglog(r_co[~co_only] / cfg.AU, t_dust, label="dust")
    plt.legend()
    return


def plot_t_structure_original(star):
    """
    Using the parameters of the original best fit model of an object, recreate the temperature array.

    :param star:
    :return:
    """

    ri_d_au = cfg.best_dust_fit_ALMA[star][6]
    ti, t_i_err, p, n_h_i, n_h_i_err, ri_au, inc, v_g = cfg.best_fit_params[star]
    rmax, rmin, ri, ri_d, r_co, co_only = \
        create_radial_array(star, ri_au, rmax_in=None, rmin_in=None, ri_d_au=ri_d_au, r_out_au=None, ti=ti, num_co=100)
    t_gas = create_t_gas_array(r_co, co_only, ti, ri, t1=None, a=None, p=p, p_d=None)

    plt.figure(5)
    plt.loglog(r_co / cfg.AU, t_gas, linestyle='--', c='k', label="original power law")
    plt.legend()
    return


def plot_cum_flux(gp, fig_ax=None):
    """

    :param gp: (Gridpoint object) defines the model,
    :return:
    """

    dF_disk, r_disk_AU = gp.return_cum_flux()

    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig, ax = fig_ax
    label_base = gp.test_param + " = " + str(gp.test_value)
    p = ax.semilogx(r_disk_AU, dF_disk[0, :, 2], label=label_base + "; fund")[0]
    ax.semilogx(r_disk_AU, dF_disk[0, :, 1], '--', c=p.get_color(), label=label_base + "; 1st ot")
    ax.semilogx(r_disk_AU, dF_disk[0, :, 0], ':', c=p.get_color(), label=label_base + "; 2nd ot")

    return


def plot_cum_flux_grid(grid):
    """
    Plot the cumulative flux plot of a grid as defines in list_of_grids.  Only do this for varied parameters in the
    grid, so not (yet) for parameters defined in all_params, such as inc, dv0. etc.

    :param grid: (list of three dicts) grid as defined in 'list_of_grids'. This grid must already have been run and
    saved through the parameter "dF".
    :return:
    """

    grid_params, all_params, test_param_dict = grid
    gridspec = dict(hspace=0, wspace=0)
    fig, axes = plt.subplots(2,2, figsize=(10, 8), sharex=True, sharey=True, constrained_layout=True,
                             gridspec_kw=gridspec)

    i = 0  # ax index
    for test_param, test_param_array in test_param_dict.items():
        if test_param in ["ti", "ni", "q", "t1"]:
            ax = axes.flatten()[i]
            for value in test_param_array:
                grid_params_use = grid_params.copy()
                if test_param in grid_params.keys():
                    grid_params_use[test_param] = test_param_array
                grid_params_use[test_param] = value
                gp = Gridpoint.Gridpoint(**grid_params_use, all_params=all_params,
                                         test_param=test_param, test_value=value)
                plot_cum_flux(gp, fig_ax=(fig, ax))
            ax.set_title(label_dict[test_param], )
            ax.legend(loc="upper right")
            ax.set_ylim(0,1.05)
            i += 1
    axes[0][0].set_ylabel(r"$dF/F_{\rm tot}$")
    axes[1][0].set_ylabel(r"$dF/F_{\rm tot}$")
    axes[1][0].set_xlabel(r"$R$ (AU)")
    axes[1][1].set_xlabel(r"$R$ (AU)")
    fig.savefig(cfg.plot_folder / "df_plots.pdf")
    plt.show()

    return
