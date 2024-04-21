import numpy as np
import matplotlib.pyplot as plt

import processing_and_plotting.plotting_routines as pltr
import model.config as cfg
import grids.list_of_grids as grids
import processing_and_plotting.Gridpoint as Gridpoint


def make_fig():
    # create_fig_and_plot()
    create_split_fig()
    plt.show()


def param_seq_dict_t_new():
    # Attach an integer value to each parameter to ease iteration when plotting.
    return {0: "ti", 1: "t1", 2: "ni", 3: "q", 4: "ri", 5: "inc_deg"}


def legend_title_dict():
    return {"ti": r"$(T_i)_{\rm ex}$ (K)",
            "t1": r"$T_1$ (K)",
            "p": "$p$",
            "ni": r"$(N_i)_{\rm H_2}$ (cm$^{-2}$)",
            "q": "$q$",
            "ri": r"$R_i$ (AU)",
            "inc_deg": r"$i$ $(^{\circ})$ ",
            "13CO": r"$^{12}$CO/$^{13}$CO"}


def legend_def():
    return dict(loc='upper right', bbox_to_anchor=(1.05, 1.0), fontsize=9.3, title_fontsize=9.6)


def create_figure(plot_count, ratios=None, figsize=None, thirteen_co=True, **grid_spec_kw):
    """
     Create the figure ax object

    :param plot_count:(int) amount of parameters to be plotted (# "rows").
    :param ratios:
    :param figsize:
    :param thirteen_co:
    :param grid_spec_kw:
    :return:
    """

    n_regions = 3  # for 1st ot, 2nd ot and fundamental.
    if ratios is None:
        ratios = np.ones(plot_count)
        ratios[2] += 0.1  # make the plot for the density slightly higher
        ratios[-1] += 0.2  # make the plot for 13CO slightly higher
    if figsize is None:
        figsize = (12, 2 * plot_count)
    gridspec = dict(hspace=0.0, height_ratios=ratios, width_ratios=[1.5, 2, 3], **grid_spec_kw)
    f, ax = plt.subplots(plot_count, n_regions, facecolor='w', figsize=figsize, sharey=True,
                         gridspec_kw=gridspec)

    # remove x labels
    for i in range(plot_count - 1):
        for j in range(n_regions):
            ax[i, j].set_xticks([])

    annotate_transitions(ax, thirteen_co)

    # Series titles
    title_font_size = 9.5
    ax[0, 0].set_title(r"Second overtone ($\Delta v=3$)", fontsize=title_font_size)
    ax[0, 1].set_title(r"First overtone ($\Delta v=2$)", fontsize=title_font_size)
    ax[0, 2].set_title(r"Fundamental ($\Delta v=1$)", fontsize=title_font_size)

    f.tight_layout()

    return f, ax


def annotate_transitions(axes, thirteen_co=True):
    """
    Mark the vibrational transitions on the two plots.

    :param axes:
    :param thirteen_co: (bool) if True mark the 13CO transitions on the last 3-in-1 plot.
    :return:
    """

    def annotate_12_co(axi, key_i, wi):
        """

        :param axi: matplotlib ax object
        :param key_i: the text to annotate, in this case a tuple with two ints for vib levels.
        :param wi: wavelength
        :return:
        """
        axi.annotate(key_i, xy=(wi + 0.0045, 1.45), rotation=90, fontsize=7.5)
        axi.axvline(wi, color='0.55', linewidth=0.2)
        return

    for ax in axes.flatten()[:-3]:
        for key, w in cfg.onset_wvl_dict.items():
            annotate_12_co(ax, key, w)

    for ax in axes.flatten()[-3:]:
        if thirteen_co:
            for key, w in cfg.CO13_onset.items():
                ax.annotate(r"$^{13}$CO " + str(key), xy=(w + 0.0045, 1.45), rotation=90, fontsize=7.5)
                ax.axvline(w, color='0.15', linewidth=0.2)
        else:
            for key, w in cfg.onset_wvl_dict.items():
                annotate_12_co(ax, key, w)

    return


def fiducial_model(grid_params, all_params):
    """
    Given the grid, return variables related to the fiducial model.

    :param grid_params:
    :param all_params:
    :return: The Gridpoint object, the model flux and wavelength and the instrumental profile.
             The latter two are valid for all parameters.
    """
    gp_fid_model = Gridpoint.Gridpoint(**grid_params, all_params=all_params)
    ip, dx = gp_fid_model.return_instrument_profile()
    fid_model_flux = gp_fid_model.read_output_per_inc(convolve=True, ip_dx=(ip, dx))
    wvl = gp_fid_model.model_wvl_array()

    return gp_fid_model, fid_model_flux, wvl, (ip, dx)


def model_fluxes_for_test_param(test_param, test_array, grid_params, all_params, ip_dx):
    """
    Given a test parameter and its range of values, return the convolved model fluxes in a dictionary, with as keys the
    values the parameter takes; e.g. for "ti", the keys would be for instance 2000, 3000, 5000.

    :param test_param:
    :param test_array:
    :param grid_params:
    :param all_params:
    :param ip_dx:
    :return:
    """
    # Inclination is treated differently, because it is not a grid parameter.
    if test_param == "inc_deg":
        all_params_use = all_params.copy()
        all_params_use["inc_deg"] = test_array
        gp = Gridpoint.Gridpoint(**grid_params, all_params=all_params_use)
        model_fluxes = gp.read_output_per_inc(convolve=True, ip_dx=ip_dx)
        model_names = {key: gp.filename_co() for key in test_array}
    else:
        grid_params_use = grid_params.copy()
        model_fluxes = {}
        model_names = {}
        for value in test_array:
            grid_params_use[test_param] = value
            gp = Gridpoint.Gridpoint(**grid_params_use, all_params=all_params)
            model_fluxes[value] = gp.read_output_per_inc(convolve=True, ip_dx=ip_dx)
            model_names[value] = gp.filename_co()

    return model_fluxes, model_names


def plot_iteratively(grid, legend_dict, fig_ax, param_seq_dict):
    """
    Iterate over the parameters in param_seq_dict and plot the variation of models for each paramter on the given plot.

    :param grid:
    :param legend_dict:
    :param fig_ax:
    :param param_seq_dict:
    :return:
    """

    f, axes = fig_ax
    grid_params, all_params, test_param_dict = grid
    gp_fid_model, fid_mod_fl, wvl, ip_dx = fiducial_model(grid_params, all_params)

    for index, param in param_seq_dict:

        model_fluxes = model_fluxes_for_test_param(param, test_param_dict[param], grid_params, all_params, ip_dx)[0]
        legend_dict["title"] = legend_title_dict()[param]

        for value in test_param_dict[param]:

            label = str(value)
            if value == gp_fid_model.return_value(param):
                # plot the fiducial model on top in black.
                pltr.plot_on_divided_axes(wvl, fid_mod_fl, fig_ax=(f, axes[index]), legend_specs=legend_dict,
                                          **dict(c="k", lw=0.8, label=label, zorder=2))
            else:
                pltr.plot_on_divided_axes(wvl, model_fluxes[value], fig_ax=(f, axes[index]), legend_specs=legend_dict,
                                          **dict(lw=0.8, label=label, zorder=1))
    return


def create_fig_and_plot(grid=None):
    """
    Bring the ingredients together and plot the fluxes.

    :return:
    """
    if grid is None:
        grid = grids.grid_for_main_figure_p4()
    grid_params, all_params, test_param_dict = grid
    gp_fid_model, fid_mod_fl, wvl, ip_dx = fiducial_model(grid_params, all_params)
    f, axes = create_figure(plot_count=len(param_seq_dict_t_new()) + 1, )  # +1 for CO13
    legend_dict = legend_def()

    plot_iteratively(grid, legend_dict, (f, axes), param_seq_dict_t_new().items())

    legend_dict["title"] = legend_title_dict()["13CO"]
    plot_13co(gp_fid_model.path_grid_folder(), gp_fid_model.filename_co(), wvl, fid_mod_fl, fig_ax=(f, axes[-1]),
              legend_dict=legend_dict, **dict(lw=0.8, zorder=1))

    return


def plot_13co(folder, filename, wvl, fid_mod_fl=None, fig_ax=None, iso_ratios=None, legend_dict=None,
              fid_mod_pl_dict=None, label=None, **plot_kwargs):
    """
    Plot the models with different N12CO/N13CO ratios, for a model with otherwise same parameters (fiducial model).
    Optionally overplot the fiducial model with no 13CO.

    :param folder: (str) folder where both the fiducial model and 13CO included models are stored.
    :param filename: (str) the filename without file-ending ('npy') for the fiducial model.
    :param fid_mod_fl: (array) optional normalized fiducial model flux.
    :param wvl: (array) wavelength array in micron, common to all the models here.
    :param fig_ax: (tuple) figure and ax object to use. If not provided something will be created.
    :param iso_ratios: (list of ints) the isotopologue (N12CO/N13CO) ratios of the models.
    :param legend_dict: dictionary for the legend to be passed to plot on separate axes.
    :param fid_mod_pl_dict: (dict) dictionary with plot specs for the fiducial model.
    :param label: (str) label for the 13CO included plot(s).
    :return:
    """
    if iso_ratios is None:
        iso_ratios = grids.default_iso_ratios()

    for ratio in iso_ratios:
        full_path = folder + filename + "_13C16O_" + str(ratio) + ".npy"
        flux = np.load(full_path)
        if label is None:
            label = ratio
        plot_kwargs["label"] = label
        pltr.plot_on_divided_axes(wvl, flux, fig_ax=fig_ax, legend_specs=legend_dict, **plot_kwargs)

    if fid_mod_fl is not None:
        if fid_mod_pl_dict is None:
            fid_mod_pl_dict = dict(lw=0.8, c="k", label=r"no $^{13}$CO", zorder=2)
        pltr.plot_on_divided_axes(wvl, fid_mod_fl, fig_ax=fig_ax, legend_specs=legend_dict,
                                  **fid_mod_pl_dict)

    return


def create_split_fig(grid=None):
    """
    Create the main figure for paper 4 split over two figures.

    :return:
    """
    if grid is None:
        grid = grids.grid_for_main_figure_p4()
    grid_params, all_params, test_param_dict = grid
    gp_fid_model, fid_mod_fl, wvl, ip_dx = fiducial_model(grid_params, all_params)

    plot_count1 = int(len(param_seq_dict_t_new()) / 2 + 1)
    ratios_1 = np.ones(plot_count1)
    plot_count2 = int(len(param_seq_dict_t_new()) / 2)
    ratios_2 = np.ones(plot_count2)
    ratios_2[-1] += 0.2
    width = 10
    height = width / 1.54  # ~A5 proportions
    f1, axes1 = create_figure(plot_count1, ratios_1, figsize=(width, height), thirteen_co=False,
                              **dict(top=0.963,
                                     bottom=0.07,
                                     left=0.035,
                                     right=0.962,
                                     wspace=0.053))
    f2, axes2 = create_figure(plot_count2, ratios_2, figsize=(width, height),
                              **dict(top=0.963,
                                     bottom=0.07,
                                     left=0.035,
                                     right=0.962,
                                     wspace=0.053))

    legend_dict = legend_def()

    # split the dictionary according to plot count

    dict_to_list = list(param_seq_dict_t_new().items())
    items_fig_1 = dict_to_list[:plot_count1]
    dict_fig_2 = {key - plot_count1: value for key, value in dict_to_list[plot_count1:]}
    plot_iteratively(grid, legend_dict, (f1, axes1), items_fig_1)
    plot_iteratively(grid, legend_dict, (f2, axes2), dict_fig_2.items())

    legend_dict["title"] = legend_title_dict()["13CO"]
    plot_13co(gp_fid_model.path_grid_folder(), gp_fid_model.filename_co(), wvl, fid_mod_fl, fig_ax=(f2, axes2[-1]),
              legend_dict=legend_dict, **dict(lw=0.8, zorder=1))

    f1.savefig(cfg.plot_folder / "main_fig_1.pdf")
    f2.savefig(cfg.plot_folder / "main_fig_2.pdf")
    return


if __name__ == "__main__":
    make_fig()
