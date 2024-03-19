import numpy as np
import matplotlib.pyplot as plt

import processing_and_plotting.plotting_routines as pltr
import model.config as cfg
import grids.list_of_grids as grids
import processing_and_plotting.Gridpoint as Gridpoint


#TODO add 13CO, split figure in two 4, 3, with 13CO in the 3 plot, paste in thesis intro to see what it looks like
#TODO make figure with T extremes to justify chosen T1 range.

def make_fig():
    create_fig_and_plot()
    plt.show()


def param_seq_dict_t_new():
    # Attach an integer value to each parameter to ease iteration when plotting.
    return {0: "ti", 1: "t1", 2: "ni", 3: "q", 4: "ri", 5: "inc_deg"}


def create_figure(plot_count, ):
    """
    Create the figure ax object

    :param plot_count:(int) amount of parameters to be plotted (# "rows").
    :return:
    """

    n_regions = 3  # for 1st ot, 2nd ot and fundamental.
    ratios = np.ones(plot_count)
    gridspec = dict(hspace=0.0, height_ratios=ratios, width_ratios=[1.5, 2, 3])
    f, ax = plt.subplots(plot_count, n_regions, facecolor='w', figsize=(12, n_regions * plot_count), sharey=True,
                         gridspec_kw=gridspec)

    # remove x labels
    for i in range(plot_count - 1):
        for j in range(n_regions):
            ax[i, j].set_xticks([])

    annotate_transitions(ax)

    f.tight_layout()

    return f, ax


def annotate_transitions(axes):
    """
    Mark the vibrational transitions on the plot.

    :param axes:
    :return:
    """
    for ax in axes.flatten():
        for key, w in cfg.onset_wvl_dict.items():
            ax.annotate(key, xy=(w - 0.0045, 1.45), rotation=90, fontsize=6.5)
            ax.axvline(w, color='0.55', linewidth=0.2)


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
    else:
        grid_params_use = grid_params.copy()
        model_fluxes = {}
        for value in test_array:
            grid_params_use[test_param] = value
            gp = Gridpoint.Gridpoint(**grid_params_use, all_params=all_params)
            model_fluxes[value] = gp.read_output_per_inc(convolve=True, ip_dx=ip_dx)

    return model_fluxes


def create_fig_and_plot():
    """
    Bring the ingredients together and plot the fluxes.

    :return:
    """
    grid_params, all_params, test_param_dict = grids.grid_for_main_figure_p4()
    gp_fid_model, fid_mod_fl, wvl, ip_dx = fiducial_model(grid_params, all_params)
    f, axes = create_figure(plot_count=len(param_seq_dict_t_new()), )

    for index, param in param_seq_dict_t_new().items():

        model_fluxes = model_fluxes_for_test_param(param, test_param_dict[param], grid_params, all_params, ip_dx)

        for value in test_param_dict[param]:

            label = pltr.label_dict[param] + " = " + str(value)
            if param in ["inc_deg", "dv0", "stars"]:
                default_value = getattr(gp_fid_model, param)[0]
            else:
                default_value = getattr(gp_fid_model, param)
            if value == default_value:
                # plot the fiducial model on top in black.
                pltr.plot_on_divided_axes(wvl, fid_mod_fl, fig_ax=(f, axes[index]),
                                          **dict(c="k", lw=0.8, label=label, zorder=2))
            else:
                pltr.plot_on_divided_axes(wvl, model_fluxes[value], fig_ax=(f, axes[index]),
                                          **dict(lw=0.8, label=label, zorder=1))

    return


if __name__ == "__main__":
    make_fig()
