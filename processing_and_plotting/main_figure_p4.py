import numpy as np
import matplotlib.pyplot as plt

import processing_and_plotting.plotting_routines as pltr
import model.config as cfg
import grids.list_of_grids as grids
import processing_and_plotting.Gridpoint as Gridpoint


# TODO plot a grid from a grid definition in list of grids.  Start with no 13CO

def make_fig():

    create_fig_and_plot()
    plt.show()


def param_seq_dict_t_new():
    # Attach an integer value to each parameter to ease iteration when plotting.
    return {0: "ti", 1: "t1", 2: "ni", 3: "q", 4: "ri", 5: "i"}


def create_figure(plot_count, ):
    """
    Create the figure ax object

    :param plot_count:(int) amount of parameters to be plotted (# "rows").
    :return:
    """

    n_regions = 3 # for 1st ot, 2nd ot and fundamental.
    ratios = np.ones(plot_count)
    gridspec = dict(hspace=0.0, height_ratios=ratios)
    f, ax = plt.subplots(plot_count, n_regions, facecolor='w', figsize=(12, n_regions * plot_count), sharey=True,
                         gridspec_kw=gridspec)

    # remove x labels
    for i in range(plot_count-1):
        for j in range(n_regions):
            ax[i, j].set_xticks([])

    annotate_transitions(ax)

    return f, ax


def annotate_transitions(axes):

    for ax in axes.flatten():
        for key, w in cfg.onset_wvl_dict.items():
            ax.annotate(key, xy=(w - 0.0045, 1.45), rotation=90, fontsize=6.5)
            ax.axvline(w, color='0.55', linewidth=0.2)


def fiducial_model(grid_params, all_params):


    gp_fid_model = Gridpoint.Gridpoint(**grid_params, all_params=all_params)
    ip, dx = gp_fid_model.return_instrument_profile()
    fid_model_flux = gp_fid_model.read_output_per_inc(convolve=True, ip_dx=(ip, dx))
    wvl = gp_fid_model.model_wvl_array()

    return fid_model_flux, wvl, (ip, dx)


def create_fig_and_plot():

    grid_params, all_params, test_param_dict = grids.grid_for_main_figure_p4()
    fid_mod_fl, wvl, ip_dx = fiducial_model(grid_params, all_params)
    f, axes = create_figure(plot_count=len(param_seq_dict_t_new()), )

    for index, param in param_seq_dict_t_new().items():
        pltr.plot_on_divided_axes(wvl, fid_mod_fl, fig_ax=(f, axes[index]), **dict(c="k") )


if __name__ == "__main__":
    make_fig()
