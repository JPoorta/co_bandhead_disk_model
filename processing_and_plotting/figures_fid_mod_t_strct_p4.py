import matplotlib.pyplot as plt

import model.config as cfg
import grids.run_grids as run_gr
import grids.list_of_grids as gr_list
import processing_and_plotting.Gridpoint as Gridpoint
import processing_and_plotting.main_figure_p4 as main_fig
import processing_and_plotting.plotting_routines as pltr


def run():
    # make_figs_B243()
    plot_prediction_incl_13_co(gr_list.fid_model_B243())
    plt.show()
    return

def make_figs_B243():
    test_dict = {"stars": ["B243"]}
    grid = gr_list.fid_model_B243(test_dict)
    save_mod = "B243_fid_mod"
    save_t_plot = "t_strct_B243"
    make_figs(grid=grid, test_dict=test_dict, save_model_plot=save_mod, save_t_plot=save_t_plot)
    return

def make_figs_B268():
    test_dict = {"stars": ["B268"]}
    grid = gr_list.fid_model_B268(test_dict)
    save_mod = "B268_fid_mod"
    save_t_plot = "t_strct_B268"
    make_figs(grid=grid, test_dict=test_dict, save_model_plot=save_mod, save_t_plot=save_t_plot)
    return


def make_figs_var_t1():
    test_dict = {"t1": [1000, 800, 500, ]}
    save_mod = "var_t1"
    save_t_plot = "t_strct_range_t1"
    make_figs(test_dict, save_mod, save_t_plot)
    return


def make_figs(grid=None, test_dict=None, save_model_plot=None, save_t_plot=None):
    """
    Make model and temperature plots of unsaved models for paper 4. Unsaved means the models will be calculated.
    The calculated models will be overplotted with original data and fits of the relevant object.
    All is plotted in JWST resolution (this can now only be changes in config!).
    Save plots to the plot folder.

    :param test_dict: (dict) Parameters and their value(s) to be varied/shown on the label(s).
    :param save_model_plot: (str) name of the model plot.
    :param save_t_plot: (str) name of the temperature plot.
    :return:
    """
    if test_dict is None:
        test_dict = {"t1": [800],
                     "p": [-2]}
    if save_model_plot is None:
        save_model_plot = "fiducial_model"
    if save_t_plot is None:
        save_t_plot = "t_strct"
    if grid is None:
        grid = gr_list.common_test_grid_original_t(test_dict)
    run_gr.run_quick_test_grid(grid, save=save_model_plot, save_t_plot=save_t_plot)
    return


def plot_mini_grid(grid, num=1, fig_ax=None):
    """
    Plot a grid on one plot. (Try out version, currently not used anywhere.)

    :return:
    """

    grid_params, all_params, test_param_dict = grid
    gp_fid_model, fid_mod_fl, wvl, ip_dx = main_fig.fiducial_model(grid_params, all_params)
    legend_dict = main_fig.legend_def()

    if fig_ax is None:
        fig_ax = pltr.create_3_in_1_figure(num)

    for param, test_array in test_param_dict.items():
        model_fluxes = \
            main_fig.model_fluxes_for_test_param(param, test_param_dict[param], grid_params, all_params, ip_dx)[0]
        legend_dict["title"] = main_fig.legend_title_dict()[param]

        for value in test_array:
            if value == gp_fid_model.return_value(param):
                pltr.plot_on_divided_axes(wvl, model_fluxes[value], fig_ax=fig_ax, legend_specs=legend_dict,
                                          **dict(lw=0.8, c="k", label=str(value), zorder=2))
            else:
                pltr.plot_on_divided_axes(wvl, model_fluxes[value], fig_ax=fig_ax, legend_specs=legend_dict,
                                          **dict(lw=0.8, label=str(value), zorder=1))

    return


def plot_prediction_incl_13_co(grid, abundance=89):
    """
    Plot the fundamental prediction for an object with previously fit overtones.
    Conditional upon the existence of a saved grid both without 13CO and with 13CO in the specified abundance.

    :param grid:
    :param abundance:
    :return:
    """

    fig, ax = pltr.create_3_in_1_figure(1)
    grid_params, all_params, test_param_dict = grid
    gp = Gridpoint.Gridpoint(**grid_params, all_params=all_params)
    filename = gp.filename_co()
    gp_fid_model, fid_mod_fl, wvl, ip_dx = main_fig.fiducial_model(grid_params, all_params)
    main_fig.plot_13co(gp.path_grid_folder(), filename, wvl=gp.model_wvl_array(), fig_ax=(fig, ax),
                       fid_mod_fl=fid_mod_fl, fid_mod_pl_dict=dict(lw=0.8, label=r"no $^{13}$CO", zorder=-1),
                       iso_ratios=[abundance], label=r"$\rm N(^{12}CO)/N(^{13}CO)=$"+str(abundance),
                       **dict(lw=0.8, zorder=3, ))
    pltr.plot_obs_spectrum(gp.star, fig_ax=(fig, ax))
    ax[1].set_title(gp.star, loc="left", pad=-18, fontsize=15)

    fig.savefig(cfg.plot_folder / (gp.star+"_fid_mod_13co.pdf"))

    return

if __name__ == "__main__":
    run()
