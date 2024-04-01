import matplotlib.pyplot as plt

import grids.run_grids as run_gr
import grids.list_of_grids as gr_list
import processing_and_plotting.main_figure_p4 as main_fig
import processing_and_plotting.plotting_routines as pltr


def run():
    test_dict = {"t1": [1000, 500, 800, ]}
    save_mod = "var_t1"
    save_t_plot = "t_strct_range_t1"
    make_figs(test_dict, save_mod, save_t_plot)
    # plot_mini_grid(gr_list.common_test_grid_save())
    plt.show()
    return


def make_figs(test_dict=None, save_model_plot=None, save_t_plot=None):
    """
    Make model and temperature plots of unsaved models for paper 4. Unsaved means the models will be calculated.
    The calculated models will be overplotted with B275 original data and fits.
    All is plotted in X-shooter resolution.
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
    run_gr.run_quick_test_grid(gr_list.common_test_grid_original_t(test_dict), save=save_model_plot,
                               save_t_plot=save_t_plot)
    return


def plot_mini_grid(grid, num=1):
    """
    Plot a grid on one plot. (Try out version, currently not used anywhere.)

    :return:
    """

    grid_params, all_params, test_param_dict = grid
    gp_fid_model, fid_mod_fl, wvl, ip_dx = main_fig.fiducial_model(grid_params, all_params)
    legend_dict = main_fig.legend_def()

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


if __name__ == "__main__":
    run()
