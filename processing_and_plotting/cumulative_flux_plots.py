import matplotlib.pyplot as plt

import model.config as cfg
import grids.list_of_grids as gr_list
import processing_and_plotting.plotting_routines as pltr
import processing_and_plotting.Gridpoint as Gridpoint
from processing_and_plotting.main_figure_p4 import legend_title_dict


color_dict = {"ti": {3000: "tab:orange", 4000: "k", 5000: "tab:green"},
              "t1": {700: "tab:orange", 800: "k", 900: "tab:green"},
              "ni": {3.9e24: "tab:orange", 3e25: "k", 6.5e26: "tab:green"},
              "q": {-0.5: "tab:orange", -1: "tab:green", -1.5: "k"}}


def plot_cum_flux_different_objects():
    """
    Plot (and save) the cumulative flux for the objects B275, B243 and B268 for paper4.

    :return:
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    plot_cum_flux_grid(gr_list.common_test_grid(test_param_dict={"dust": [True]}), fig_ax=(fig, ax),
                       label_base="B275")
    plot_cum_flux_grid(gr_list.fid_model_B243(), fig_ax=(fig, ax), label_base="B243")
    plot_cum_flux_grid(gr_list.fid_model_B268(), fig_ax=(fig, ax), label_base="B268")
    plt.ylabel(r"$\frac{dF}{F_{\rm tot}}$", fontsize=15)
    plt.xlabel(r"$R$ (AU)", fontsize=12)
    plt.ylim(0, 1.05)
    ax.axvline(1.7, color="tab:orange", linewidth=1, )
    ax.axvline(1.8, color="tab:green", linewidth=1, )
    ax.axvline(4.1, color="tab:blue", linewidth=1, )
    plt.tight_layout()
    plt.savefig(cfg.plot_folder / "df_plots_diff_obj.pdf")
    plt.show()
    return


def plot_cum_flux_grid(grid, fig_ax, label_base=None):
    """
    Plot the cumulative flux for a grid. (dF option should be specified in grid for this to work.)

    :param grid:
    :param fig_ax:
    :param label_base:
    :return:
    """
    grid_params, all_params, test_param_dict = grid

    for test_param, test_param_array in test_param_dict.items():

        for value in test_param_array:
            if test_param in grid_params:
                grid_params_use = grid_params.copy()
                grid_params_use[test_param] = value
                gp = Gridpoint.Gridpoint(**grid_params_use, all_params=all_params,
                                         test_param=test_param, test_value=value)
            elif test_param in all_params:
                all_params_use = all_params.copy()
                all_params_use[test_param] = value
                gp = Gridpoint.Gridpoint(**grid_params, all_params=all_params_use,
                                         test_param=test_param, test_value=value)
            pltr.plot_cum_flux(gp, fig_ax=fig_ax, label_base=label_base)
            plt.legend()
    return


def plot_cum_flux_main_fig(grid):
    """
    Plot the cumulative flux plot for grid_for_main_figure_p4.  Only do this for varied parameters in the
    grid, so not (yet) for parameters defined in all_params, such as inc, dv0. etc.

    :param grid: (list of three dicts) grid as defined in 'list_of_grids'. This grid must already have been run and
    saved through the parameter "dF".
    :return:
    """

    grid_params, all_params, test_param_dict = grid
    gridspec = dict(hspace=0, wspace=0, top=0.995, bottom=0.065, left=0.075, right=0.995, )
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, gridspec_kw=gridspec)

    i = 0  # ax index
    for test_param, test_param_array in test_param_dict.items():
        if test_param in ["ti", "ni", "q", "t1"]:
            ax = axes.flatten()[i]
            if test_param == "ni":
                test_array = test_param_array[1:-1]
            else:
                test_array = test_param_array[1:]
            for value in test_array:
                grid_params_use = grid_params.copy()
                grid_params_use[test_param] = value
                gp = Gridpoint.Gridpoint(**grid_params_use, all_params=all_params,
                                         test_param=test_param, test_value=value)
                pltr.plot_cum_flux(gp, fig_ax=(fig, ax), label_base=str(gp.test_value),
                                   color=color_dict[test_param][value])
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95), fontsize=10, title_fontsize=12,
                      title=legend_title_dict()[test_param])
            ax.set_ylim(0, 1.05)
            i += 1

    label_fs = 15
    axes[0][0].set_ylabel(r"$\frac{dF}{F_{\rm tot}}$", fontsize=label_fs)
    axes[1][0].set_ylabel(r"$\frac{dF}{F_{\rm tot}}$", fontsize=label_fs)
    axes[1][0].set_xlabel(r"$R$ (AU)", fontsize=12)
    axes[1][1].set_xlabel(r"$R$ (AU)", fontsize=12)
    fig.savefig(cfg.plot_folder / "df_plots.pdf")
    plt.show()

    return


if __name__ == "__main__":
    plot_cum_flux_different_objects()