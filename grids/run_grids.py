import matplotlib.pyplot as plt

import processing_and_plotting.plotting_routines as pltr
import model.flat_disk_log_grid as fld
import model.example as ex
import model.config as cfg
import grids.list_of_grids as list_of_grids


def run_quick_test_grid(grid, quick_plots=False, plot_B275_checks=False, save=None, save_t_plot=None):
    """
    Run the grid not as a grid, but going through each test value separately as set in 'model.example'.

    :param grid: (list of three dicts) grid as defined in 'list_of_grids'. In all_params "convolve" must be set to True,
    and "save" to None. Pretty much any grid with the default grid params as defined in config should work.

    :return:
    """

    grid_params, all_params, test_param_dict = grid

    fig, ax = pltr.create_3_in_1_figure(num=3)
    for test_param, test_param_array in test_param_dict.items():
        gp, star, wvl, continuum_flux = ex.run_test(test_param, test_param_array, grid_params.copy(), all_params.copy(),
                                                    (fig, ax), quick_plots)

    ex.finish_plots(gp, star, wvl, continuum_flux, (fig, ax), quick_plots, plot_B275_checks)

    if save is not None:
        fig.savefig(cfg.plot_folder / (save + ".pdf"))
    if save_t_plot is not None:
        plt.figure(5)
        plt.savefig(cfg.plot_folder / (save_t_plot + ".pdf"))

    plt.show()

    return


def save_run_variation_around_one_model(grid=None):

    if grid is None:
        grid = list_of_grids.grid_for_main_figure_p4()

    grid_params, all_params, test_param_dict = grid

    for test_param, test_param_array in test_param_dict.items():
        grid_params_use = grid_params.copy()
        all_params_use = all_params.copy()
        if test_param in grid_params.keys():
            grid_params_use[test_param] = test_param_array
        elif test_param in all_params.keys():
            all_params_use[test_param] = test_param_array

        model_grid = ex.make_grid(**grid_params_use)
        fld.run_grid_log_r(model_grid, **all_params_use)

    return
