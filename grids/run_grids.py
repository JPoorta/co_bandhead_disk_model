import matplotlib.pyplot as plt

import processing_and_plotting.plotting_routines as pltr
import model.example as ex


def run_quick_test_grid(grid, quick_plots=False, plot_B275_checks=False):
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

    plt.show()


    return


# TODO: define a function to save the grid for the main figure in P4.
