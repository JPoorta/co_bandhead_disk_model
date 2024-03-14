import matplotlib.pyplot as plt

import model.example as ex


def run_quick_test_grid(grid, **kwargs):
    """
    Run the grid not as a grid, but going through each test value separately as set in 'model.example'.

    :param grid: (list of three dicts) grid as defined in 'list_of_grids'. In all_params "convolve" must be set to True,
    and "save" to None. Pretty much any grid with the default grid params as defined in config should work.
    :param kwargs: parameters to be passed to 'ex.run_test', mainly quick_plots so far.
    :return:
    """

    grid_params, all_params, test_param_dict = grid

    for test_param, test_param_array in test_param_dict.items():
        ex.run_test(test_param, test_param_array, grid_params, all_params, **kwargs)

    plt.show()
    return


# TODO: define a function to save the grid for the main figure in P4.
