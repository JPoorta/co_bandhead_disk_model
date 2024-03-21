import model.config as cfg


def vibrational_levels_for_3_ro_vib_series():
    """
    Define the vibrational line levels for including the fundamental and first and second overtones.

    :return:
    """
    vupper = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    vlower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
              0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    return vupper, vlower


def default_iso_ratios():
    """
    Default isotopologue ratio's for 13CO.

    :return:
    """
    return [4, 10, 30, 50, 89]


def grid_including_13co(species, iso_ratio, grid=None):
    """
    For a given grid, adjust the saving folder, the isotopologue, and the iso_ratio parameters. Return the updated grid.

    :param species: (str) name of isotopologue
    :param iso_ratio: (int) N12CO/Nisotopologue (only 13CO so far)
    :param grid: (tuple) the three dictionaries containing all that specifies a grid. Default is the one for the main
    figure paper 4.
    :return:
    """

    if grid is None:
        grid = grid_for_main_figure_p4()

    grid_params, all_params, test_param_dict = [x.copy() for x in grid]

    all_params["species"] = species
    all_params["save"] += "_" + species + "_" + str(iso_ratio)
    all_params["iso_ratio"] = iso_ratio

    return grid_params, all_params, test_param_dict


def grid_for_main_figure_p4(star=None):
    """


    :param star:
    :return:
    """

    if star is None:
        star = "B275"

    grid_params, all_params = cfg.get_default_params(star)

    # Set the parameters to be tested.
    test_param_dict = {"t1": [600, 700, 800, 900],
                       "ti": [2000, 3000, 4000, 5000],
                       "ni": [5e23, 3.9e24, 3e25, 6.5e26, 5e27],
                       "q": [1, -0.5, -1, -1.5],
                       "ri": [0.261, 0.5, 1],
                       "inc_deg": [20, 30, 40, 60, 80]
                       }

    # Adjust defaults if different from defined in config.
    vupper, vlower = vibrational_levels_for_3_ro_vib_series()
    all_params["vupper"] = vupper
    all_params["vlower"] = vlower
    all_params["Rmax_in"] = 100
    all_params["dF"] = None
    all_params["num_CO"] = 100
    all_params["convolve"] = False
    all_params["maxmin"] = [5, -5]
    all_params["print_Rs"] = False
    all_params["save"] = "main_fig_p4"
    all_params["save_reduced_flux"] = False

    return grid_params, all_params, test_param_dict


def common_test_grid(test_param_dict=None):
    """
    Grid most commonly used in preparation for P4 to test stuff with adaptable test parameter grid.
    Note that the arrays for cumulative flux calculation are being saved for this grid.

    :return:
    """

    grid_params, all_params = cfg.get_default_params("B275")
    if test_param_dict is None:
        test_param_dict = {"t1": [700, 800]}
    vupper, vlower = vibrational_levels_for_3_ro_vib_series()
    all_params["vupper"] = vupper
    all_params["vlower"] = vlower
    all_params["Rmax_in"] = 100
    all_params["dF"] = ""

    return grid_params, all_params, test_param_dict


def common_test_grid_original_t(test_param_dict=None):
    """
    Grid to show what the effect is of the original temperature structure.
    Note that the outer disk radius "Rmax_in" is *not* the original value 'None'.

    :return:
    """

    grid_params, all_params = cfg.get_default_params("B275")
    if test_param_dict is None:
        test_param_dict = {"dust": [True, False]}
    vupper, vlower = vibrational_levels_for_3_ro_vib_series()
    all_params["vupper"] = vupper
    all_params["vlower"] = vlower
    grid_params["t1"] = None
    grid_params["a"] = -11
    all_params["Rmax_in"] = 100

    return grid_params, all_params, test_param_dict
