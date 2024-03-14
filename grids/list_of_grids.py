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
                       "ni": [5e23, 3.9e24, 8.3e25, 6.5e26, 5e27],
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
    all_params["save"] = "main_fig_p4"

    return grid_params, all_params, test_param_dict


def common_test_grid(test_param_dict=None):
    """
    Grid most commonly used in preparation for P4 to test stuff.

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
