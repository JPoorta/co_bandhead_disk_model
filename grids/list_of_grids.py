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
    :param iso_ratio: (int) N12CO/N(isotopologue) (only 13CO so far)
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
    Varying the parameter space around 1 fiducial model. Both the cumulative flux arrays and the final models are saved.

    :param star:
    :return:
    """

    if star is None:
        star = "B275"

    grid_params, all_params = cfg.get_default_params(star)

    # Set the parameters to be tested.
    test_param_dict = {"ti": [2000, 3000, 4000, 5000],
                       "t1": [600, 700, 800, 900],
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
    all_params["dF"] = ""
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
    Note that the arrays for cumulative flux calculation are being saved for this grid, but not the final models.

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


def common_test_grid_save(test_param_dict=None):
    """
    Used for P4 to create the variation of T1 figure.
    The model fluxes are being saved for this grid, but not the cum flux arrays.

    :return:
    """

    grid_params, all_params = cfg.get_default_params("B275")
    if test_param_dict is None:
        test_param_dict = {"t1": [500, 800, 1000, 1200]}
    vupper, vlower = vibrational_levels_for_3_ro_vib_series()
    all_params["vupper"] = vupper
    all_params["vlower"] = vlower
    all_params["Rmax_in"] = 100
    all_params["convolve"] = False
    all_params["maxmin"] = [100, -100]
    all_params["print_Rs"] = False
    all_params["save_reduced_flux"] = False
    all_params["save"] = "common_test"

    return grid_params, all_params, test_param_dict


def plots_for_intro_grid():
    """
    Grid to create plots for thesis introduction.

    :return:
    """
    star = "B275"
    grid_params, all_params = cfg.get_default_params(star)

    test_param_dict = {"Rmin_in": [0.261]}  # , 0.446, 0.667

    all_params["vupper"] = [1, 2, 3]  # [2] for stages in bh plot
    all_params["vlower"] = [0, 0, 0]  # [0] for stages in bh plot
    all_params["dust"] = True
    grid_params["p"] = -2
    grid_params["t1"] = None  # 800  #
    grid_params["a"] = None  # -11  #
    grid_params["ri"] = 0.261
    all_params["Rmax_in"] = None  # all_params["Rmin_in"]+0.001 for ring in stages of bh plot
    all_params["dF"] = None  # ""
    all_params["num_CO"] = 5  # for the tau for different temperatures;
    # all_params["num_CO"] = 2 for ring in stages of bh plot

    return grid_params, all_params, test_param_dict


def fid_model_B243(test_param_dict=None):
    grid_params, all_params = cfg.get_default_params("B243")
    if test_param_dict is None:
        test_param_dict = {"t1": [800]}
    vupper, vlower = vibrational_levels_for_3_ro_vib_series()
    grid_params["t1"] = 800
    grid_params["a"] = -11
    all_params["vupper"] = vupper
    all_params["vlower"] = vlower
    all_params["Rmax_in"] = 100
    all_params["dF"] = ""
    all_params["dust"] = True
    all_params["convolve"] = False
    all_params["maxmin"] = [100, -100]
    all_params["print_Rs"] = False
    all_params["save"] = "prediction_p4"
    all_params["save_reduced_flux"] = False

    return grid_params, all_params, test_param_dict


def fid_model_B268(test_param_dict=None):
    grid_params, all_params = cfg.get_default_params("B268")
    if test_param_dict is None:
        test_param_dict = {"t1": [1000], }
    vupper, vlower = vibrational_levels_for_3_ro_vib_series()
    grid_params["t1"] = 1000
    grid_params["a"] = -8
    all_params["vupper"] = vupper
    all_params["vlower"] = vlower
    all_params["Rmax_in"] = 100
    all_params["dF"] = ""
    all_params["dust"] = True
    all_params["convolve"] = False
    all_params["maxmin"] = [100, -100]
    all_params["print_Rs"] = False
    all_params["save"] = "prediction_p4"
    all_params["save_reduced_flux"] = False

    return grid_params, all_params, test_param_dict


def fid_model_B268_no_dust(test_param_dict=None):
    grid_params, all_params = cfg.get_default_params("B268")
    if test_param_dict is None:
        test_param_dict = {"dust": [False, ], }
    vupper, vlower = vibrational_levels_for_3_ro_vib_series()
    grid_params["t1"] = 1000
    grid_params["a"] = -8
    all_params["vupper"] = vupper
    all_params["vlower"] = vlower
    all_params["Rmax_in"] = 100
    all_params["dF"] = "no_dust"
    all_params["dust"] = False

    return grid_params, all_params, test_param_dict


def fid_model_B275_no_dust(test_param_dict=None):
    """
    Grid most commonly used in preparation for P4 to test stuff with adaptable test parameter grid.
    Note that the arrays for cumulative flux calculation are being saved for this grid, but not the final models.

    :return:
    """

    grid_params, all_params = cfg.get_default_params("B275")
    if test_param_dict is None:
        test_param_dict = {"dust": [False]}
    vupper, vlower = vibrational_levels_for_3_ro_vib_series()
    all_params["vupper"] = vupper
    all_params["vlower"] = vlower
    all_params["Rmax_in"] = 100
    all_params["dF"] = "_no_dust"
    all_params["dust"] = False

    return grid_params, all_params, test_param_dict
