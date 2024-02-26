import matplotlib.pyplot as plt

import Gridpoint
import cumulative_flux as df
import model.config as cfg


def run():
    star = "B275"

    grid_params, all_params = cfg.get_default_params(star)
    # Adjust defaults if wanted (optional).

    # set the parameter to be tested (optional).
    test_param = "p"  # "Ti"
    test_param_array = [-2, -3 ,-5] # [-0.5, -0.75, -1, -2]

    all_params["vupper"] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    all_params["vlower"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    all_params["dust"] = True
    grid_params["p"] = -0.75
    all_params["Rmax_in"] = 100
    all_params["dF"] = True

    for value in test_param_array:
        grid_params[test_param] = value
        gp = Gridpoint.Gridpoint(**grid_params, star=star, sub_folder = "dF_almadust_p2_-0.5") # "dF"
        df.plot_cum_flux(gp)

    plt.legend()
    plt.show()

    return


if __name__ == "__main__":
    run()
