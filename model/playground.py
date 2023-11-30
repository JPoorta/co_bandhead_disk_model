import model.example as ex
import config as cfg

grid_params = {"Ri": 4.8,
               "Ti": 4000, # [500,1000,2500,4000]
               "p": -0.75,  # [-0.5, -0.75, -2]
               "Ni": 3.e25,
               "q": -1.5,
               }

all_params = {"inc_deg": [40],  # 10,20,30,40,50,60,70,80
              "stars": ['B275'],  # , 'B243', 'B275', 'B163', 'B331']
              "dv0": [1],
              "vupper": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
              "vlower": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
              "nJ": 150,
              "dust": True,
              "sed_best_fit": True,
              # From here on only optional parameters.
              "num_CO": 100,
              "num_dust": 200,
              "Rmin_in": None,
              "Rmax_in": None,
              "print_Rs": True,
              "convolve": True,
              "save": None,
              "maxmin": (1.3, 1.02),
              "lisa_it": None,
              "saved_list": None
              }

cfg.max_wvl = 5

ex.all_params = all_params
ex.grid_params = grid_params


test_param = "Rmax_in"  # "Ti"
test_param_array = [12.4, 18.5, 92.3, 923, 1846] #[500, 1000, 2500, 4000]

ex.run(test_param, test_param_array)