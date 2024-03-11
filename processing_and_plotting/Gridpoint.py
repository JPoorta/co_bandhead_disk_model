import numpy as np
import model.config as cfg

from model.flat_disk_log_grid import create_t_gas_array, create_t_dust_array, create_radial_array


class Gridpoint:
    """
    Object with methods to apply to gridpoints in the parameter space of a model grid.
    """

    def __init__(self, sub_folder="",
                 star=None,
                 results_folder=cfg.results_folder,
                 dv0=None,
                 ri=None,
                 q=None,
                 ni=None,
                 p=None,
                 ti=None,
                 t1=None,
                 a=None,
                 test_param=None,
                 test_value=None,
                 all_params=None,
                 inc_deg=None,
                 ):

        # define default attributes
        self.sub_folder = sub_folder
        self.star = star
        self.results_folder = results_folder
        self.dv0 = dv0
        self.ri_au = ri
        self.ri_cm = ri * cfg.AU
        self.q = q
        self.ni = ni
        self.p = p
        self.ti = ti
        self.t1 = t1
        self.a = a
        self.all_params = all_params
        self.test_param = test_param
        self.test_value = test_value
        self.inc_deg = inc_deg

        if self.all_params is not None:
            self.dv0 = self.all_params["dv0"]
            self.inc_deg = self.all_params["inc_deg"]
            self.stars = self.all_params["stars"]
            if len(self.stars) > 1:
                print("WARNING: an array of stars was passed to Gridpoint. By default `self.stars[0]` will be used.")
            if self.star is None:
                self.star = self.stars[0]
            elif self.stars[0] != self.star:
                print("ERROR: The explicit parameter `star` and the one passed through `all_params[\"stars\"]` are "
                      "not the same. Please check input.")
                return

        if self.star is None:
            print("WARNING: no star is defined for this gridpoint.")
            return

    def filename_co(self):
        """
        Needs ri (in AU),ti,p,ni and q.
        :return: The concatenated strings of the gridpoint defining parameters.
        """
        try:
            return cfg.filename_co_grid_point(self.ti, self.p, self.ni, self.q, self.ri_au, t1=self.t1, a=self.a,
                                              dv=self.dv0)

        except TypeError:
            print("WARNING: Variables ri (in AU),ti,p,ni,q of the gridpoint object are not (all) defined.")

            return

        # define combined attributes

    def path_co(self):
        """
        Path to folder where saved model arrays are stored.

        :return: (str) path to folder
        """
        return str(self.results_folder / self.star / self.sub_folder) + "/"

    def obtain_dust_fit_params(self):
        """
        Obtain the best fit parameters of the dust model.

        :return:
        [0] temperature at ri,
        [1] temp power law exponent
        [2]
        """
        modelname, ti_d, p_d, ni_d, q_d, i_d, ri_d_au, r_turn, beta, r_out_au = cfg.best_dust_fit_ALMA[self.star]
        # TODO: make this return a dictionary.
        return ti_d, p_d, ni_d, q_d, i_d, ri_d_au, r_turn, beta, r_out_au

    def obtain_radial_model_array(self):
        """
        With the all input parameters of a model, obtain the radial array with its gas-only mask and the
        temperature arrays of dust and gas.

        :return: the radial array of the gas disk in cm and the mask marking the dust free zone.
        """

        # Get the necessary parameters from the input.
        ti_d, p_d, ni_d, q_d, i_d, ri_d_au, r_turn, beta, r_out_au = self.obtain_dust_fit_params()
        rmax_in, rmin_in, num_co = (self.all_params["Rmax_in"], self.all_params["Rmin_in"], self.all_params["num_CO"])

        # Refer to function in the model to get the output array.
        rmax, rmin, ri, ri_d, r_co, co_only = \
            create_radial_array(self.star, self.ri_au, rmax_in, rmin_in, ri_d_au, r_out_au, self.ti, num_co)

        return r_co, co_only

    def obtain_model_t_gas(self):
        """

        :return: The gas temperature array of the model in K.
        """

        # Refer to functions in the model to get the output array.
        r_co, co_only = self.obtain_radial_model_array()
        p_d = self.obtain_dust_fit_params()[1]
        return create_t_gas_array(r_co, co_only, self.ti, self.ri_cm, self.t1, self.a, self.p, p_d)

    def obtain_model_t_dust(self):
        """

        :return: The dust temperature array of the model in K.
        """
        ti_d, p_d, ni_d, q_d, i_d, ri_d_au, r_turn, beta, r_out_au = self.obtain_dust_fit_params()
        r_co, co_only = self.obtain_radial_model_array()

        return create_t_dust_array(r_co, co_only, ti_d, ri_d_au*cfg.AU, p_d)
