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
        self.ri = ri
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

    def filename_co(self):
        """
        Needs ri (in AU),ti,p,ni and q.
        :return: The concatenated strings of the gridpoint defining parameters.
        """
        try:
            return cfg.filename_co_grid_point(self.ti, self.p, self.ni, self.q, self.ri, t1=self.t1, a=self.a,
                                              dv=self.dv0)

        except TypeError:
            print("WARNING: Variables ri (in AU),ti,p,ni,q of the gridpoint object are not (all) defined.")

            return

        # define combined attributes

    def path_co(self):
        return str(self.results_folder / self.star / self.sub_folder) + "/"

    def obtain_model_arrays_from_params(self):
        """
        With the all input parameters of a model, obtain the radial array with its gas-only mask and the
        temperature arrays of dust and gas.

        :return:
        """

        # Get the necessary parameters from the input.
        modelname, ti_d, p_d, ni_d, q_d, i_d, ri_d_au, r_turn, beta, r_out_au = cfg.best_dust_fit_ALMA[self.star]
        rmax_in, rmin_in, num_co = (self.all_params["Rmax_in"], self.all_params["Rmin_in"], self.all_params["num_CO"])

        # Refer to functions in the model to get the output array.
        rmax, rmin, ri, ri_d, r_co, co_only = \
            create_radial_array(self.star, self.ri, rmax_in, rmin_in, ri_d_au, r_out_au, self.ti, num_co)
        t_gas = create_t_gas_array(r_co, co_only, self.ti, ri, self.t1, self.a, self.p, p_d)
        t_dust = create_t_dust_array(r_co, co_only, ti_d, ri_d, p_d)

        return r_co, co_only, t_gas, t_dust
