import numpy as np
import model.config as cfg

from model.flat_disk_log_grid import create_t_gas_array, create_t_dust_array, create_radial_array


class Gridpoint:
    """
    Object with methods to apply to gridpoints in the parameter space of a model grid.
    """

    def __init__(self, sub_folder=None,
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
            if self.sub_folder is None and self.all_params["dF"] is not None:
                self.sub_folder = "dF"+self.all_params["dF"]
            if len(self.stars) > 1:
                print("WARNING: an array of stars was passed to Gridpoint. By default `self.stars[0]` will be used.")
            if self.star is None:
                self.star = self.stars[0]
            elif self.stars[0] != self.star:
                print("ERROR: The explicit parameter `star` and the one passed through `all_params[\"stars\"]` are "
                      "not the same. Please check input.")
                return
            r_co, co_only = self.obtain_radial_model_array()
            self.r_co = r_co
            self.co_only = co_only

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
                                              dv=self.dv0[0])

        except TypeError:
            print("WARNING: Variables ri (in AU),ti,p,ni,q, t1, a, dv0 of the gridpoint object are not (all) "
                  "correctly defined.")

            return

        # define combined attributes

    def path_co(self):
        """
        Path to folder where saved model arrays are stored.

        :return: (str) path to folder
        """
        return str(self.results_folder / self.star / self.sub_folder) + "/"

    def full_path(self):
        """

        :return: Full path including the filename, excluding file ending.
        """
        return self.path_co() + self.filename_co()

    def saved_wvl_array(self):
        """
        Only works if a wavelength array was indeed saved for this model in the specified 'subfolder'.

        :return: saved wavelength array.
        """
        return np.load(self.full_path() + "_wvl.npy")

    def saved_intensity_conv_co(self):
        """
        Reads the saved convolved intensity from the part of the disk where only CO gas exists.
        Only works if for this model this was saved in 'subfolder'.

        :return: the saved, disk convolved intensity of the CO gas.
        """
        return np.load(self.full_path() + "_CO.npy")

    def saved_intensity_conv_mix(self):
        """
        Reads the saved total convolved intensity from the part of the disk where both CO and dust exist.
        Only works if for this model this was saved in 'subfolder'.

        :return: the saved, disk convolved intensity of dust and gas mix.
        """
        return np.load(self.full_path() + "_mix.npy")

    def saved_intensity_conv_dust(self):
        """
        Reads the saved convolved intensity of the dust (from the part of the disk where both CO and dust exist).
        Only works if for this model this was saved in 'subfolder'.

        :return: the saved, disk convolved intensity of dust and gas mix.
        """
        return np.load(self.full_path() + "_dust.npy")

    def get_total_co_intensity(self):
        """
        Concatenate the co intensity with the mix-dust intensity to obtain the full CO intensity over the entire disk.

        :return: the total CO gas intensity (radial dimension should cover entire disk).
        """
        intensity_conv_co = self.saved_intensity_conv_co()
        intensity_conv_mix = self.saved_intensity_conv_mix()
        intensity_conv_dust = self.saved_intensity_conv_dust()

        return np.concatenate((intensity_conv_co, intensity_conv_mix - intensity_conv_dust), axis=1)

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

    def calc_flux(self, intensity=None, r=None):
        """
        Integrate 3D intensity [dv0, R, wvl] over R to obtain flux.
        *NOT corrected for inclination and distance*
        If intensity in cgs, then flux in erg/s/micron, so integrated over the disk, but not corrected for distance.
        If intensity and radial array provided, units should be in agreement.
        By default, self.r_co and total CO intensity over the disk (so without dust)
         will be used, based on saved arrays.

        :return: (2D array) flux with dimensions [len(dv0), len(wvl)]
        """
        if intensity is None:
            intensity = self.get_total_co_intensity()
        if r is None:
            r = self.r_co
        flux = np.trapz(intensity * r[None, :, None], x=r, axis=1)

        return flux

    def calc_inclined_d_corrected_flux(self, flux=None, i=None):
        """
         Inclination corrected flux.

        :param flux:
        :param i: in degrees.
        :return:
        """
        if flux is None:
            flux = self.calc_flux()
        if i is None:
            i = self.inc_deg[0]
        inc_rad = i * np.pi / 180  # inclination from degrees to radians
        return flux * np.cos(inc_rad) / cfg.d ** 2

    def wvl_integrated_flux(self, flux=None, wvl=None, wvl_indices=None):
        """
        Integrate the flux over (a part of) the wavelength array.
        Units of flux and wavelength should be in agreement; e.g. flux in erg/s/cm^2/micron and wvl in micron.

        :param flux:
        :param wvl:
        :param wvl_indices: optional index mask for wavelength.
        If not given, integration will be done over entire array.
        :return:
        """
        if flux is None:
            flux = self.calc_flux()
        if wvl is None:
            wvl = self.saved_wvl_array()
        if wvl_indices is None:
            wvl_indices = np.arange(len(wvl))
        return np.trapz(flux[:, wvl_indices], x=wvl[wvl_indices])

