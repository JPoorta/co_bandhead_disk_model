import numpy as np
import os
import model.config as cfg

from model.flat_disk_log_grid import create_t_gas_array, create_t_dust_array, create_radial_array, instrumental_profile


class Gridpoint:
    """
    Object with methods to apply to grid points in the parameter space of a model grid.
    """

    def __init__(self,
                 star=None,
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
                 instrument_profile=None
                 ):

        # define default attributes
        self.star = star
        self.dv0 = dv0
        self.ri = ri
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
        self.ip_dx = instrument_profile

        if self.all_params is not None:
            self.dv0 = self.all_params["dv0"]
            self.inc_deg = self.all_params["inc_deg"]
            self.stars = self.all_params["stars"]
            if self.all_params["dF"] is not None:
                self.sub_folder_df = "dF" + self.all_params["dF"]
            if self.all_params["save"] is not None:
                self.grid_folder = self.all_params["save"]
            # Define the star for the grid from all_params if not already explicitly defined.
            if self.star is None:
                self.star = self.stars[0]
                if len(self.stars) > 1:
                    print("WARNING: an array of stars was passed to Gridpoint. "
                          "By default `stars[0]`("+self.star+") is used.")
            # When all_params is defined the radial array and its mask is added to attributes.
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
            return cfg.filename_co_grid_point(self.ti, self.p, self.ni, self.q, self.ri, t1=self.t1, a=self.a,
                                              dv=self.dv0[0])

        except TypeError:
            print("WARNING: Variables ri (in AU),ti,p,ni,q, t1, a, dv0 of the gridpoint object are not (all) "
                  "correctly defined.")

            return

    def path_co_df(self):
        """
        Path to folder where saved intensity and wavelength arrays for the calculation of cumulative flux are stored.

        :return: (str) path to folder
        """
        return str(cfg.results_folder / self.star / self.sub_folder_df) + "/"

    def path_grid_folder(self):
        """
        Path to folder where the grid of models is stored.

        :return: (str) path to folder
        """
        return str(cfg.results_folder / self.star / self.grid_folder) + "/"

    def full_path_model_file(self):
        """
        Full path including the filename to model file, including file ending.

        :return:
        """
        return self.path_grid_folder() + self.filename_co() + ".npy"

    def full_path_df(self):
        """

        :return: Full path including the filename to intensity and wvl arrays for cumulative flux,
        excluding file ending.
        """
        return self.path_co_df() + self.filename_co()

    def delete_df_file(self, df_file):
        """
        If it is indeed a file in the dF folder, delete the given file.

        :param df_file:
        :return:
        """
        if self.full_path_df() in df_file:
            os.remove(df_file)
            print(df_file+" has been deleted.")
        else:
            print("WARNING: "+df_file+" has NOT been deleted, because it is not in the cumulative flux folder.")
        return

    def saved_wvl_array_df(self):
        """
        Wavelength array for cumulative flux calculation in the dF folder.

        :return: the saved wavelength array in the dF folder.
        """
        return np.load(self.full_path_df() + "_wvl.npy")

    def model_wvl_array(self):
        """
        Final model wavelength array as saved in the grid folder.

        :return: the saved wavelength array in grid_folder.
        """
        return np.load(self.path_grid_folder() + "wvl_re.npy")

    def saved_intensity_conv_co(self):
        """
        Reads the saved convolved intensity from the part of the disk where only CO gas exists.
        Only works if for this model this was saved in 'subfolder'.

        :return: the saved, disk convolved intensity of the CO gas.
        """
        return np.load(self.full_path_df() + "_CO.npy")

    # TODO change the output of these functions to paths to use them for deleting files.

    def saved_intensity_conv_mix(self):
        """
        Reads the saved total convolved intensity from the part of the disk where both CO and dust exist.
        Only works if for this model this was saved in 'subfolder'.

        :return: the saved, disk convolved intensity of dust and gas mix.
        """
        return np.load(self.full_path_df() + "_mix.npy")

    def saved_intensity_conv_dust(self):
        """
        Reads the saved convolved intensity of the dust (from the part of the disk where both CO and dust exist).
        Only works if for this model this was saved in 'subfolder'.

        :return: the saved, disk convolved intensity of dust and gas mix.
        """
        return np.load(self.full_path_df() + "_dust.npy")

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
            create_radial_array(self.star, self.ri, rmax_in, rmin_in, ri_d_au, r_out_au, self.ti, num_co)

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

        return create_t_dust_array(r_co, co_only, ti_d, ri_d_au * cfg.AU, p_d)

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
            wvl = self.saved_wvl_array_df()
        if wvl_indices is None:
            wvl_indices = np.arange(len(wvl))
        return np.trapz(flux[:, wvl_indices], x=wvl[wvl_indices])

    def calc_cumulative_flux(self, wvl=None, intensity=None, r=None):
        """
        Calculate the normalized cumulative flux as a function of disk radius.

        :param wvl: wavelength array in micron
        :param intensity: (3D array) intensities, shape [len(dv0), len(r), len(wvl)]
        :param r:
        :return: 3D array with dimensions [len(dv0), len(r), len(wvl[])] that is,
        the cumulative flux as a function of r.
        """
        if intensity is None:
            intensity = self.get_total_co_intensity()
        if wvl is None:
            wvl = self.saved_wvl_array_df()
        if r is None:
            r = self.r_co

        second_ot = np.array(np.where(wvl < 2.25))[0]  # 1.85
        first_ot = np.where(np.logical_and(wvl > 2.25, wvl < 4.2))[0]  # 3.25
        fundamental = np.where(wvl > 4.2)[0]

        flux = self.calc_flux()

        dF = np.zeros((intensity.shape[0], intensity.shape[1], 3))
        for n, el in enumerate([second_ot, first_ot, fundamental]):
            total = self.wvl_integrated_flux(flux=flux, wvl=wvl, wvl_indices=el)
            for k in range(len(r)):
                flux_till_k = self.calc_flux(intensity=intensity[:, :k, el], r=r[:k])
                dF[:, k, n] = self.wvl_integrated_flux(flux=flux_till_k, wvl=wvl[el]) / total

        return dF

    def return_cum_flux(self):

        cum_flux_extension = "_dF_CO_lines"

        # Check if calculations have been made.
        try:
            dF_disk, r_disk_AU = np.load(self.full_path_df() + cum_flux_extension + ".npy")
        # If not do the total calculation.
        except FileNotFoundError:
            dF_disk = self.calc_cumulative_flux()
            r_disk_AU = self.r_co / cfg.AU
            np.save(self.full_path_df() + cum_flux_extension, [dF_disk, r_disk_AU])

        return dF_disk, r_disk_AU

    def read_output_per_inc(self, convolve=False, ip_dx=None):
        """
        Read the co_bandhead numpy file which stores the normalized bandheads per inclination.

        :param convolve: (Bool) if True, convolve the models with instrumental profile.
        :param ip_dx: (tuple of two arrays or None) if provided should be the instrument profile to convolve the fluxes
         with. Note that even if convolve=True ip_dx need not be provided, it will default to local method.
        :return: a dictionary with keys inclination and values the model spectra.
        If there is only one inclination just return the flux array.
        """
        results = np.load(self.full_path_model_file())

        model_per_inc_dict = {}
        if convolve and ip_dx is None:
            ip_dx = self.return_instrument_profile()

        for j, i in enumerate(self.inc_deg):
            if convolve:
                flux = self.convolve_flux(flux=results[j, :], ip_dx=ip_dx)
            else:
                flux = results[j, :]

            model_per_inc_dict[i] = flux

        if len(self.inc_deg) == 1:
            return model_per_inc_dict[self.inc_deg[0]]
        else:
            return model_per_inc_dict

    def return_instrument_profile(self, res=None):
        """
        Return the instrument profile. If it was not passed as an attribute, it is calculated with class method that
        uses the JWST NIRSPEC resolution in config and the saved wavelength array.

        :param res: optional instrumental resolution.
        :return: two arrays used for convolutions: ip (to convolve with) and dx (to multiply by).
        """
        if self.ip_dx is not None:
            return self.ip_dx  # first try if the instrument profile is explicitly passed at initialisation of gp.
        # If not calculate it.
        if res is None:
            res = cfg.jwst_nirspec_res
        return instrumental_profile(self.model_wvl_array(), res)

    def convolve_flux(self, flux, ip_dx=None):
        """
        Convolve flux with instrumental profile.

        :param flux: model flux to convolve, should have same dimension as provided instrument profile or, if ip_dx is
         not provided, as the saved model wavelength array.
        :param ip_dx: optional instrument profile.
        :return:
        """
        if ip_dx is None:  # If not passed to function define the profile from in-house method.
            ip, dx = self.return_instrument_profile()
        else:
            ip, dx = ip_dx
        return np.convolve(flux, ip, mode="same") * dx

    def return_value(self, param):
        """
        Return the value of a given parameter. For parameter that have array value (see below), return the first item.

        :param param: (str) parameter name.
        :return:
        """
        if param in ["inc_deg", "dv0", "stars"]:
            value = getattr(self, param)[0]
        else:
            value = getattr(self, param)

        return value
