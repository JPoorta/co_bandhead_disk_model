"""
The module containing all the actual model grid calculations. More information can be found in the
docstrings and in-code comments.
"""

from time import time

import numpy as np
from scipy.integrate import fixed_quad
from scipy.signal import fftconvolve

import model.config as cfg
import model.sed_calculations as seds


def create_freq_array_profile_dict(freq_trans, dv, dv0):
    """
    Create the wavelength array from a regularly (dv) spaced velocity array, and calculate the gaussian line profile
    for each transition in frequency space with width(s) dv0.

    :param freq_trans: Frequency of the transitions in cm**-1.
    :param dv: resolution of velocity/wavelength and frequency arrays in cm/s.
    :param dv0: list of doppler widths of the lines in cm/s.
    :return:
        - wavelength array in micron, sorted from low to high and with equal spacing in velocity space.
        - Line profile dictionary of `len(freq_trans)` with
            - keys: transition in cm^-1;
            - values: list of two 2D ``numpy`` arrays: gaussian profiles, and indices on the frequency (and wavelength)
              array, both of shape (len(dv0), len(indices)).
    """

    # --------------------------------------------------------------
    # Create a regularly spaced velocity array (in cm/s) (for convolution with velocity profiles),
    # based on maximum and minimum wavelengths. Convert velocities back to wavelengths (micron).
    # --------------------------------------------------------------

    a = 1 - dv / cfg.c

    min_wvl = np.min(1.e4 / freq_trans) - 0.01
    max_wvl = np.min(((np.max(1.e4 / freq_trans) + 0.01), cfg.max_wvl))
    max_vel = np.log(min_wvl / max_wvl) / np.log(a) * dv
    vel = np.arange(0, max_vel, dv)
    wvl = min_wvl * np.power(np.ones(len(vel)) / a, np.arange(len(vel)))

    # --------------------------------------------------------------
    # Create a gaussian line profile (in frequency space) for each transition and every line width provided in dv0;
    # store in a dictionary.
    # --------------------------------------------------------------
    freq = 1.e4 * cfg.c / wvl
    freq_trans_hz = freq_trans * cfg.c

    profile_dict = {}
    for i, trans in enumerate(freq_trans_hz):
        vel_array = cfg.c * (1 - freq / trans)
        indices = np.array([np.where(np.logical_and(vel_array > - 5 * x, vel_array < 5 * x))[0] for x in dv0])
        # Store also the indices on the frequency (wavelength) array.
        gaussians = np.array([cfg.gaussian(freq[x], trans, trans * (y / cfg.c)) for x, y in zip(indices, dv0)])
        profile_dict[freq_trans[i]] = [gaussians, indices]

    return wvl, profile_dict


def co_bandhead(T_gas, NCO, wave, A_einstein, jlower, jupper, freq_trans, Elow, prof_dict, dust, dust_params=None):
    """
    Calculates the source function and opacities ifo wavelength for the CO-bandheads and (iff dust=True) for the
    dust continuum.

    :param T_gas: Temperature of the CO gas in function of R.
    :type T_gas: 1D array with `len(R)`
    :param NCO: Surface number density (in cm^-2) of the CO gas in function of R.
    :type NCO: 1D array with `len(R)`
    :param wave: wavelength array for which the output is wanted.
    :type wave: 1D array
    :param A_einstein: Einstein A coefficients of all the transitions.
    :type A_einstein: 1D array with `len(freq_trans)`
    :param jlower: lower rotational quantum numbers of all the transitions.
    :type jlower: 1D array with `len(freq_trans)`
    :param jupper: upper rotational quantum numbers of all the transitions.
    :type jupper: 1D array with `len(freq_trans)`
    :param freq_trans: transition frequencies in cm^-1.
    :type freq_trans: 1D array
    :param Elow: energies of the lower levels of the transitions in Kelvin.
    :type Elow: 1D array with `len(freq_trans)`
    :param prof_dict:

        - keys: transition in cm^-1
        - values: gaussian profile, indices on the wavelength array; each an array of the length of dv0.
    :type prof_dict: line profile `dictionary`
    :param dust: if True dust is included and dust_params has to be provided. If False no continuum is returned.
    :type dust: Bool
    :param dust_params: H2 surface density and dust temperatures ifo R.
    :type dust_param: list of two 1D arrays of `len(R)`: [NH, T_dust]
    :return: iff dust == `False`: Source function and opacity for CO; resp. 2D array of shape (len(R),len(wave)) and 3D
        array of shape `(len(dv0), len(R), len(wave))`.

        iff dust = `True`: Total source function and opacity for CO+dust; two 3D arrays of shape (len(dv0),len(R),
        len(wave)), and opacity and source function for dust; two 2D arrays of shape (len(R),len(wave)).
        Total four arrays.
    """

    # -----------------------------------------------------------------------------
    # Reading of the partition sums and calculation fractional level populations in LTE
    # -----------------------------------------------------------------------------
    Q = np.interp(T_gas, cfg.Q_T['T'], cfg.Q_T['Q'])
    x = (2 * jlower[None, :] + 1) / Q[:, None] * np.exp(-Elow[None, :] / T_gas[:, None])

    # c**2 is not lacking in this equation! It is incorporated through the units of the transition frequencies.
    lines = (A_einstein[None, :] / (8.0 * np.pi * freq_trans[None, :] ** 2)) * \
            (2. * jupper[None, :] + 1.) / (2. * jlower[None, :] + 1.) * \
            (1.0 - np.exp(-cfg.h * freq_trans[None, :] * cfg.c / cfg.kB / T_gas[:, None])) * x

    alpha = np.zeros((len(prof_dict.get(freq_trans[0])[0]), len(T_gas),
                      len(wave)))  # tau_CO = alpha * NCO (alpha is CO optical depth per particle)

    for i in range(lines.shape[1]):
        gauss, indices = prof_dict.get(freq_trans[i])
        for j in range(len(gauss)):
            alpha[j, ..., indices[j]] += lines[..., i][None, :] * gauss[j][:, None]

    tau_CO = alpha * NCO[None, :, None]  # optical depth of CO/gas

    # Source function gas in erg/s/cm^2/sr/micron
    BB_CO = cfg.planck_wvl(wave[None, :] / 1.e4, T_gas[:, None])

    if not dust:

        return BB_CO, tau_CO

    else:
        NH, T_dust = dust_params

        tau_cont_wsil = ((np.pi * cfg.radg ** 2) * cfg.Qabs[None, :]) * NH[:, None] \
                        * 2 * cfg.mass_proton / cfg.mass_grain / cfg.gas_to_solid
        tau_cont = np.array([np.interp(wave, cfg.wsil[::-1], tau_cont_wsil[nh, :][::-1]) for nh in range(len(NH))])
        # Source function dust erg/s/cm^2/sr/micron
        BB_dust = cfg.planck_wvl(wave[None, :] / 1.e4, T_dust[:, None])
        # Total optical depth
        tau_tot = tau_CO + tau_cont[None, :, :]
        # Total source function erg/s/cm^2/sr/micron
        S_tot = (tau_CO * BB_CO[None, :, :] + tau_cont[None, :, :] * BB_dust[None, :, :]) / tau_tot

        return S_tot, tau_tot, tau_cont, BB_dust


def calculate_flux(S, tau, i, R, wvl, convolve=False, int_theta=None, dv=None, dF=False):
    """
    Calculate the slab model flux (in erg/s/cm^2/micron) of a thin disk from source function and opacity per ring (R).
    Before integrating over R (optionally) convolve with velocity profiles using int_theta.

    :param S: source function (erg/s/cm^2/sr/micron) in function of wvl and R (2D array of shape (len(R),len(wvl)),
        sometimes provided for multiple dv0 (if S_tot) in which case 3D array of shape (len(dv0),len(R),len(wvl).
    :param tau: opacity in function of wvl and R (3D array of shape (len(dv0),len(R),len(wvl)).
    :param i: inclination in radians.
    :param R: radial distances (from the stellar surface) of the rings (in cm) (1D array).
    :param wvl: wavelength array in micron, equally spaced in velocity (see create_freq_array_profile_dict) Used to
            calculate the flux per ring (integration over wvl - dF).
    :param convolve: if True intensities will be convolved with a velocity profile provided through int_theta.
    :param int_theta: (2D array of shape (len(R), len(vel_Kep)) where vel_Kep represents an array of Keplerian
            velocities with spacing dv. This parameter is a velocity profile (by construction integrated over a ring, i.e.,
            2 pi) for each radius (units (cm/s)^-1).
    :param dv: velocity resolution in cm/s of vel_interp and of vel_Kep (see int_theta).
    :param dF: (boolean) if True calculate flux per ring.
    :return: flux in erg/s/cm^2/micron and dF/dR: wavelength integrated flux per ring in erg/s/cm^2/cm.
            (Two 1D arrays with len(wvl).)
    """

    # Intensity (optionally) convolved with velocity profile (int_theta).
    if S.ndim == 2:
        intensity = S[None, :, :] * (1. - np.exp(-tau / np.cos(i)))
    elif S.ndim == 3:
        intensity = S * (1. - np.exp(-tau / np.cos(i)))

    if convolve:
        intensity_conv = np.zeros(intensity.shape)
        for k in range(intensity.shape[0]):
            intensity_conv[k, ...] = np.array([fftconvolve(intensity[k, j, :], int_theta[j, :], mode="same") * dv
                                               for j in range(len(R))])
    else:
        intensity_conv = intensity * 2 * np.pi

    # Flux and differential of the flux.
    flux = np.array([np.trapz(intensity_conv[j, ...].T * R, x=R, axis=1) * np.cos(i) / cfg.d ** 2
                     for j in range(intensity.shape[0])])
    if dF:
        dF = np.array([np.trapz(intensity_conv[j, ...].T * R * np.cos(i) / cfg.d ** 2, x=wvl, axis=0)
                       for j in range(intensity.shape[0])])

    return flux, dF


def instrumental_profile(wvl, res, vupper=None, vlower=None, center_wvl=None):
    """
    Calculates instrumental profile for convolution to compare with observations. The units of the instrumental
    profile are micron^-1; the convolution product should be multiplied by dx to approximate continuous convolution.

    :param wvl: wavelength array that goes with the flux, in micron.
    :param res: spectral resolution R of the observation.
    :param vupper: array of upper vibrational quantum numbers. (optional, should be provided if no center_wvl is given)
    :param vlower: array of lower vibrational quantum numbers. (optional, should be provided if no center_wvl is given)
    :param center_wvl: reference wavelength for resolution and pivot point for convolution (0-point velocity).
            (optional, but either this or vupper and vlower should be provided).
    :return: a gaussian profile (in micron^-1) and the wavelength step dx in micron.
    """

    # Wavelength of onset of the bandhead in micron.
    if center_wvl is not None:
        center = center_wvl
    else:
        center = np.min([cfg.onset_wvl_dict.get((vu, vl)) for vu, vl in zip(vupper, vlower)])

    sigma = center / res  # sigma in micron
    # Wavelengths in micron within 5 sigma.
    wvl_ip = wvl[np.where(np.abs(wvl - center) < 5 * sigma)]
    ip = cfg.gaussian(wvl_ip, center, sigma)  # gaussian centered around w0 (in micron^-1)
    dx = np.mean(wvl_ip[1:] - wvl_ip[0:-1])

    return ip, dx


def run_grid_log_r(grid, inc_deg, stars, dv0, vupper, vlower, nJ, dust, sed_best_fit, num_CO=100,
                   num_dust=200, Rmin_in=None, Rmax_in=None, print_Rs=False, convolve=False, save=None,
                   maxmin=(1.3, 1.02), lisa_it=None, saved_list=None):
    """
    Calculates a grid of CO bandhead profiles and optionally save the normalized profiles and the wavelength array.
    For a test run use scalar values in the grid and for dv0, and set convolve = True.

    :param grid: grid with 5 variables, created as follows by np.meshgrid:

        .. code-block:: python

            tiv, pv, niv, qv, riv = np.meshgrid(Ti, p, Ni, q, Ri,sparse=False)
            grid = [tiv, pv, niv, qv, riv]

        With Ti, p, Ni, q, Ri arrays or scalars:

         - Ti: initial CO gas temperature  at Ri (in K).
         - p: power law exponent for the gas temperature (p<0).
         - Ni: initial gas (H2) surface density at Ri (in cm^-2).
         - q: power law exponent for the gas surface density (q<0).
         - Ri: initial radius for the powerlaws for the CO gas disk (in stellar radii R*).
    :param inc_deg: array of inclination(s) in degrees. This can only contain values from [10,20,30,40,50,60,70,80]
    :param stars: array of string(s) with the object names.
    :param dv0: (scalar or array) doppler width of the lines in km/s. If array, for each element the bandheads
        are calculated and (if relevant) saved in separate files with corresponding filenames.
    :param vupper: list of upper level(s) of the vibrational transition(s).
    :param vlower: list of lower level(s) of the vibrational transition(s).
    :param nJ: number of rotational transitions to be taken into account.
    :param dust: boolean. If True dust is included. If False the continuum is assumed to be stellar only.
    :param sed_best_fit: boolean.
        - If True the best fit SED to the photometry is used for the continuum and all its dust disk parameters are adopted
        (including inclination).
        - If False the best fit SED to the photometry that has the same inclination as the CO disk is used for the continuum.
        - If dust = False this parameter is ignored.
    :param num_CO: (optional) integer specifying amount of radial points for the CO gas disk. (default = 50)
    :param num_dust: (optional) integer specifying amount of radial points for the dust disk. (default = 200)
    :param Rmin_in: (optional) initial radius for the CO gas disk (in stellar radii R*). If not provided defaults to Ri.
    :param Rmax_in: (optional) outer radius for the CO gas disk (in stellar radii R*). If not provided defaults
            to the point where the gas temperature drops below 1000 K (is thus dependent on Ri, Ti, p and hence different
            per grid point).
    :param print_Rs: (optional) boolean. If True information on the radial arrays is printed. (default = False)
    :param convolve: (optional) boolean. If True the fluxes are convolved with the instrumental profile and
            returned in the first loop over the grid and inclination. The grid is then not iterated over and no fluxes are
            saved regardless of the value of save. (default = False)
    :param save: (optional) string specifying the name of the folder where the normalized bandheads and the wavelength
            array are to be saved. If not provided, nothing is saved. (default = None)
    :param maxmin: (optional) tuple (max, min). The maximum and minimum values maximum normalized flux points can take.
            If the maximum of the normalized flux is higher than max or lower than min the model is not saved, even if
            save is true.
    :param lisa_it: (int (from 0 to 15) or None) refers to the thread number when the code is run on 16 parallel cores
            in LISA. If used, the grid is divided into 16 equal parts, and the part with this thread number is run.
            If this parameter is set to None, the full grid for all objects will be calculated serially.
    :param saved_list: (optional) string, specifying the filename extension (in folder aux_files, and beginning with a
            star name) which contains a list of filenames of the models that need to be calculated (all other models
            will be skipped!). To be used if large parts of the parameter space in a grid are models that don't get
            saved anyway as for e.g. low (<0.5) p values.
    :return: the wavelength array and a 2D array containing the normalized fluxes for each inclination and
            the last gridpoint in grid.

            If convolve is `True`: the wavelength array, total extincted flux, normalized flux, convolved total flux
            and convolved normalized flux are returned for the first gridpoint and first inclination.
    """

    # Time the duration.
    start = time()

    # --------------------------------------------------------------
    # Select relevant levels and sort according to frequency (from high to low)
    # --------------------------------------------------------------

    max_freq = 1.e4 / cfg.max_wvl  # micron to cm^-1
    mask = np.zeros(len(cfg.freq_tr), dtype=bool)

    for vu, vl in zip(vupper, vlower):
        mask += np.logical_and.reduce(
            (cfg.vhigh == vu, cfg.vlow == vl, cfg.jh <= nJ, cfg.jl_all <= nJ, cfg.freq_tr >= max_freq))

    ind = np.argsort((cfg.freq_tr[mask]))
    vl, jlower, vh, jupper, A_einstein, Elow, freq_trans = \
        [(a[mask][ind]) for a in [cfg.vlow, cfg.jl_all, cfg.vhigh, cfg.jh, cfg.A, cfg.El_all, cfg.freq_tr]]

    # ---------------------------------------------------------
    # Unit conversions
    # ---------------------------------------------------------
    # velocity in cm/s for construction of gaussian line profiles, convert input to a list
    dv0 = np.array(dv0) * 1.e5
    if type(dv0) is np.float64:
        dv0_cm = [dv0]
    else:
        dv0_cm = dv0.tolist()

    dv = cfg.min_dv_cm  # velocity resolution in cm/s
    inc = np.array(inc_deg) * np.pi / 180  # inclination from degrees to radians

    # --------------------------------------------------------------
    # Calculation of the velocity and wavelength arrays (vel, wvl) and gaussian line profiles.
    # Reading the object wavelength array.
    # --------------------------------------------------------------

    wvl, profile_dict = create_freq_array_profile_dict(freq_trans, dv, dv0_cm)
    obj_wvl_arr = np.load(cfg.obj_wvl_array)

    # --------------------------------------------------------------
    # ITERATIONS OVER THE OBJECTS.
    # --------------------------------------------------------------

    for st in stars:

        if saved_list is not None:
            to_be_calculated = np.loadtxt(cfg.aux_files + st + saved_list, dtype=str).T[1]

        # --------------------------------------------------------------
        # Folder for saving.
        # --------------------------------------------------------------
        if save is not None:
            folder = cfg.results_folder / st / save
            np.save(folder / "wvl_re", wvl)

        # --------------------------------------------------------------
        # Get stellar parameters.
        # --------------------------------------------------------------

        Mstar, T_eff, log_g, Rstar, Res, SNR, R_v, A_v, RV = cfg.stel_parameter_dict.get(st)
        best_fit = cfg.best_dust_fit.get(st)
        Ri_d = best_fit[6] * cfg.AU

        # --------------------------------------------------------------
        # Best dust fit parameters (iff dust = True) and continuum fluxes for different cases.
        # --------------------------------------------------------------

        if dust:
            # 0 modelname, 1 Ti_d, 2 p_d, 3 Ni_d, 4 q_d, 5 i_d, 6 Ri_d_AU, gas_mass, chi_sq, chi_sq_red
            inc_dust_dict = cfg.load_inc_dust_dict(st)
            for element in inc_deg:
                fit = inc_dust_dict.get(element)[0]
                flux_cont = seds.SED_full(fit[6], fit[1], fit[2], fit[3], fit[4], fit[5],
                                          st, A_v, R_v, wvl=wvl, num_r=num_dust)[4]
                inc_dust_dict[element] = (fit, flux_cont)

            best_fit_cont = seds.SED_full(best_fit[6], best_fit[1], best_fit[2], best_fit[3], best_fit[4], best_fit[5],
                                          st, A_v, R_v, wvl=wvl, num_r=num_dust)[4]

            if sed_best_fit:
                flux_cont_tot = best_fit_cont

        elif not dust:
            # --------------------------------------------------------------
            # No dust: Only stellar continuum.
            # --------------------------------------------------------------

            flux_cont_tot = cfg.stellar_cont(st, wvl)

        # --------------------------------------------------------------
        # ITERATIONS OVER THE GRID.
        # --------------------------------------------------------------

        it = np.nditer(grid, ["ranged"])

        if lisa_it is not None:
            chunk_size, rest = np.divmod(np.nditer(grid).itersize, 16)
            if lisa_it == 15:
                it.iterrange = (lisa_it * chunk_size, (lisa_it + 1) * chunk_size + rest)
            else:
                it.iterrange = (lisa_it * chunk_size, (lisa_it + 1) * chunk_size)

        for x in it:

            # ---------------------------------------------------------
            # Extract the disk parameters.
            # ---------------------------------------------------------
            ti, p, ni, q, ri_R = (x[0], x[1], x[2], x[3], x[4])

            if ti < 0:
                ti = T_eff
            # ---------------------------------------------------------
            # Creation of the radial arrays.
            # ---------------------------------------------------------
            ri = ri_R * cfg.R_sun * Rstar

            filename = cfg.filename_co_grid_point(ti, p, ni, q, ri_R)
            if saved_list is not None:
                if not np.any([filename == listed for listed in to_be_calculated]):
                    print(filename + " skip")
                    continue

            if Rmax_in is not None:
                Rmax = Rmax_in * cfg.R_sun * Rstar
            else:
                Rmax = np.min(((cfg.min_T_gas / ti) ** (1 / p) * ri, cfg.r_max_def))

            if Rmin_in is not None:
                Rmin = Rmin_in * cfg.R_sun * Rstar
            else:
                Rmin = ri

            R_CO = np.geomspace(Rmin, Rmax, num=num_CO)
            CO_only = (R_CO < Ri_d)
            R_dust = np.geomspace(Ri_d, cfg.r_max_def, num=num_dust)
            mix = (R_dust <= R_CO[-1])

            dust_in_gas = not np.array_equal(R_CO, R_CO[CO_only])
            gas_only_exist = (len(R_CO[CO_only]) > 0)

            # --------------------------------------------------------------
            # Create velocity array for velocity profiles per ring.
            # --------------------------------------------------------------

            max_Kep = cfg.v_kep_cm_s(Mstar, Rmin)
            nv = np.int(np.around(max_Kep / dv)) + 100
            vel_Kep = np.arange(-nv, nv + 1.) * dv

            # ---------------------------------------------------------
            # Print information about radial arrays (if so wished).
            # ---------------------------------------------------------

            if print_Rs:
                to_print = '\n R_max = ' + str((R_CO[-1]) / cfg.AU) + ' AU' \
                           + '\n R = ' + str(R_CO / cfg.AU) + ' AU, ' \
                           + '\n length R_CO =' + str(len(R_CO[CO_only])) \
                           + '\n length R_CO_dust =' + str(len(R_dust[mix])) \
                           + '\n length R_dust[~mix] =' + str(len(R_dust[~mix])) \
                           + "\n length wvl = " + str(len(wvl))

                print(to_print)

            # --------------------------------------------------------------
            # Gas temperature, density, opacities and source function where there is only gas.
            # --------------------------------------------------------------
            if gas_only_exist:
                T_gas = cfg.t_ex(R_CO[CO_only], ti, ri, p)
                NCO = cfg.nco(R_CO[CO_only], ni, ri, q)
                S_CO, tau_CO = co_bandhead(T_gas=T_gas, NCO=NCO, wave=wvl,
                                           A_einstein=A_einstein, jlower=jlower, jupper=jupper,
                                           freq_trans=freq_trans, Elow=Elow,
                                           prof_dict=profile_dict, dust=False)
                R_CO_only = R_CO[CO_only]

            else:
                flux_CO = np.zeros(len(wvl))

            # --------------------------------------------------------------
            #
            # --------------------------------------------------------------
            if not dust_in_gas and dust:
                # --------------------------------------------------------------
                # Dust is included and gas and dust are separate.
                # --------------------------------------------------------------

                # Continuum flux (not extincted).
                flux_cont_tot = best_fit_cont

            if not dust and dust_in_gas:
                # --------------------------------------------------------------
                # Dust is *not* included, but the calculated S_CO does not cover the entire CO disk.
                # --------------------------------------------------------------

                # Gas temperature, density, opacities and source function in whole disk.
                # Continuum flux was calculated earlier.
                T_gas = cfg.t_ex(R_CO, ti, ri, p)
                NCO = cfg.nco(R_CO, ni, ri, q)
                S_CO, tau_CO = co_bandhead(T_gas=T_gas, NCO=NCO, wave=wvl, A_einstein=A_einstein,
                                           jlower=jlower, jupper=jupper, freq_trans=freq_trans, Elow=Elow,
                                           prof_dict=profile_dict, dust=False)

                R_CO_only = R_CO
                gas_only_exist = True

            # --------------------------------------------------------------
            # ITERATIONS OVER INCLINATION.
            # --------------------------------------------------------------
            to_save = np.zeros((len(dv0_cm), len(inc), len(obj_wvl_arr)))
            all_in_norm = np.ones((len(dv0_cm), len(inc)), dtype=bool)

            for j, i in enumerate(inc):
                # --------------------------------------------------------------
                # CO flux where there is only gas.
                # --------------------------------------------------------------
                if gas_only_exist:
                    # Integrate over all angles to get velocity profile for convolution.
                    int_theta_CO = fixed_quad(cfg.integrand_gauss, 0, 2 * np.pi,
                                              args=(vel_Kep, R_CO_only, Mstar, i), n=100)[0]

                    # CO flux where there is only gas.
                    flux_CO, dF_CO = calculate_flux(S_CO, tau_CO, i, R_CO_only, wvl,
                                                    convolve=True, int_theta=int_theta_CO, dv=dv)

                # --------------------------------------------------------------
                # Total fluxes for different cases.
                # --------------------------------------------------------------
                if not dust or not dust_in_gas:
                    # --------------------------------------------------------------
                    # Dust is not included or there is no overlap or both.
                    # --------------------------------------------------------------
                    flux_tot = flux_cont_tot + flux_CO

                elif dust and dust_in_gas:
                    # --------------------------------------------------------------
                    # Dust is included, and gas and dust overlap. Flux_mix has to be calculated.
                    # --------------------------------------------------------------

                    # Find the best dust fit for this inclination.
                    modelname, Ti_d, p_d, Ni_d, q_d, i_d, Ri_d_AU, gas_mass, chi_sq, chi_sq_red = inc_dust_dict.get(
                        inc_deg[j])[0]

                    # ---------------------------------------------------------
                    # Temperatures, densities and source function for parts where gas and dust overlap.
                    # (The dust has its own parameters, independent of the gas, dependent on the inclination.)
                    # ---------------------------------------------------------

                    T_gas_mix = cfg.t_ex(R_dust[mix], ti, ri, p)
                    NCO_mix = cfg.nco(R_dust[mix], ni, ri, q)

                    NH = cfg.nco(R_dust[mix], Ni_d, Ri_d, q_d) * cfg.H_CO
                    T_dust = cfg.t_ex(R_dust[mix], Ti_d, Ri_d, p_d)

                    S_mix, tau_mix, tau_cont, BB_dust = co_bandhead(T_gas=T_gas_mix, NCO=NCO_mix, wave=wvl,
                                                                    A_einstein=A_einstein, jlower=jlower, jupper=jupper,
                                                                    freq_trans=freq_trans, Elow=Elow,
                                                                    prof_dict=profile_dict,
                                                                    dust=True, dust_params=[NH, T_dust])

                    # ---------------------------------------------------------
                    # Dust continuum from part where gas and dust overlap.
                    # ---------------------------------------------------------
                    flux_dust, dF_flux_dust = calculate_flux(BB_dust, tau_cont, i, R_dust[mix], wvl)

                    # ---------------------------------------------------------
                    # Total continuum if inclination dependent.
                    # ---------------------------------------------------------
                    if not sed_best_fit:
                        flux_cont_tot = inc_dust_dict.get(inc_deg[j])[1]

                    # ---------------------------------------------------------
                    # Total flux from of part where gas and dust overlap.
                    # ---------------------------------------------------------

                    # Integrate over all angles to get velocity profile for convolution.
                    int_theta_dust = fixed_quad(cfg.integrand_gauss, 0, 2 * np.pi,
                                                args=(vel_Kep, R_dust[mix], Mstar, i), n=100)[0]

                    flux_mix, dF_mix = calculate_flux(S_mix, tau_mix, i, R_dust[mix], wvl,
                                                      convolve=True, int_theta=int_theta_dust, dv=dv)

                    # ---------------------------------------------------------
                    # Total flux of all parts before extinction.
                    # ---------------------------------------------------------

                    flux_tot = flux_CO + flux_mix + flux_cont_tot - flux_dust

                # ---------------------------------------------------------
                # Total fluxes after extinction and normalize.
                # ---------------------------------------------------------

                flux_norm = flux_tot / flux_cont_tot

                for k in range(len(dv0_cm)):
                    flux_norm_intp = np.interp(obj_wvl_arr, wvl, flux_norm[k])
                    to_save[k, j, :] = flux_norm_intp
                    flux_norm_max = np.max(flux_norm_intp)
                    all_in_norm[k, j] = maxmin[0] > flux_norm_max > maxmin[1]

                if convolve:
                    # ---------------------------------------------------------
                    # Convolution with the instrumental profile. Fluxes in erg/s/cm^2/micron.
                    # ---------------------------------------------------------
                    flux_tot_ext = cfg.flux_ext(np.squeeze(flux_tot), wvl, A_v, R_v)
                    flux_cont_tot_ext = cfg.flux_ext(np.squeeze(flux_cont_tot), wvl, A_v, R_v)
                    flux_norm_ext = flux_tot_ext / flux_cont_tot_ext

                    ip, dx = instrumental_profile(wvl, Res, vupper=vupper, vlower=vlower)
                    obs_flux = np.convolve(flux_tot_ext, ip, mode="same") * dx
                    obs_flux_norm = np.convolve(flux_norm_ext, ip, mode="same") * dx

                    # Time the duration.
                    end = time()
                    runtime = "\n runtime grid = " + str((end - start) // 60) + " min " + str(
                        (end - start) % 60) + " sec"
                    print(runtime)

                    return wvl, flux_tot_ext, flux_norm_ext, flux_norm_intp, obs_flux, obs_flux_norm

            if save is not None:

                out_file = open(folder / "out.txt", "a")

                for k, element in enumerate(dv0_cm):

                    filename = cfg.filename_co_grid_point(ti, p, ni, q, ri_R, dv=element / 1.e5)

                    if True in all_in_norm[k, :]:
                        np.save(folder / filename, to_save[k, ...])
                        out_file.write(st + " " + filename + " saved. \n")
                        print(st, filename, " saved.")
                    else:
                        out_file.write(st + " " + filename + " discarded. \n")
                        print(st, filename, " discarded.")

                out_file.close()

    # Time the duration.
    end = time()
    runtime = "\n runtime grid = " + str((end - start) // 60) + " min " + str((end - start) % 60) + " sec"
    print(runtime)

    return
