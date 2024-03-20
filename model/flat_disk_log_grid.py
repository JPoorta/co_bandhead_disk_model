"""
The module containing all the actual model grid calculations. More information can be found in the
docstrings and in-code comments.
"""

from time import time
import os

import numpy as np
from scipy.integrate import fixed_quad
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

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
    prune_mask = np.ones(len(wvl), dtype=bool)
    prune_mask[np.where(np.logical_and(wvl < 2.25, wvl > 1.85))] = False
    if max_wvl > 3:
        prune_mask[np.where(np.logical_and(wvl < 4.2, wvl > 3.25))] = False

    wvl = wvl[prune_mask]

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


def def_rmax(st, rmax_in, r_out_au, ti, ri):
    """
    Outer radius of both gas and dust disk (if applicable).

    :param st: (str) star name
    :param rmax_in: (float, int, or None-type) outer radius as input parameter to the model in AU.
    :param r_out_au: (float, int) outer radius as derived from ALMA observations in AU.
    :param ti: (float, int) temperature at ri in K, grid parameter to model.
    :param ri: (float, int) reference radius for temperature and density power laws in cm.
    Grid parameter to model is in AU.
    :return: outer disk radius in cm.
    """
    if rmax_in is not None:
        rmax = np.min((rmax_in, r_out_au)) * cfg.AU  # the outer radius is maximally as constrained by ALMA
    else:
        rmax = np.min(((cfg.min_T_gas / ti) ** (1 / cfg.best_fit_params[st][2]) * ri, cfg.r_max_def))

    return rmax


def def_rmin(rmin_in, ri):
    """
    Inner radius of gas disk. Can be different from reference radius of the power laws iff 'rmin_in' is given.

    :param rmin_in: (float, int, or None-type) inner model disk radius as input parameter to the model in AU.
    :param ri: (float, int) reference radius for temperature and density power laws in cm.
    Grid parameter to model is in AU.
    :return: inner disk radius in cm.
    """
    if rmin_in is not None:
        rmin = rmin_in * cfg.AU
    else:
        rmin = ri

    return rmin


def def_r_co(rmin, rmax, num_co):
    """
    Full radial array for gas disk model.

    :param rmin: (float or int) inner radius of gas disk in same units as 'rmax'.
    :param rmax: (float or int) outer radius of gas disk in same units as 'rmin'.
    :param num_co: (int) number of radial points, input parameter to model.
    :return: radial array for gas disk model in same units as rmin and rmax.
    """
    return np.geomspace(rmin, rmax, num=num_co)


def co_only_mask(r_co, ri_d):
    """
    Mask on the r_co indicating where there is no dust.

    :param r_co: radial array for entire co disk in same units as ri_d
    :param ri_d: inner radius of dust disk in same units as r_co.
    :return: mask on r_co
    """
    return r_co < ri_d


def create_radial_array(st, ri_au, rmax_in, rmin_in, ri_d_au, r_out_au, ti, num_co):
    """
    Define and return the radial array and all associated variables. All output is in cm (except the 'co_only' mask).

    :param st: (str) star name.
    :param ri_au: (float, int) reference radius for temperature and density power laws in AU, grid parameter to model.
    :param rmax_in: (float, int, or None-type) outer radius as input parameter to the model in AU.
    :param rmin_in: (float, int, or None-type) inner model disk radius as input parameter to the model in AU.
    :param ri_d_au: (float, int) inner radius of dust disk in AU, from SED fit results.
    :param r_out_au: (float, int) outer radius as derived from ALMA observations in AU, stored with SED fit results.
    :param ti: (float, int) temperature at ri in K, grid parameter to model.
    :param num_co: (int) number of radial points, input parameter to model.
    :return:
    """

    # All length units below are cm.
    ri = ri_au * cfg.AU
    ri_d = ri_d_au * cfg.AU

    rmax = def_rmax(st, rmax_in, r_out_au, ti, ri)
    rmin = def_rmin(rmin_in, ri)
    r_co = def_r_co(rmin, rmax, num_co)
    co_only = co_only_mask(r_co, ri_d)

    return rmax, rmin, ri, ri_d, r_co, co_only


def create_t_gas_array(r_co, co_only, ti, ri, t1=None, a=None, p=None, p_d=None):
    """
    CO gas temperature structure, based on given input. If 't1' and 'a' are both given (i.e. not None), the temperature
    exponentially decays until the dust sublimation radius(dsr). After the dsr it follows the same power law as the
    dust. If either 't1' or 'a' is None, the original power law is returned using p.
    Both t1 and a OR p have to be provided.
    All array types below have to have 1 element except r_co.

    :param r_co: (array) the full co disk radial array in cm.
    :param co_only: (array mask on r_co) marks the radii within the dust sublimation radius.
    :param ti: (scalar or array) temperature in K at ri.
    :param ri: (scalar or array) reference disk radius for ti in cm.
    :param t1: (scalar or NoneType) the baseline to which the temperature asymptotically decays.
    :param a: (scalar or NoneType) the decay rate.
    :param p: (scalar or array) power law exponent. Only used if 't1' or 'a' is not given.
    :param p_d:(scalar or array) power law exponent of the dust. Is needed is 't1' and 'a' are provided.
    :return:
    """

    if t1 is not None and a is not None:
        T_gas_1 = cfg.exp_t(r_co[co_only], ti, ri, t1, a)
        T_gas_2 = cfg.t_simple_power_law(r_co[~co_only], T_gas_1[-1], r_co[co_only][-1], p_d)
        T_gas = np.concatenate((T_gas_1, T_gas_2))
    else:
        T_gas = cfg.t_simple_power_law(r_co, ti, ri, p)

    return T_gas


def create_t_dust_array(r_co, co_only, ti_d, ri_d, p_d):
    """
    Temperature of the dust in the part of the disk where it exists.

    :param r_co: radial array of the gas disk in same units as ri_d.
    :param co_only: mask to r_co where there is only gas.
    :param ti_d: dust temperature in K at inner dust disk radius ri_d.
    :param ri_d: inner radius of dust disk (and ref radius for ti_d) in same units as r_co.
    :param p_d: power law exponent of the dust.
    :return:
    """

    return cfg.t_simple_power_law(r_co[~co_only], ti_d, ri_d, p_d)


def co_bandhead(t_gas, NCO, wave, A_einstein, jlower, jupper, freq_trans, Elow, prof_dict, dust, dust_params=None,
                plot=False, species="12C16O"):
    """
    Calculates the source function and opacities ifo wavelength for the CO-bandheads and (iff dust=True) for the
    dust continuum.

    :param t_gas: Temperature of the CO gas in function of R.
    :type t_gas: 1D array with `len(R)`
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
    :type dust_params: list of two 1D arrays of `len(R)`: [NH, T_dust]
    :param plot: if True plot the CO optical depth for the first temperature in the T array on figure 7.
    :return: iff dust == `False`: Source function and opacity for CO; resp. 2D array of shape (len(R),len(wave)) and 3D
        array of shape `(len(dv0), len(R), len(wave))`.

        iff dust = `True`: Total source function and opacity for CO+dust; two 3D arrays of shape (len(dv0),len(R),
        len(wave)), and opacity and source function for dust; two 2D arrays of shape (len(R),len(wave)).
        Total four arrays.
    """

    # -----------------------------------------------------------------------------
    # Reading of the partition sums and calculation fractional level populations in LTE
    # -----------------------------------------------------------------------------
    q_t = cfg.partition_sums(species)
    Q = np.interp(t_gas, q_t['T'], q_t['Q'])
    g_i = cfg.g_i_dict.get(species)  # state independent statistical weight factor, see config
    x = g_i * (2 * jlower[None, :] + 1) / Q[:, None] * np.exp(-Elow[None, :] / t_gas[:, None])

    # c**2 is not lacking in this equation! It is incorporated through the units of the transition frequencies.
    # The g_i factor in the statistical weights cancels out in the fraction and is therefore not explicitly included.
    lines = (A_einstein[None, :] / (8.0 * np.pi * freq_trans[None, :] ** 2)) * \
            (2. * jupper[None, :] + 1.) / (2. * jlower[None, :] + 1.) * \
            (1.0 - np.exp(-cfg.h * freq_trans[None, :] * cfg.c / cfg.kB / t_gas[:, None])) * x

    alpha = np.zeros((len(prof_dict.get(freq_trans[0])[0]), len(t_gas),
                      len(wave)))  # tau_CO = alpha * NCO (alpha is CO optical depth per particle)

    for i in range(lines.shape[1]):
        gauss, indices = prof_dict.get(freq_trans[i])
        for j in range(len(gauss)):
            alpha[j, ..., indices[j]] += lines[..., i][None, :] * gauss[j][:, None]

    tau_CO = alpha * NCO[None, :, None]  # optical depth of CO/gas

    if plot:
        plt.figure(7)
        plt.title("tau_CO")
        T = int(t_gas[0])
        plt.plot(wave, tau_CO[0][0], label='T_eff=' + str(T) + " K", zorder=T)
        plt.ylim(0.1, 7.e4)
        plt.legend()
        plt.xlabel("wvl (um)")

    # Source function gas in erg/s/cm^2/sr/micron
    BB_CO = cfg.planck_wvl(wave[None, :], t_gas[:, None])

    if not dust:

        return BB_CO, tau_CO

    else:
        NH, T_dust = dust_params

        kappa = cfg.dust_opacity_dict_alma.get(cfg.default_opacities[0])[cfg.default_opacities[1]]
        kappa_int = np.interp(wave, cfg.dust_wvl_alma, kappa)
        tau_cont = NH[:, None] * kappa_int[None, :]  # opacities in function of [radius,wavelength]
        bb_dust = cfg.planck_wvl(wave[None, :], T_dust[:, None])  # source function in function of [radius, wvl]

        # Total optical depth
        tau_tot = tau_CO + tau_cont[None, :, :]
        # Total source function erg/s/cm^2/sr/micron
        S_tot = (tau_CO * BB_CO[None, :, :] + tau_cont[None, :, :] * bb_dust[None, :, :]) / tau_tot

        return S_tot, tau_tot, tau_cont, bb_dust


def calculate_flux(S, tau, i, R, wvl, convolve=False, int_theta=None, dv=None, dF=None, plot=None):
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
            velocities with spacing dv. This parameter is a velocity profile (by construction integrated over a ring,
            i.e., 2 pi) for each radius (units (cm/s)^-1).
    :param dv: velocity resolution in cm/s of vel_interp and of vel_Kep (see int_theta).
    :param dF: (None or str) if given should specify the filename to which the arrays can be saved that are needed to
            calculate flux per ring.
    :param plot: option to plot the convolved (fig 8) and unconvolved (fig 9) CO intensities along with
    the source functions. To use pass a list or tuple containing a string and a parameter value, to be used for the
    label. Tested for runs with different temperatures.
    :return: flux in erg/s/cm^2/micron (1D arrays with len(wvl))
     and the filename for the saved arrays to calculate dF or None.
    """

    # Intensity (optionally) convolved with velocity profile (int_theta).
    if S.ndim == 2:
        intensity = S[None, :, :] * (1. - np.exp(-tau / np.cos(i)))
    elif S.ndim == 3:
        intensity = S * (1. - np.exp(-tau / np.cos(i)))

    if convolve:
        intensity_conv = np.array([fftconvolve(intensity[j, ...], int_theta, mode="same", axes=-1) * dv
                                   for j in range(intensity.shape[0])])
    else:
        intensity_conv = intensity * 2 * np.pi

    if plot is not None:
        plt.figure(8)
        plt.title("Intensities")
        p = plt.plot(wvl, intensity_conv[0][0], label=plot[0] + str(plot[1]))
        c = p[0].get_color()
        plt.plot(wvl, S[0], color=c, label="Source function (BB)")
        plt.xlabel('wvl (um)')
        plt.ylim(10 ** 4, 10 ** 10)
        plt.legend()

        plt.figure(9)
        plt.title("Unconvolved intensities")
        plt.plot(wvl, intensity[0][0] * 2 * np.pi, color=c, label=plot[0] + str(plot[1]))
        plt.plot(wvl, S[0], color=c, label="Source function (BB)")
        plt.xlabel('wvl (um)')
        plt.ylim(10 ** 4, 10 ** 10)
        plt.legend()

    # Flux and differential of the flux.
    flux = np.array([np.trapz(intensity_conv[j, ...].T * R, x=R, axis=1) * np.cos(i) / cfg.d ** 2
                     for j in range(intensity.shape[0])])
    if "dF" in dF:
        np.save(dF, intensity_conv)

    return flux, dF


def instrumental_profile(wvl, res, center_wvl=None):
    """
    Calculates instrumental profile for convolution to compare with observations. The units of the instrumental
    profile are micron^-1; the convolution product should be multiplied by dx to approximate continuous convolution.

    :param wvl: wavelength array that goes with the flux, in micron.
    :param res: spectral resolution R of the observation.
    :param center_wvl: reference wavelength for resolution and pivot point for convolution (0-point velocity).
    (optional, any wavelength point in wvl is allowed, with no significant effects on the ip*dx result,
     provided it is more than 5 sigma away from min or max)
    :return: a gaussian profile (in micron^-1) and the wavelength step dx in micron.
    """

    # Wavelength of onset of the bandhead in micron.
    if center_wvl is not None:
        center = center_wvl
    else:
        center = np.median(wvl)

    sigma = center / res  # sigma in micron
    # Wavelengths in micron within 5 sigma.
    wvl_ip = wvl[np.where(np.abs(wvl - center) < 5 * sigma)]
    ip = cfg.gaussian(wvl_ip, center, sigma)  # gaussian centered around w0 (in micron^-1)
    dx = np.mean(wvl_ip[1:] - wvl_ip[0:-1])

    return ip, dx


def run_grid_log_r(grid, inc_deg, stars, dv0, vupper, vlower, nJ, dust, species, iso_ratio, num_CO=100,
                   Rmin_in=None, Rmax_in=None, print_Rs=False, convolve=False, save=None, maxmin=(1.3, 1.02),
                   lisa_it=None, saved_list=None, dF=None, save_reduced_flux=True):
    """
    Calculates a grid of CO bandhead profiles and optionally save the normalized profiles and the wavelength array.
    For a test run use scalar values in the grid and for dv0, and set convolve = True.

    :param iso_ratio: (int or None) the relative N12C16O/'species' abundance. If None, 12C16O is assumed to be the
    only molecule. If passed with "12C16O", the total NCO/NH abundance is assumed to be shared with another
    isotopologue (so far only 13CO), i.e. the 12C16O abundance is then reduced by the amount of the other isotope.
    :param species:(str) isotopologue name; so far only "12C16O" or "13C16O".
    :param grid: grid with 5 variables, created as follows by np.meshgrid:

        .. code-block:: python

            tiv, t1v, av, pv, niv, qv, riv = np.meshgrid(ti, t1, a, p, ni, q, ri, sparse=False)
            grid = [tiv, t1v, av, pv, niv, qv, riv]

        With ti, t1, a, p, ni, q, ri arrays or scalars:

         - ti: initial CO gas temperature at ri (in K).
         - t1: baseline temperature in K for exponential decoy law. This is default, unless 't1' or 'a' is None, then
         code defaults to power law.
         - a: decay rate in case an exponential decoy law is used (a<0). (if None, code defaults to power law)
         - p: power law exponent for the gas temperature (p<0). Default is exponential decay law even if p is provided.
         Either 'p' or 't1' and 'a' should be provided.
         - ni: initial gas (H2) surface density at Ri (in cm^-2).
         - q: power law exponent for the gas surface density (q<0).
         - ri: initial radius for the powerlaws for the CO gas disk (in AU).
    :param inc_deg: array of inclination(s) in degrees. This can only contain values from [10,20,30,40,50,60,70,80]
    :param stars: array of string(s) with the object names.
    :param dv0: (scalar or array) doppler width of the lines in km/s. If array, for each element the bandheads
        are calculated and (if relevant) saved in separate files with corresponding filenames.
    :param vupper: list of upper level(s) of the vibrational transition(s).
    :param vlower: list of lower level(s) of the vibrational transition(s).
    :param nJ: number of rotational transitions to be taken into account.
    :param dust: boolean. If True dust is included. If False the total continuum (for normalizing) still includes dust,
    but the gas emission is calculated without involving dust.
    :param num_CO: (optional) integer specifying amount of radial points for the CO gas disk. (default = 50)
    :param Rmin_in: (optional) initial radius for the CO gas disk (in AU). If not provided defaults to Ri.
    :param Rmax_in: (optional) outer radius for the CO gas disk (in AU). If not provided defaults
            to the point where the gas temperature drops below 1000 K (is thus dependent on Ri, Ti, p and hence
            different per grid point).
    :param print_Rs: (optional) boolean. If True information on the radial arrays is printed. (default = False)
    :param convolve: (optional) boolean. If True the fluxes are convolved with the instrumental profile and
            returned in the first loop over the grid and inclination. The grid is then not iterated over and no fluxes
            are saved regardless of the value of save. (default = False)
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
    :param dF: (None or str) if provided, the necessary arrays to calculate the cumulative flux are saved to the results
            folder of the relevant star in a folder starting with "dF". The string will be appended to this folder name.
            If an empty string: arrays are saved to dF. If None, nothing is saved.
    :param save_reduced_flux: (bool) if True save flux cast on a reduced wavelength array adapted to X-shooter
            observations (this was done to save space for large grids).
            If False the original flux with model wavelength array will be saved.
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
    vlow, jl_all, vhigh, jh, A, El_all, freq_tr = cfg.co_data(species)

    max_freq = 1.e4 / cfg.max_wvl  # micron to cm^-1
    mask = np.zeros(len(freq_tr), dtype=bool)

    for vu, vl in zip(vupper, vlower):
        mask += np.logical_and.reduce(
            (vhigh == vu, vlow == vl, jh <= nJ, jl_all <= nJ, freq_tr >= max_freq))

    ind = np.argsort((freq_tr[mask]))

    vl, jlower, vh, jupper, A_einstein, Elow, freq_trans = \
        [(a[mask][ind]) for a in [vlow, jl_all, vhigh, jh, A, El_all, freq_tr]]

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
            os.makedirs(folder, exist_ok=True)
            if save_reduced_flux:
                np.save(folder / "wvl_re", obj_wvl_arr)
            else:
                np.save(folder / "wvl_re", wvl)

        # --------------------------------------------------------------
        # Get stellar parameters.
        # --------------------------------------------------------------

        Mstar, T_eff, log_g, Rstar, Res, SNR, R_v, A_v, RV = cfg.stel_parameter_dict.get(st)

        # --------------------------------------------------------------
        # Continuum flux (not extincted), always includes stellar and dust continuum (also when dust=False).
        # Based on SED fit of object. Independent of inclination and outer radius (Rmax) of gas disk.
        # --------------------------------------------------------------

        modelname, ti_d, p_d, ni_d, q_d, i_d, ri_d_AU, r_turn, beta, r_out_AU = cfg.best_dust_fit_ALMA.get(st)
        sed_params = {"obj": st, "ri": ri_d_AU, "ni": ni_d, "ti": ti_d, "p": p_d, "inc": i_d, "r_out": r_out_AU,
                      "q": q_d, "opacities": cfg.default_opacities, "beta": beta, "r_half": r_turn, "wvl": wvl}
        flux_cont_tot = seds.full_sed_alma(**sed_params)[0]

        # --------------------------------------------------------------
        # ITERATIONS OVER THE GRID.
        # --------------------------------------------------------------

        it = np.nditer(grid, ["ranged", "refs_ok"])

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
            ti, t1, a, p, ni, q, ri_R = (x[0], x[1], x[2], x[3], x[4], x[5], x[6])

            if ti < 0:
                ti = T_eff

            filename = cfg.filename_co_grid_point(ti, p, ni, q, ri_R, t1=t1[()], a=a[()])
            if dF is not None:
                dF_use = str(cfg.results_folder / st / ("dF" + dF) /
                             cfg.filename_co_grid_point(ti, p, ni, q, ri_R, dv=dv0_cm[0] / 1.e5, t1=t1[()], a=a[()]))
                np.save(dF_use + "_wvl", wvl)
            else:
                dF_use = ""

            if saved_list is not None:
                if not np.any([filename in listed for listed in to_be_calculated]):
                    print(filename + " skip")
                    continue

            # ---------------------------------------------------------
            # Creation of the radial arrays.
            # ---------------------------------------------------------
            Rmax, Rmin, ri, ri_d, R_CO, CO_only = \
                create_radial_array(st, ri_R, Rmax_in, Rmin_in, ri_d_AU, r_out_AU, ti, num_CO)

            # --------------------------------------------------------------
            # Dust and gas can overlap (dust_in_gas=True), and if the gas disk starts (Rmin) earlier than the dust
            # (ri_d) there is a part of the disk with only gas (gas_only_exist).
            # --------------------------------------------------------------
            dust_in_gas = ri_d <= Rmax
            gas_only_exist = Rmin < ri_d

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
                           + '\n length R_CO_dust =' + str(len(R_CO[~CO_only])) \
                           + '\n dust disk =' + str(ri_d_AU) + " to " + str(r_out_AU) \
                           + "\n length wvl = " + str(len(wvl))

                print(to_print)

            # --------------------------------------------------------------
            # Gas and dust temperatures and densities.
            # --------------------------------------------------------------
            T_gas = create_t_gas_array(R_CO, CO_only, ti, ri, t1[()], a[()], p, p_d)
            T_dust = create_t_dust_array(R_CO, CO_only, ti_d, ri_d, p_d)

            NCO = cfg.n_co(nh=cfg.nh(R_CO, ni, ri, q,), species=species, n12co_species_ratio=iso_ratio)
            dust_to_gas = seds.logistic_func_gas_dust_ratio(R_CO[~CO_only], beta=beta, rhalf=r_turn)
            NH = cfg.nh(R_CO[~CO_only], ni_d, ri_d, q_d) * dust_to_gas * 2 * cfg.mass_proton  # dust mass column density

            # Opacities and source function where there is only gas.
            if gas_only_exist:
                S_CO, tau_CO = co_bandhead(t_gas=T_gas[CO_only], NCO=NCO[CO_only], wave=wvl,
                                           A_einstein=A_einstein, jlower=jlower, jupper=jupper,
                                           freq_trans=freq_trans, Elow=Elow, species=species,
                                           prof_dict=profile_dict, dust=False, plot=False)
                R_CO_only = R_CO[CO_only]

            else:
                flux_CO = np.zeros(len(wvl))

            # --------------------------------------------------------------
            #
            # --------------------------------------------------------------

            if not dust and dust_in_gas:
                # --------------------------------------------------------------
                # Dust is *not* included, but the calculated S_CO does not cover the entire CO disk.
                # --------------------------------------------------------------

                # Gas opacities and source function in whole disk.
                # Continuum flux was calculated earlier.
                S_CO, tau_CO = co_bandhead(t_gas=T_gas, NCO=NCO, wave=wvl, A_einstein=A_einstein,
                                           jlower=jlower, jupper=jupper, freq_trans=freq_trans, Elow=Elow,
                                           prof_dict=profile_dict, dust=False, species=species)

                R_CO_only = R_CO
                gas_only_exist = True

            # --------------------------------------------------------------
            # ITERATIONS OVER INCLINATION.
            # --------------------------------------------------------------
            if save_reduced_flux:
                to_save = np.zeros((len(dv0_cm), len(inc), len(obj_wvl_arr)))
            else:
                to_save = np.zeros((len(dv0_cm), len(inc), len(wvl)))
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
                                                    convolve=True, int_theta=int_theta_CO, dv=dv,
                                                    plot=None, dF=dF_use + "_CO")
                    # e.g. plot = ['T_eff= ',int(t_gas[0])])

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

                    # ---------------------------------------------------------
                    # Temperatures, densities and source function for parts where gas and dust overlap.
                    # (The dust has its own parameters, independent of the gas, dependent on the inclination.)
                    # ---------------------------------------------------------

                    S_mix, tau_mix, tau_cont, BB_dust = co_bandhead(t_gas=T_gas[~CO_only], NCO=NCO[~CO_only], wave=wvl,
                                                                    A_einstein=A_einstein, jlower=jlower, jupper=jupper,
                                                                    freq_trans=freq_trans, Elow=Elow,
                                                                    prof_dict=profile_dict, species=species,
                                                                    dust=True, dust_params=[NH, T_dust])

                    # ---------------------------------------------------------
                    # Dust continuum from part where gas and dust overlap.
                    # ---------------------------------------------------------
                    flux_dust, dF_flux_dust = calculate_flux(BB_dust, tau_cont, i, R_CO[~CO_only], wvl,
                                                             dF=dF_use + "_dust")

                    # ---------------------------------------------------------
                    # Total flux from of part where gas and dust overlap.
                    # ---------------------------------------------------------

                    # Integrate over all angles to get velocity profile for convolution.
                    int_theta_dust = fixed_quad(cfg.integrand_gauss, 0, 2 * np.pi,
                                                args=(vel_Kep, R_CO[~CO_only], Mstar, i), n=100)[0]

                    flux_mix, dF_mix = calculate_flux(S_mix, tau_mix, i, R_CO[~CO_only], wvl,
                                                      convolve=True, int_theta=int_theta_dust, dv=dv,
                                                      dF=dF_use + "_mix")

                    # ---------------------------------------------------------
                    # Total flux of all parts before extinction.
                    # ---------------------------------------------------------

                    flux_tot = flux_CO + flux_mix + flux_cont_tot - flux_dust

                # ---------------------------------------------------------
                # Total fluxes after extinction and normalize.
                # ---------------------------------------------------------

                flux_norm = flux_tot / flux_cont_tot

                for k in range(len(dv0_cm)):
                    if save_reduced_flux:
                        flux_norm_save = np.interp(obj_wvl_arr, wvl, flux_norm[k])
                    else:
                        flux_norm_save = flux_norm[k]
                    to_save[k, j, :] = flux_norm_save
                    flux_norm_max = np.max(flux_norm_save)
                    all_in_norm[k, j] = maxmin[0] > flux_norm_max > maxmin[1]

                if convolve:
                    # ---------------------------------------------------------
                    # Convolution with the instrumental profile. Fluxes in erg/s/cm^2/micron.
                    # ---------------------------------------------------------
                    flux_tot_ext = cfg.flux_ext(np.squeeze(flux_tot), wvl, A_v, R_v)
                    flux_cont_tot_ext = cfg.flux_ext(np.squeeze(flux_cont_tot), wvl, A_v, R_v)
                    flux_norm_ext = flux_tot_ext / flux_cont_tot_ext

                    ip, dx = instrumental_profile(wvl, Res)
                    obs_flux = np.convolve(flux_tot_ext, ip, mode="same") * dx
                    obs_flux_norm = np.convolve(flux_norm_ext, ip, mode="same") * dx

                    # Time the duration.
                    end = time()
                    runtime = "\n runtime grid = " + str((end - start) // 60) + " min " + str(
                        (end - start) % 60) + " sec"
                    print(runtime)

                    return wvl, flux_tot_ext, flux_norm_ext, flux_norm_save, obs_flux, obs_flux_norm

            if save is not None:

                out_file = open(folder / "out.txt", "a")

                for k, element in enumerate(dv0_cm):

                    filename = cfg.filename_co_grid_point(ti, p, ni, q, ri_R, t1=t1[()], a=a[()], dv=element / 1.e5)

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
