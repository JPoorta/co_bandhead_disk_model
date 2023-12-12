"""
Contains a function to calculate an SED from dust-disk and stellar parameter input.
"""
import numpy as np

import model.config as cfg


def SED_full(Ri, Ti, p, Ni, q, inc, star, A_v, R_v, wvl=None, num_r=200, R_max=None, dust_params=None):
    """
    Calculates one full SED (dust + star) from free parameter input and stellar parameters in function of
    wavelength points <wvl> (default cfg.wsil).

    :param Ri: inner radius in AU.
    :param Ti: initial temperature in K.
    :param p: power law exponent for the dust temperature.
    :param Ni: initial H2 surface density in cm^-2.
    :param q: power law exponent for the H2 surface density.
    :param inc: inclination in degrees (scalar!).
    :param star: string with name of the star.
    :param A_v: Extinction parameter Av.
    :param R_v: Extinction parameter Rv.
    :param wvl: wavelength array to calculate the SED on. (default `cfg.wsil`)
    :param num_r: number of radial grid points in the disk.
    :param R_max: outer radius, default from config is 500 AU.
    :param dust_params: list containing arrays: [R,NH,T_dust]. If provided overrides all other disk parameters except
            inclination; also only the non-extincted flux is returned.
    :return: wavelength array in micron, total flux, extincted star and dust fluxes, all in ergs/s/cm^2/micron.
    """

    if wvl is None:
        wvl = cfg.wsil

    if R_max is None:
        R_max = cfg.r_max_def

    inc_rad = inc * np.pi / 180

    if dust_params is None:
        R = np.geomspace(Ri * cfg.AU, R_max, num=num_r)
        NH = Ni * (R / (Ri * cfg.AU)) ** q  # column density H2
        Tdust = Ti * (R / (Ri * cfg.AU)) ** p

    else:
        R, NH, Tdust = dust_params

    # Optical depth of the dust -> Should it not be 2*protonmass (H2 and not H)?:
    # yes, implemented 03-02-2020.
    tau = (((np.pi * cfg.radg ** 2) * cfg.Qabs[None, :]) * NH[:, None] *
           2 * cfg.mass_proton / cfg.mass_grain / cfg.gas_to_solid)
    tau_int = np.array([np.interp(wvl, cfg.wsil[::-1], tau[r, :][::-1]) for r in range(len(R))])
    BB_dust = cfg.planck_wvl(wvl[None, :], Tdust[:, None])

    flux_star = cfg.stellar_cont(star, wvl)

    intensity_dust = BB_dust * (1. - np.exp(-tau_int / np.cos(inc_rad)))
    flux_dust = 2 * np.pi * np.trapz(intensity_dust.T * R, x=R) * np.cos(inc_rad) / cfg.d ** 2

    flux_tot_not_ext = flux_star + flux_dust

    if dust_params is not None:
        return flux_tot_not_ext

    else:
        fl_dust_ext = cfg.flux_ext(flux_dust, wvl, A_v, R_v)
        fl_star_ext = cfg.flux_ext(flux_star, wvl, A_v, R_v)
        flux_tot = fl_dust_ext + fl_star_ext

        return wvl, flux_tot, fl_star_ext, fl_dust_ext, flux_tot_not_ext


def full_sed_alma(obj, **sed_params):
    """
    Calculate a full (star+dust) extincted and non-extincted SED in units erg/s/cm2/micron for an object, as a function
    of the wavelength array given in sed_params.
    :param obj: (str) object name.
    :param sed_params: (dict) parameters to be passed on to the dust sed (dust_sed)
    :return: a full extincted SED in units erg/s/cm2/micron.
    """
    dust_sed, dust_sed_ext = sed_dust_alma(obj, **sed_params)  # dust seds
    wvl = sed_params["wvl"]
    flux_star = cfg.stellar_cont(obj, wvl)
    r_v, a_v = cfg.stel_parameter_dict.get(obj)[6:8]
    fl_star_ext = cfg.flux_ext(flux_star, wvl, a_v, r_v)

    return dust_sed + flux_star, dust_sed_ext + fl_star_ext


def sed_dust_alma(obj, ri, ni, wvl, ti=1500, p=-0.5, inc=45, r_out=None, q=-1.5,
                  opacities=None, dust_to_gas=None, beta=None, r_half=None, dflux=False):
    """
    Calculate the dust emission based on disk parameters, dust opacities in function of the wavelength array that comes
    with the opacities.

    :param obj: (string) object name.
    :param ri: inner disk radius in AU, default is taken to be the point where the dust sublimation temperature 1500 K
    is reached.
    :param ni: gas mass column density at ri in g/cm**2, if not provided based on CO results.
    :param wvl: wavelength array
    :param ti: dust temperature at ri (default 1500).
    :param p: temperature power law exponent.
    :param inc: inclination of the disk in degrees.
    :param r_out: outer disk radius in AU. If neither mass nor r_out is provided, defaults to half the ALMA resolution.
    :param q: column density power law exponent.
    :param opacities: a 2-item list containing the preferred density and an integer referring to ice-coating: 0 for no,
    2 for thin ice coating (1: thick is irrelevant). Default: [1e6, 2]
    :param dust_to_gas: the dust to gas mass ratio. If provided the ratio is (the provided) constant throughout the
    disk, regardless of input beta and r_half. If not provided a logistic function is used based on beta and
    r_half. If nothing is provided defaults to the dust to gas ratio defined in config.
    :param beta: (float) the rate of growth for the dust to gas ratio as a logistic function of r (only if both beta
    and r_half are provided and dust_to_gas is None).
    :param r_half: (float) radius at which half the final dust to gas ratio is reached in the logistic function of r
    determining the dust to gas ratio (only if both beta and r_half are provided and dust_to_gas is None).
    :param dflux: if True, do not return the total flux, but the cumulative flux in function of [radius, wvl].
    :return: both the unextincted and extincted flux in units of erg/s/cm2/micron.
    """

    t_s, log_g, r_s, logL, r_v, a_v = cfg.stel_parameter_dict.get(obj)[1:7]

    inc *= np.pi / 180  # convert inclination to radians

    r = np.linspace(ri, r_out, num=400)
    nh = ni * (r / ri) ** q * 2 * cfg.mass_proton  # number density to mass density
    tdust = ti * (r / ri) ** p
    r *= cfg.AU

    if dust_to_gas is not None:
        dust_to_gas = np.ones(len(nh)) * dust_to_gas
    elif beta is not None and r_half is not None:
        dust_to_gas = logistic_func_gas_dust_ratio(r, beta=beta, rhalf=r_half)
    else:
        dust_to_gas = np.ones(len(nh)) / cfg.gas_to_solid

    kappa = cfg.dust_opacity_dict_alma.get(opacities[0])[2]
    kappa_int = np.interp(wvl, cfg.dust_wvl_alma, kappa)
    tau = nh[:, None] * dust_to_gas[:, None] * kappa_int[None, :]  # opacities in function of [radius,wavelength]
    bb_dust = cfg.planck_wvl(wvl[None, :], tdust[:, None])  # source function in function of [radius, wvl]

    intensity = bb_dust * (1 - np.exp(-tau / np.cos(inc)))  # intensity in function of [radius, wvl]
    flux = 2 * np.pi * np.trapz(intensity.T * r, x=r) * np.cos(inc) / cfg.d ** 2  # integrate over r to obtain flux
    if dflux:  # normalized cumulative flux in function of [radius, wvl]
        dflux = np.zeros_like(intensity)
        for j in range(len(r)):
            dflux[j, :] = 2 * np.pi * np.trapz(intensity[:j, :].T * r[:j], x=r[:j]) * np.cos(inc) / cfg.d ** 2 / flux
        return dflux

    flux_ext = cfg.flux_ext(flux, wvl, a_v, r_v)

    return flux, flux_ext


def logistic_func_gas_dust_ratio(r, beta, rhalf=None, final_ratio1=1e-8, final_ratio2=0.01):
    """
    The dust to gas ratio as a function of r.

    :param r: radial points array in cm
    :param beta: power law exponent determining the rate of growth of the logistic curve.
    :param rhalf: radius in AU at which half the ratio is reached.
    :param final_ratio1: baseline ratio (at r << rhalf)
    :param final_ratio2: limiting ratio at (r >> rhalf)
    :return: The gas to dust ratio as a logistic function of r.
    """
    radii = r / cfg.AU
    if rhalf is None:
        rhalf = radii[50]

    return final_ratio2 / (1 + np.exp(-beta * (radii - rhalf))) + final_ratio1
