import numpy as np

import model.config as cfg


def SED_full(Ri, Ti, p, Ni, q, inc, star, A_v, R_v, wvl=cfg.wsil, num_r=200, R_max=cfg.r_max_def, dust_params=None):
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
    :param wvl: wavelength array to calculate the SED on.
    :param num_r: number of radial grid points in the disk.
    :param R_max: outer radius, default from config (500 AU).
    :param dust_params: list containing arrays: [R,NH,T_dust]. If provided overrides all other disk parameters except
    inclination; also only the non-extincted flux is returned.
    :return: wavelength array in micron, total flux, extincted star and dust fluxes, all in ergs/s/cm^2/micron.
    """
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
    BB_dust = cfg.planck_wvl(wvl[None, :] / 1.e4, Tdust[:, None])

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
