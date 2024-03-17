"""
This script declares all global constants and parameters, reads in the necessary data (from aux_files),
sets folder paths, contains widely used general functions (e.g. black body function) etc. Most parameters and
functions are explained in docstrings and/or comments.
"""

import pickle

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

import os
from pathlib import Path

# CONSTANTS

kB = 1.380649e-16  # Boltzman constant erg/K
h = 6.62607015e-27  # Planck constant erg.s
c = 2.99792458e10  # speed of light cm/s
c_km = 2.99792458e5  # speed of light km/s
G = 6.674e-8  # Gravitational constant in cgs (cm3/g/s^2)
cm_K = c * h / kB  # conversion cm-1 to K
mass_proton = 1.67262e-24  # proton mass in g
CO_mass = 28.010 / 6.022e23  # mass in g of a CO molecule (molecular weight of CO in g/mol divided by Avogadro's nr)
pc = 3.086e18  # parsec in cm
AU = 1.496e13  # AU in cm
R_sun = 6.955e10  # solar radius cm
M_sun = 1.988475e+33  # solar mass in gr

# PARAMETERS

# Various
sampling = 2  # Shannon minimum sampling
max_wvl = 5  # Maximum wavelength in micron - cut off used in co_bandhead
min_dv_cm = 1. * 1.e5  # velocity resolution in cm/s for construction of frequency and velocity arrays
jwst_nirspec_res = 2700  # NIRSPEC resolution from Boker et al 2023

# Outer boundary temperature
min_T_gas = 600  # Minimum gas temperature that defines outer boundary of the model.

# Default outer boundary for log_grid
r_max_def = 100 * AU

# Dust parameters
radg = 1e-5  # grain radius in cm (0.1 micron)
rho_grain = 2.0  # grain density (g/cm^3)
mass_grain = rho_grain * (4 / 3) * np.pi * radg ** 3  # grain mass (g?)
max_T_dust = 1500  # maximum dust temperature
min_dust_T = 0.01  # dust temperature used when dust is considered to be absent from the model
gas_to_solid = 100  # canonical gas to dust ratio

# Stellar parameters and extinction
# [stellar mass from track in M_sun,T_eff(K),log_g,stellar radius in R_sun, Resolution, SNR, R_v,A_v, RV(in km/s)]
stel_parameter_dict = {'B275': [7.2, 12750, 3.44, 9.5,  11300., 100, 3.8, 7.41, -11.2],
                       'B331': [10.0, 13000, 4.13, 18.7, 11300, 100, 4.6, 13.3, 13.9],
                       'B243': [4.2, 11900, 3.78, 4.6, 11300., 100, 4.7, 7.92, 20.2],
                       'B268': [4.5, 11300, 3.78, 5.6, 11300., 100, 4.6, 7.51, 4.3],
                       'B163': [6.0, 8250, 3.63, 8.7, 11300., 100, 4.0, 13.21, 0.0],
                       }

# Best fit parameters from previous paper: T_i (K) [0], upper/lower errors [1], p [2], N_H_i [cm^-2] [3],
# upper/lower errors [4], R_i [AU] [5], inclination (degrees) [6], v_G [7] from best fits, Table 7
best_fit_params = {'B163': [4000, [1700, 1700], -3, 8.3e25, [96, 7.7], 0.155, 50, 1],
                   'B243': [2000, [3000, 0], -3, 1.1e25, [15, 1], 0.488, 50, 1],
                   'B268': [4500, [1000, 1000], -0.75, 3e25, [5.8, 2], 0.0941, 80, 2],
                   'B275': [4000, [2500, 1000], -2, 3e25, [22, 2.6], 0.261, 30, 1],
                   'B331': [3000, [2500, 1000], -3, 1.1e25, [17, 1], 0.486, 30, 1]}


def get_default_params(star=None):
    """
    For a star return the default model parameters, based on the best fit of the CO overtone bandheads.

    :param star: (str) default is B275
    :return: grid_params and all_params dictionaries.
    """
    if star is None:
        star = "B275"
    t_i, t_i_err, p, n_h_i, n_h_i_err, r_i, inc, v_G = best_fit_params[star]

    grid_params = {"ri": r_i,
                   "ti": t_i,  # [500,1000,2500,4000] ,
                   "p": p,  # [-0.5, -0.75, -2]
                   "ni": n_h_i,
                   "q": -1.5,
                   "t1": 800,
                   "a": -11 #TODO If the purpose here is to truly define the original defaults, t1 or a should be None.
                   }

    all_params = {"inc_deg": [inc],  # 10,20,30,40,50,60,70,80
                  "stars": [star],  # , 'B243', 'B275', 'B163', 'B331']
                  "dv0": [v_G],
                  "vupper": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 2, 3, 4, 5, 6, 7],
                  "vlower": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 0, 1, 2, 3, 4, 5],
                  "nJ": 150,
                  "dust": True,
                  # From here on only optional parameters.
                  "num_CO": 100,
                  "Rmin_in": None,
                  "Rmax_in": None,
                  "print_Rs": True,
                  "convolve": True,
                  "save": None,
                  "maxmin": (1.3, 1.02),
                  "lisa_it": None,
                  "saved_list": None,
                  "dF": None,
                  "save_reduced_flux": True
                  }

    return grid_params, all_params


# Best dust fit for each star (to avoid copying unnecessary data to LISA).
best_dust_fit = {'B163': (
    '1500_-0.1_1e+22_-4.0_80_1.4', 1500., -0.1, 1.e+22, -4.0285716, 80., 1.3682581, 2.1831985e-08, 208.63118,
    69.543724), 'B243': (
    '1400_-0.7_1e+21_-5.6_80_5.8', 1400., -0.7, 1.e+21, -5.6, 80., 5.753369, 2.1751644e-08, 228.30612, 57.07653),
    'B268': (
        '1500_-0.7_1e+21_-5.4_80_4.7', 1500., -0.7, 1.e+21, -5.4035716, 80., 4.652704, 1.5046174e-08, 45.305992,
        22.652996), 'B275': (
        '1500_-0.9_1e+21_-2.3_60_7.8', 1500., -0.9, 1.e+21, -2.2607143, 60., 7.8317246, 3.6823135e-07, 34.621643,
        8.655411), 'B331': (
        '400_-5.1_1e+21_-5.6_80_144.8', 400., -5.1, 1.e+21, -5.6, 80., 144.82933, 1.3624263e-05, 371.14423, 61.857372)}

# 0 modelname, 1 Ti_d, 2 p_d, 3 Ni_d, 4 q_d, 5 i_d, 6 Ri_d_AU, 7 R_turn, 8 beta, 9 r_out (AU)
best_dust_fit_ALMA = {'B163': ('', 1500., -1, 3.05762e+24),
                      'B243': ('', 1500., -1, 2.36156e+25, -2, 50, 1.7, 14, 0.84, 68),
                      'B268': ('', 1500., -1, 1.04626e+24, -0.8, 80., 1.8, 38, 0.2, 60),
                      'B275': ('', 1500., -1, 3.88612e+24, -1.1, 30, 4.1, 50, 0.2, 60),
                      'B331': ('', 1500., -1, 2.48114e+24, -0.8, 30, 8.5, 47, 10, 51)}

# Distance to the star in cm
d = 1.7e3 * pc

# Wavelengths of the lowest wavelength transition (the onset) of each bandhead in micron
# (according to the atomic data file with nJ = 120).

onset_wvl_dict = {(1, 0): 4.2949, (2, 1): 4.3504, (3, 2): 4.4071, (4, 3): 4.4652, (5, 4): 4.5246, (6, 5): 4.5855,
                  (7, 6): 4.6477, (8, 7): 4.7115, (9, 8): 4.7768, (10, 9): 4.8436,
                  (2, 0): 2.2935, (3, 1): 2.3227, (4, 2): 2.3525, (5, 3): 2.3829, (6, 4): 2.4141, (7, 5): 2.4461,
                  (3, 0): 1.5582, (4, 1): 1.5779, (5, 2): 1.5982, (6, 3): 1.6189, (7, 4): 1.6401, (8, 5): 1.6618,
                  (9, 6): 1.684, (10, 7): 1.7067, (11, 8): 1.73, (12, 9): 1.7538, (13, 10): 1.7783, (14, 11): 1.8033,
                  (15, 12): 1.829, (16, 13): 1.8553, (17, 14): 1.8823, (18, 15): 1.91, (19, 16): 1.9384,
                  (20, 17): 1.9675,
                  (4, 0): 1.1814, (5, 1): 1.1964, (6, 2): 1.2118, (7, 3): 1.2275, (8, 4): 1.2437, (9, 5): 1.2601,
                  (10, 6): 1.277, (11, 7): 1.2943, (12, 8): 1.3121, (13, 9): 1.3302, (14, 10): 1.3488, (15, 11): 1.3679,
                  (16, 12): 1.3874, (17, 13): 1.4074, (18, 14): 1.428, (19, 15): 1.4491, (20, 16): 1.4707}

CO13_onset = {(1, 0): 4.3915, (2, 1): 4.447, (3, 2): 4.5037, (4, 3): 4.5617, (5, 4): 4.621, (6, 5): 4.6816,
              (7, 6): 4.7437,
              (8, 7): 4.8072, (9, 8): 4.8721, (10, 9): 4.9387,
              (2, 0): 2.3448, (3, 1): 2.3739, (4, 2): 2.4037, (5, 3): 2.4341}

# FOLDERS and DATA
project_folder = Path(__file__).parent.parent

# to move up from docs and handle being in Ubuntu in CI
if "GENERATING_SPHINX_DOCS" in os.environ:
    project_folder = project_folder / os.pardir

# Folders for results
results_folder = project_folder / "results/"
# Folder for data files
aux_files = project_folder / "aux_files/"
# Folder for plots
plot_folder = project_folder / "plots/"

# folder with Kurucz models
Kurucz_dir = aux_files / "Castelli-Kurucz/"
# folder for NIR-spectra
spectra_dir = aux_files / "best_fits_and_data"
# Atomic data.
species = "12C16O"
# TODO MAIN FIG Make 'species' an input parameter to flat_disk_log_grid: with options 12CO, 13CO (with given abundance)
#  or both. The atomic data related definitions can become functions of species.
atomic_data = aux_files / ("hitran_table_" + species)
partition_sums = aux_files / ("Q_" + species + ".npy")
# State independent statistical weight factors g_i according to eq.22 in Simeckova 2006.
# The statistical weights (2*J+1) should be multiplied by this factor.
g_i_dict = {"12C16O": 1, "13C16O": 2}
# Abundance H2/12CO. Elemental abundance of 12C/13C is about 69 (source: R.Visser 2009)
# We use the hitran value for the relative abundance N(12CO)/N(13CO): 89 (https://hitran.org/docs/iso-meta/)
N12CO_N13CO = 89
H_CO = {"12C16O": 1.e4, "13C16O": 1.e4 * N12CO_N13CO}
dust_data = aux_files / 'eps_Sil'
dust_data_alma = aux_files / "dust_opacities_ossenkopf_1994.dat"
ice_dict = {0: ["no ice mantle", '-'], 1: ["thick ice mantle", '..'],
            2: ["thin ice mantle", '--']}  # dictionary for dust opacities
photometry = aux_files / 'photometry'
photometry_naira = aux_files / 'photometry_naira'
# Wavelength array onto which all models are interpolated before saving. Created by including all data points beyond
# 1.54 micron process_molecfit_output_fits [conflicted] in X-shooter data folder.
obj_wvl_array = aux_files / "wvl_obj.npy"
# Name of the pickle file where the best fits for the preferred grid is stored.
best_fits_dict = aux_files / 'best_fits_dict'


def get_dust_opacities(plot=False):
    """
    Get the dust opacities from Ossenkopf V., Henning T <Astron. Astrophys. 291, 943 (1994)> sorted in a dictionary.
    Mass absorption coefficients for MRN size distributions in cm^2/g.
    Example usage of output dictionary for thin ice mantle, density 10^6:
    >>> dust_opacity_dict.get(10**6)[2]
    :return: Wavelength array, dictionary with gas densities (0, 10^5 - 10^8) as key and the three kappa arrays as
    values.
    K0: no ice mantles
    K1: thick ice mantles
    K2: thin ice mantles
    """

    nH_rows = np.array(np.split(np.loadtxt(dust_data_alma, delimiter='|', usecols=[0, 1, 2, 4, 6]), 5)).T
    wvl = nH_rows[1, :, 0]
    nH = nH_rows[0, 0, :]
    opacity_dict = {}
    for i, nH in enumerate(nH):
        opacity_dict[nH] = nH_rows[2:, :, i]
        if plot:
            for j in [2, 4]:
                ice_mantle, linest = ice_dict.get(j - 2)
                plt.title(ice_mantle)
                if j == 2:
                    p, = plt.loglog(wvl, nH_rows[j, :, i], linest,
                                    label="nH = " + '{0:.2g}'.format(nH) + " " + ice_mantle)
                else:
                    plt.loglog(wvl, nH_rows[j, :, i], linest, c=p.get_color(),
                               label="nH = " + '{0:.2g}'.format(nH) + " " + ice_mantle)

    return wvl, opacity_dict


# Get the dust opacities for further use.
dust_wvl_alma, dust_opacity_dict_alma = get_dust_opacities(plot=False)
default_opacities = [1e8, 0]


# Pickle file location and extraction function for the dictionaries with the best dust fits per inclination.
# (to avoid copying unnecessary data to LISA)
def load_inc_dust_dict(st):
    """
    Load the dictionary with the best dust fits per inclination. This function replaces the call to seds.plot_best_fits
    in flat_disk_log_grid.
    The pickle files were created using
    sed_calculations.plot_best_fits(inc_deg, 'i', st, folder="sed_Ri_dsr", Ti= <best fit Ti> ,plot=False) with
    - Ti = 1500 for st = 'B268','B243','B275','B163' and Ti = 400 for st = 'B331'
    - inc_deg = np.array([0,10,20,30,40,50,60,70,80,89])

    :param st: String with the star name for which the dictionary is needed.
    :return:
    """
    name = aux_files / ("inc_dust_dict_" + st + ".pkl")
    with open(name, 'rb') as f:
        return pickle.load(f)


def filename_co_grid_point(ti, p, ni, q, ri_r, dv=None, extra_param=None, t1=None, a=None):
    """
    Standardized filename for saving grid items.

    :param ti: (int or float) initial temperature at ri_r in K.
    :param p: (float) temperature exponent.
    :param ni: (float) initial column density in cm^-2.
    :param q: (float) density exponent.
    :param ri_r: (float) initial (reference) disk radius in stellar radii.
    :param dv: (int or float) gaussian width in km/s. (The None option is retained for older model grids,
        that don't vary dv.)
    :param extra_param: (int or float) implemented to name varied parameters that are not in the original grid
        (like dust, yes or no, or outer radius) but not yet used. Possibly redundant.
    :return: (str) name of the model file
    """
    if t1 is None or a is None:
        base_str = str(np.int(ti)) + '_' + '{0:g}'.format(p) + '_' \
                   + str(np.format_float_scientific(int(ni), precision=2, exp_digits=2, trim='-')) + '_' \
                   + str(np.around(q, 1)) + '_' + str(np.around(ri_r, 1))
    else:
        base_str = str(np.int(ti)) + '_' + str(int(t1)) + '_' + str(int(a)) + '_' \
               + str(np.format_float_scientific(int(ni), precision=2, exp_digits=2, trim='-')) + '_' \
               + str(np.around(q, 1)) + '_' + str(np.around(ri_r, 2))
    full_name = base_str
    if dv is not None:
        full_name = full_name + '_dv' + '{0:g}'.format(dv)
    if extra_param is not None:
        full_name = full_name + '{1:g}'.format(extra_param)

    return full_name


# Molecular data
#  ------------------------------------------
# From https://hitran.org/hitemp/
# G. Li, I.E. Gordon et al. 2015, The Astrophysical Journal Supplement Series 21615.
#
# Column    Units   Label     Explanations
#
#   0        ---     vlow        Vibrational quantum number of the lower level
#   1        ---     jl_all      Rotational quantum number of the lower level
#   2        ---     vhigh       Vibrational quantum number of the upper level
#   3        ---     jh        Rotational quantum number of the upper level
#   4        s-1     A         Einstein A-coefficient
#   5        cm-1    El_all     Energies of the lower level of the transitions
#   6        cm-1    freq_tr    Frequency of the transition
#
#  -----------------------------------------------------

vlow, jl_all, vhigh, jh, A, El_all, freq_tr = np.genfromtxt(atomic_data).T
El_all *= cm_K

# Precalculated partition sums (cross-checked with the values provided by Li, Gordon et al. 2015):
# >>> T_gas = np.arange(1,9001,0.1)
# >>> El_all, uni_indices = np.unique(cfg.El_all, return_index=True)
# >>> Q = np.sum((2 * cfg.jl_all[uni_indices][:, None] + 1) * np.exp(-El_all[:, None] / T_gas[None, :]), axis=0)
# >>> to_save = np.core.records.fromarrays(np.array([T_gas,Q]), names='T, Q', formats='f4, f4')
# >>> np.save(cfg.aux_files + "Q_12C16O",to_save)
Q_T = np.load(partition_sums)

# Dust data

# ----------------------------------------------
# CONTINUUM EMISSION
# ----------------------------------------------

#  ---------------------------------------------
#  Table from https://www.astro.princeton.edu/~draine/dust/dust.diel.html
#  Astronomical silicate, radius (micron) = 1.000E-01
#  B.T.Draine, Princeton Univ. (cf Laor, A., & Draine, B.T. 1993, ApJ 402,441)
#
# Column    Units    Label     Explanations
#   0        micron   w         wavelength
#   1        ---      Re(eps-1) Real part of dielectric function (material polarization)
#   2        ---      Im(eps)   Imaginary part of dielectric function (absorption properties)
#   3        ---      Re(m-1)   Real part of ??
#   4        ---      Im(m)     Imaginary part of ??
#  ---------------------------------------------
# Read in file

wsil, Re_eps_sil, Im_eps_sil, Re_m_sil_table, Im_m_sil = np.genfromtxt(dust_data, unpack=True)

# Cut out the far UV.
mask = np.where(wsil > 0.2)
wsil = wsil[mask]

Re_m_sil = Re_eps_sil[mask] + 1  # Why?
mm = Re_m_sil[mask] - Im_eps_sil[mask] * 1j
Qabs = -4. * (2 * np.pi * radg / wsil / 1e-4) * np.imag((mm ** 2 - 1) / (mm ** 2 + 2))


# Photometric data

def jy_to_erg_s_cm2_um(f_vu, wvl):
    """
    Convert flux units. Uses c as defined in config (converts from cm/s to um/s).

    :param f_vu: Flux in Jy.
    :param wvl: Wavelength in micron.
    :return: Flux in erg/s/cm^2/um
    """
    return 1.e-23 * c * 1.e4 / wvl ** 2 * f_vu


# Flux zero points J, H, and K_s : https://old.ipac.caltech.edu/2mass/releases/allsky/doc/sec6_4a.html
# http://irsa.ipac.caltech.edu/data/SPITZER/docs/spitzermission/missionoverview/spitzertelescopehandbook/19/
# http://casa.colorado.edu/~ginsbura/filtersets.htm

# Dictionary with photometric flux zero's and wavelength points.
# (for details see photometry in auxfiles).
# (wavelength,zero point flux, error on zeropoint) in (micron, erg/s/cm^-2/um, erg/s/cm^-2/um)

m17_phot_dict = {'I': [0.791, jy_to_erg_s_cm2_um(2499, 0.791), np.nan], 'J': [1.235, 3.129e-6, 5.464e-8],
                 'H': [1.662, 1.133e-6, 2.212e-8], 'K': [2.159, 4.283e-7, 8.053e-9],
                 '3.4um': [3.3526, 8.1787e-08, 1.2118e-09],
                 '3.6um': [3.550, jy_to_erg_s_cm2_um(280.9, 3.55), jy_to_erg_s_cm2_um(4.1, 3.55)],
                 '4.5um': [4.493, jy_to_erg_s_cm2_um(179.7, 4.493), jy_to_erg_s_cm2_um(2.6, 4.493)],
                 '4.6um': [4.6028, 2.4150e-08, 3.5454e-10],
                 '5.8um': [5.731, jy_to_erg_s_cm2_um(115.0, 5.731), jy_to_erg_s_cm2_um(1.7, 5.731)],
                 '8.0um': [7.872, jy_to_erg_s_cm2_um(63.13, 7.872), jy_to_erg_s_cm2_um(0.92, 7.872)]}

M17_phot = np.loadtxt(photometry_naira,
                      dtype=({'names': ('Obj', 'I', 'e_I', 'J', 'e_J', 'H', 'e_H', 'K', 'e_K', '3.4um',
                                        'e_3.4', '3.6um', 'e_3.6', '4.5um', 'e_4.5', '4.6um', 'e_4.6', '5.8um',
                                        'e_5.8', '8.0um', 'e_8.0'),
                              'formats': ('U10', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                                          'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8')})).T

phot_frame_m17 = pd.DataFrame(M17_phot, index=M17_phot['Obj'])


def mag_to_flux(band, mag, phot_dict, sigma=None, sigma_nan=0.5):
    """
    Convert magnitude to flux in erg s^-1 cm^-2 um^-1.

    :param band: photometry band as listed in the phot_dict
    :type band: string
    :param mag: magnitude of the object
    :param phot_dict: the photometry dictionary containing per band (wavelength,zero point flux, error on zeropoint).
    :param sigma: optional error on magnitude.
    :param sigma_nan: the fractional error to be assigned when sigma is nan.
    :return: flux in erg s^-1 cm^-2 um^-1
    """
    f0 = phot_dict.get(band)[1]
    fm = 10 ** (-0.4 * mag)
    f = f0 * fm
    if sigma is None:
        return f
    else:
        sigma_f0 = phot_dict.get(band)[2]
        sigma_m = sigma
        sigma_fm = sigma_m * 2.3 / 2.5 * fm
        sigma_f = f * np.sqrt((sigma_f0 / f0) ** 2 + (sigma_fm / fm) ** 2)
        if np.isnan(sigma_f):
            sigma_f = sigma_nan * f
        return np.array([f, sigma_f])


def flux_ext(flux, wvl, a_v, r_v, dered=False):
    """
    Extinction correction according to (Cardelli, Clayton, and Mathis 1989, ApJ 345, 245.)
    Calculates the extinction corrected flux according to Cardelli extinction law, interpolated over wvl,
    for the given A_V and R_V.

    :param flux: flux array in arbitrary units
    :param wvl: wavelength array in micron
    :param a_v: Extinction in V band.
    :param r_v: Slope of the extinction in V band.
    :param dered: if True, then deredden observed flux.
    :return: Extinction corrected flux for wavelengths between 0.36-3.45 um.
             Flux-values outside this range are returned unaltered.
    """

    # The Cardelli wavelengths and coefficients. A 0-point was added to help the fit.
    x_Card = np.array([2.78, 2.27, 1.82, 1.43, 1.11, .8, .63, .46, .29, 0.01])
    a_Card = np.array([.9530, .9982, 1., .8686, .6800, .4008, .2693, .1615, .0800, 0])
    b_Card = np.array([1.9090, 1.0495, 0., -.3660, -.6239, -.3679, -.2473, -.1483, -.0734, 0])

    # Fit a 4th degree polynomial to the inverse wavelengths and apply to the wavelength array.
    Al = (a_Card + b_Card / r_v) * a_v
    coeffs_C = np.polyfit(x_Card, Al, 4)
    Al_interp = np.poly1d(coeffs_C)(wvl ** -1)

    # Select the wavelength point with 0 (or minimum) extinction and redden all blue-ward flux-points.
    max_wvl_ext = wvl[np.argmin(np.abs(Al_interp))]

    wvl_indexes = np.where(wvl <= max_wvl_ext)[0]
    ext_mask = np.ones(len(flux), np.bool)
    ext_mask[wvl_indexes] = 0

    Al_final = np.poly1d(coeffs_C)(wvl[wvl_indexes] ** -1)

    # Extincted or dereddened flux.
    if dered:
        dered_fl = np.asarray(flux[wvl_indexes]) * 10 ** (Al_final / 2.5)
        tot_fl = np.concatenate((flux[ext_mask], dered_fl))
    else:
        ext_fl = np.asarray(flux[wvl_indexes]) * 10 ** (-Al_final / 2.5)
        tot_fl = np.concatenate((flux[ext_mask], ext_fl))

    return tot_fl


def get_and_convert_mag_naira(obj, phot_frame=None, phot_dict=None):
    """
    Get and convert all the magnitudes to flux. Gets the photometric fluxes and errors for each object to
    be collected in `photometry_dict`.

    :param obj: (str) object name.
    :param phot_frame: ``panda`` dataframe created from the photometry file.
    :param phot_dict: dictionary with photometric flux zero's and wavelength points.
    :return: None
    """
    if phot_frame is None:
        phot_frame = phot_frame_m17
    if phot_dict is None:
        phot_dict = m17_phot_dict

    flux = []
    wvl = []
    for key in phot_dict.keys():
        mag = phot_frame.loc[obj, key]
        sigma_key = ('e_' + key).strip('um')
        sigma = phot_frame.loc[obj, sigma_key]
        if np.isfinite(mag):
            wvl.append(phot_dict.get(key)[0])
            flux.append(mag_to_flux(key, mag, phot_dict, sigma))
    # Add additional fluxpoints for B275 and B331 'manually'.
    if obj == 'B275':
        wvl.extend([10.6, 20])
        flux.append(jy_to_erg_s_cm2_um(np.array([1.9, 0.32]), 10.6))
        flux.append(jy_to_erg_s_cm2_um(np.array([7.9, 0.5 * 7.9]), 20))
    if obj == 'B331':
        wvl.extend([9.8, 10.53, 11.7, 20.6, 37])
        flux.append(jy_to_erg_s_cm2_um(np.array([1.5, 0.2]), 9.8))
        flux.append(jy_to_erg_s_cm2_um(np.array([1.8, 0.3]), 10.53))
        flux.append(jy_to_erg_s_cm2_um(np.array([2.1, 0.1]), 11.7))
        flux.append(jy_to_erg_s_cm2_um(np.array([6.4, 0.87]), 20.6))
        flux.append(jy_to_erg_s_cm2_um(np.array([21.89, 2.19]), 37))

    # Get the indices of the corresponding points in wsil (for SED fitting).
    ind = [np.argmin(abs(wsil - i)) for i in wvl]
    phot_flux_ind = np.array([wvl, flux, ind], dtype=object)

    return phot_flux_ind


photometry_dict = {obj: get_and_convert_mag_naira(obj) for obj in phot_frame_m17.index}


#  FUNCTIONS


def planck_wvl(wvl_um, t):
    """
    Black body function.
    :param wvl_um: (scalar or array)  wavelength in micron.
    :param t: (scalar or array) temperature in K.
    :return: black body flux at temperature t, as a function of x in erg/s/cm2/micron.
    """
    # Convert micron to cm.
    wvl_cm = wvl_um * 1e-4
    return 2 * h * c ** 2 / wvl_cm ** 5 / (np.exp(h * c / kB / t / wvl_cm) - 1) * 1e-4


def planck_freq(x, temp):
    return 2 * (x / c) ** 2 * (h * x) / (np.exp(h * x / kB / temp) - 1)


def gaussian(x, mu, sigma):
    """
    Normalized Gaussian.

    :param x: variable.
    :param mu: center.
    :param sigma: standard deviation.
    :return: a normalized Gaussian in as a function of x.
    """

    return np.exp(- (((x - mu) / sigma) ** 2) / 2.) / sigma / np.sqrt(2 * np.pi)


def integrand_gauss(theta, v, r, mstar, inc, dv=sampling * min_dv_cm):
    """
    The gaussian function peaking at a velocity corresponding to a certain radius and angle.
    In flat_disk_log_grid this function is integrated over the angle theta to obtain a velocity profile for each radius.

    :param mstar: Stellar mass.
    :param inc: Inclination of the disk
    :param dv: width of the gaussian velocity distribution peaking at the Keplerian velocity (in cm/s).
        The default is twice the resolution of the wavelength array, making it de facto a delta function,
        i.e. assuming no macro turbulence.
    :param theta: Angular coordinate.
    :param v: velocity array in cm/s.
    :param r: Radial coordinate.
    :return: A gaussian peaking at the velocity at (theta,r) in the disk (units: (cm/s)^-1).
    """

    if type(v) is np.ndarray:
        mu = v_kep_cm_s(mstar, r)[:, None] * np.sin(inc) * np.sin(theta)[None, :]
        return gaussian(v[None, :, None], mu[:, None, :], dv)
    else:
        mu = v_kep_cm_s(mstar, r) * np.sin(inc) * np.sin(theta)
        return gaussian(v, mu, dv)


def v_kep_cm_s(m_star, r):
    """
    Keplarian velocity in cm/s.

    :param m_star: in units solar mass.
    :param r: stellar radius in cm.
    :return: Keplarian velocity in cm/s.
    """
    return np.sqrt(G * m_star * M_sun / r)


# Temperature structures
def exp_t(r, t0, r0, t1, a):
    """
    Exponential decay law for the temperature in function of r.
    t1 and a can both be free parameters.

    :param r: radial array in cm.
    :param t0: initial temperature in K at r0.
    :param r0: the reference radius for t0.
    :param t1: the "zero-point" (or minimal) temperature to which the T decays.
    :param a: the decay rate, typically high to get a good fit for our objects.
    :return: unit K
    """
    return t1 + (t0 - t1) * np.exp(a * (r/AU - r0/AU))


def t_simple_power_law(r, ti, ri, p=-0.5):
    """
    # Temperature power law.

    :param r: Radial coordinate in arbitrary units, but same as ri.
    :type r: array
    :param ti: Initial temperature at ri in K.
    :param ri: Initial radius in arbitrary units, but same as r.
    :param p: Power law exponent.
    :return: Return excitation temperature in function of r.
    """
    return ti * (r / ri) ** p


def nco(r, ni, ri, q):
    """
    The CO surface density powerlaw (in cm^-2).

    :param r: radius in cm
    :param ni: Gas (H2) surface density at initial radius (in cm^-2).
    :param ri: Initial radius in cm
    :param q: Power law exponent for gas density
    :return: The CO surface density in cm^-2.
    """
    return ni * (r / ri) ** q / H_CO[species]


def stellar_cont(star, wc):
    """
    Returns the (not-extincted) stellar continuum on the given wavelength array.

    :param star: name of the star.
    :type star: string
    :param wc: wavelength array in micron (of model).
    :type wc: array
    :return: Stellar continuum according to Kurucz model in the relevant wavelength range.
    """

    # Load saved Kurucz models, using the original stellar_cont function.
    # Model flux at Earth in units: ergs/cm^2/s/micron
    mod_wvl_micron, mod_flux_full = np.load(aux_files / ("kurucz_" + star + ".npy"))

    # Select relevant wavelengths.
    IR = np.where(np.logical_and(mod_wvl_micron < max(wc), mod_wvl_micron > min(wc)))
    mod_wvl = mod_wvl_micron[IR]
    mod_flux = mod_flux_full[IR]
    # Interpolate onto wavelength grid.
    stel_cont = np.exp(interp1d(np.log(mod_wvl), np.log(mod_flux), fill_value='extrapolate')(np.log(wc)))

    return stel_cont
