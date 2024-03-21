import matplotlib.pyplot as plt
import numpy as np

import processing_and_plotting.plotting_routines as pltr
import model.flat_disk_log_grid as fld
import model.example as ex
import model.config as cfg
import grids.list_of_grids as list_of_grids
from processing_and_plotting.main_figure_p4 import model_fluxes_for_test_param, fiducial_model


def run_quick_test_grid(grid, quick_plots=False, plot_B275_checks=False, save=None, save_t_plot=None):
    """
    Run the grid not as a grid, but going through each test value separately as set in 'model.example'.

    :param grid: (list of three dicts) grid as defined in 'list_of_grids'. In all_params "convolve" must be set to True,
    and "save" to None. Pretty much any grid with the default grid params as defined in config should work.
    :param quick_plots:
    :param plot_B275_checks:
    :param save:
    :param save_t_plot:
    :return:
    """

    grid_params, all_params, test_param_dict = grid

    fig, ax = pltr.create_3_in_1_figure(num=3)
    for test_param, test_param_array in test_param_dict.items():
        gp, star, wvl, continuum_flux = ex.run_test(test_param, test_param_array, grid_params.copy(), all_params.copy(),
                                                    (fig, ax), quick_plots)

    ex.finish_plots(gp, star, wvl, continuum_flux, (fig, ax), quick_plots, plot_B275_checks)

    if save is not None:
        fig.savefig(cfg.plot_folder / (save + ".pdf"))
    if save_t_plot is not None:
        plt.figure(5)
        plt.savefig(cfg.plot_folder / (save_t_plot + ".pdf"))

    plt.show()

    return


# TODO: In this version the fiducial model is calculated several times and saved under the same name.
#  only because inc_deg is last in the test arrays, the variation in the inclination is recorded.

def save_run_variation_around_one_model(grid=None, ):
    """
    Save set of models by varying given parameters (specified test_param_dict) around a fiducial model.

    :param grid:
    :return:
    """

    if grid is None:
        grid = list_of_grids.grid_for_main_figure_p4()

    grid_params, all_params, test_param_dict = grid

    for test_param, test_param_array in test_param_dict.items():
        grid_params_use = grid_params.copy()
        all_params_use = all_params.copy()
        if test_param in grid_params.keys():
            grid_params_use[test_param] = test_param_array
        elif test_param in all_params.keys():
            all_params_use[test_param] = test_param_array

        model_grid = ex.make_grid(**grid_params_use)
        fld.run_grid_log_r(model_grid, **all_params_use)

    return


def save_grid_including_13co(grid, iso_ratios=None):
    """
    For al the isotopologues defined through the statistical weight dictionary in config, and for all ratios,
     run the full specified grid, saving it in a folder named after the original grid, the species and the ratio.
     Note that this creates len(species)*len(iso_ratios) folders with entire grids of models.

    :param grid: the grid to calculate.
    :param iso_ratios: (list of int) number abundance ratio with respect to 12C16O.
    :return:
    """

    if iso_ratios is None:
        iso_ratios = list_of_grids.default_iso_ratios()
    for ratio in iso_ratios:
        for species in cfg.g_i_dict.keys():
            grid_use = list_of_grids.grid_including_13co(species, ratio, grid)
            save_run_variation_around_one_model(grid_use)

    return


def get_fid_model_dict(species_list, grid, iso_ratios):
    """
    Create a dictionary containing per species a tuple, consisting of:
     - the Gridpoint object for the fiducial model
     - the fiducial model flux
     - the fiducial model wavelength
     - the fiducial model instrument profile
     The fiducial model here is the model in grid with default parameters (this means in grid this should be one model,
     and the variation of it is passed through test parameters), and the first iso ratio in the list.

    :param species_list: (list of str) isotopologue names (so far only 12C16O and 13C16O)
    :param grid: tuple specifying grid
    :param iso_ratios: (list of int)
    :return: The dictionary described above and the test_paramter dictionary of grid (which should not change with
             isotopologue).
    """

    fid_model_dict = {}
    for i, species in enumerate(species_list):
        if i == 0:
            grid_params, all_params, test_param_dict = grid
        else:
            grid_params, all_params, test_param_dict = list_of_grids.grid_including_13co(species, iso_ratios[0], grid)
        gp_fid_model, fid_mod_fl, wvl, ip_dx = fiducial_model(grid_params, all_params)
        fid_model_dict[species] = [gp_fid_model, fid_mod_fl, wvl, ip_dx]

    return fid_model_dict, test_param_dict


def add_saved_isotope_grid(grid, iso_ratios=None):
    """
    For all the values in the test parameters of a grid find, combine, and save the fluxes of saved models per
    isotopologue and its abundance (passed through iso_ratios).

    :param grid:
    :param iso_ratios:
    :return:
    """
    if iso_ratios is None:
        iso_ratios = list_of_grids.default_iso_ratios()

    species_list = list(cfg.g_i_dict.keys())

    fid_model_dict, test_param_dict = get_fid_model_dict(species_list, grid, iso_ratios)

    for ratio in iso_ratios:
        for test_param, test_array in test_param_dict.items():
            if test_param != "inc_deg":
                iso_flux_dict = {}
                for species in species_list:
                    grid_use = list_of_grids.grid_including_13co(species, ratio, grid)
                    grid_params, all_params, test_param_dict = grid_use
                    model_fluxes, model_names = model_fluxes_for_test_param(test_param, test_array, grid_params,
                                                                            all_params,
                                                                            ip_dx=fid_model_dict[species][3])

                    iso_flux_dict[species] = (model_fluxes, model_names)
                add_and_save(iso_flux_dict, test_array, fid_model_dict, species_list, ratio)

    return


def add_and_save(iso_flux_dict, test_array, fid_model_dict, species_list, ratio, name=None):
    """
    For a set of test parameters, get the model fluxes for all the isotopologues in species_list, interpolate them
    onto the common wavelength array of 12CO, add them to the 12CO flux, and save the result to the folder where the
    original grid is stored.

    :param iso_flux_dict: dictionary containing per species a tuple, consisting of
                          - a dictionary with the model fluxes stored per parameter in 'test_array'
                          - a dictionary with the model names stored per parameter in 'test_array'
    :param test_array: (list) contains the keys to above two dictionaries.
    :param fid_model_dict: dictionary containing per species a tuple, consisting of
                          - the Gridpoint object for the fiducial model
                          - the fiducial model flux
                          - the fiducial model wavelength
                          - the fiducial model instrument profile
    :param species_list: (list of strings) list of isotopologues, the first should be always "12C16O".
    :param ratio: (int) the ratio N12CO over N(isotopologue), so far only 13C16O.
    :param name: (str) last part of filename to which the model should be saved.
    :return:
    """

    if name is None:
        name = species_list[1] + "_" + str(ratio)

    base_species = species_list[0]  # Always 12C16O in our case.
    base_wvl = fid_model_dict[base_species][2]  # get the common wavelength array
    folder = fid_model_dict[base_species][0].path_grid_folder()  # Get the folder to save the model to.
    base_fluxes, model_names = iso_flux_dict[base_species]  # Unpack the two dictionaries for 12C16O.
    cut_conv_artefact = 450  # Cut the convolution artefact before interpolating.

    wvls = [fid_model_dict[sp][2] for sp in species_list[1:]]

    for value in test_array:
        base_flux = base_fluxes[value]
        model_name = model_names[value] + "_" + name
        for species, wvl in zip(species_list[1:], wvls):  # So far only one species, i.e. 13C16O, is implemented.
            flux = iso_flux_dict[species][0][value]
            intp_flux = np.interp(base_wvl, wvl[cut_conv_artefact:], flux[cut_conv_artefact:])
            base_flux += intp_flux - 1

        np.save(folder + model_name, base_flux)

    return
