import numpy as np
import matplotlib.pyplot as plt

import model.config as cfg


def plot_cum_flux(gp):
    """

    :param gp: (Gridpoint object) defines the model, including parameters for which
    :return:
    """

    full_path = gp.path_co() + gp.filename_co()
    cum_flux_extension = "_dF_CO_lines"

    try:
        dF_disk, r_disk_AU = np.load(full_path + cum_flux_extension + ".npy")
    except FileNotFoundError:
        wvl, intensity_cont_subtr, r_disk = read_and_combine_arrays(full_path)
        dF_disk, r_disk_AU = calc_totals(wvl, intensity_cont_subtr, r_disk)
        np.save(full_path + cum_flux_extension, [dF_disk, r_disk_AU])

    plt.figure(20)

    p = plt.plot(r_disk_AU, dF_disk[0, :, 2], label=str(gp.p) + "; fund")[0]
    plt.plot(r_disk_AU, dF_disk[0, :, 1], '--', c=p.get_color(), label=str(gp.p) + "; 1st ot")
    plt.plot(r_disk_AU, dF_disk[0, :, 0], ':', c=p.get_color(), label=str(gp.p) + "; 2nd ot")

    return


def read_and_combine_arrays(full_path):

    wvl_CO, intensity_conv_CO, r_CO = np.load(full_path + "_CO.npy")
    wvl_mix, intensity_conv_mix, r_mix = np.load(full_path + "_mix.npy")
    wvl_dust, intensity_conv_dust, r_dust = np.load(full_path + "_dust.npy")

    if np.array_equal(r_dust, r_mix):
        r_disk = np.concatenate((r_CO, r_mix))
        intensity_tot = np.concatenate((intensity_conv_CO, intensity_conv_mix-intensity_conv_dust), axis=1)
    else:
        print("Error: radial arrays of dust disk and gas and dust mixed are not the same.")
        return

    if np.array_equal(wvl_CO, wvl_mix):
        wvl = wvl_CO
    else:
        print("Error: wavelength arrays CO and mix not consistent.")
        return

    # total_flux_CO = calc_total_flux(intensity_conv_CO, r_CO)
    # total_flux_mix = calc_total_flux(intensity_conv_mix, r_mix)
    # total_flux_dust = calc_total_flux(intensity_conv_dust, r_dust)
    # total_flux_cont_subtr1 = total_flux_CO + total_flux_mix - total_flux_dust
    # total_flux_cont_subtr2 = calc_total_flux(intensity_tot, r_disk) - total_flux_dust

    # # plt.figure(1)
    # # plt.plot(wvl, total_flux_cont_subtr1[0,:]/total_flux_cont_subtr2[0,:])
    # # #
    # plt.figure(2)
    # # plt.plot(wvl, total_flux_cont_subtr1[0, :], label = "as in model")
    # plt.plot(wvl, total_flux_cont_subtr2[0, :], label = "total cont subtr flux")
    # plt.plot(wvl, total_flux_CO[0, :], label = "CO")
    # plt.plot(wvl, total_flux_mix[0, :], label="mix")
    # plt.plot(wvl, total_flux_dust[0, :], label= "dust")
    #
    # plt.legend()
    #
    # plt.show()
    # #
    # plt.figure(3)
    # plt.scatter(r_disk/cfg.AU, np.arange(len(r_disk)))
    # plt.scatter(r_CO/cfg.AU, np.arange(len(r_CO)))
    # plt.scatter(r_mix/cfg.AU, np.arange(len(r_mix)))

    return wvl, intensity_tot, r_disk


def calc_totals(wvl, intensity, r_disk):
    total_flux = calc_total_flux(intensity, r_disk)
    dF_disk = calc_cumulative_flux(wvl, intensity, r_disk, total_flux)

    return dF_disk, r_disk / cfg.AU


def calc_total_flux(intensity_conv, r):
    """
    Calculate the total flux for the two regions in the model: the CO gas only region, and where the gas and dust are
    mixed. Flux is in arbitrary units, used to normalize the cumulative fluxes.

    :param intensity_conv:
    :param r:
    :return: 2D array, with first dimension dv0 values and second wvl
    """
    flux = np.array([np.trapz(intensity_conv[j, ...].T * r, x=r, axis=1)
                     for j in range(intensity_conv.shape[0])])
    return flux


def calc_cumulative_flux(wvl, intensity_conv, r, total_flux):
    """
    Calculate the cumulative flux as a function of disk radius.

    :param wvl: wavelength array in micron
    :param intensity_conv: (3D array) intensities, shape (len
    :param r:
    :param total_flux:
    :return:
    """

    second_ot = np.array(np.where(wvl < 1.85))[0]
    first_ot = np.where(np.logical_and(wvl > 2.25, wvl < 3.25))[0]
    fundamental = np.where(wvl > 4.2)[0]
    dF = np.zeros((intensity_conv.shape[0], intensity_conv.shape[1], 3))

    for n, el in enumerate([second_ot, first_ot, fundamental]):

        total = np.trapz(total_flux[:, el], x=wvl[el])
        for k in range(len(r)):
            dF[:, k, n] = np.trapz(
                np.trapz(
                    intensity_conv[:, :k, el].T * r[None, :k, None], x=r[:k], axis=1
                ).T / total, x=wvl[None, el]
            )

    return dF
