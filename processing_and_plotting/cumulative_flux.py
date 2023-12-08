import numpy as np

import model.config as cfg







def calc_cumulative_flux(wvl, intensity_conv, r):
    """

    :param wvl:
    :param intensity_conv:
    :param r:
    :return:
    """

    second_ot = np.array(np.where(wvl < 1.85))[0]
    first_ot = np.where(np.logical_and(wvl > 2.25, wvl < 3.25))[0]
    fundamental = np.where(wvl > 4.2)[0]
    dF = np.zeros((intensity_conv.shape[0], intensity_conv.shape[1], 3))
    flux = np.array([np.trapz(intensity_conv[j, ...].T * r, x=r, axis=1)
                     for j in range(intensity_conv.shape[0])])

    for n, el in enumerate([second_ot, first_ot, fundamental]):

        total = np.trapz(flux[:, el], x=wvl[el])
        for k in range(len(r)):
            dF[:, k, n] = np.trapz(
                np.trapz(
                    intensity_conv[:, :k, el].T * r[None, :k, None], x=r[:k], axis=1
                ).T / total, x=wvl[None, el]
            )

    # plt.figure(20)
    # plt.plot(R_CO_only/cfg.AU, dF_CO[0, :, 2], label=str(p)+"; CO only, fund")
    # plt.plot(R_CO_only/cfg.AU, dF_CO[0, :, 1], '--', label=str(p)+"; CO only, 1st ot")
    # plt.legend()

    # plt.figure(20)
    # plt.plot(R_dust[mix]/cfg.AU, dF_mix[0, :, 2],  label=str(p) + "; mix, fund")
    # plt.plot(R_dust[mix]/cfg.AU, dF_mix[0, :, 1], '--', label=str(p) + "; mix, 1st ot")
    # plt.legend()

    return

