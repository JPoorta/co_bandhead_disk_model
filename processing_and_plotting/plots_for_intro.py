import numpy as np
import matplotlib.pyplot as plt

import model.config as cfg

rot_series_dict = {"P": dict(m=0, plot_args={"c": 'b', "label": "P"}),
                   "Q": dict(m=1, plot_args={"c": 'k', "label": "Q"}),
                   "R": dict(m=2, plot_args={"c": 'g', "label": "R"})}

vib_transitions = [(1, 0), (2, 1), (2, 0), (3, 1), (3, 0), (4, 1)]


def run():
    vu, vl = vib_transitions[0]
    plot_multiple_ro_vib_e_levels(vib_transitions, rot_series_dict)
    plt.legend()

    plt.show()

    return


def plot_multiple_ro_vib_e_levels(vib_trans_list, plot_spec_dict):

    for vu, vl in vib_trans_list:
        plot_ro_vib_energy_levels(vu, vl, plot_spec_dict)

    return


def plot_ro_vib_energy_levels(vu, vl, plot_spec_dict, species="12C16O"):
    """

    :param vu:
    :param vl:
    :param plot_spec_dict:
    :return:
    """
    mask, ind = create_mask_for_vib_level(vu, vl, species)
    vl, jlower, vh, jupper, a_einstein, elow, freq_trans = apply_mask_to_transitions(mask, ind, species)
    plot_rot_energy_levels_pqr(freq_trans, jlower, jupper, plot_spec_dict)
    return


def plot_rot_energy_levels_pqr(freq_trans, jlower, jupper, plot_spec_dict):
    """
    Plot the rotational energy levels, marking the P, Q, and R branches separatly.

    :param freq_trans:
    :param jlower:
    :param jupper:
    :param plot_spec_dict:
    :return:
    """

    pqr_mask = rotational_series_mask(freq_trans, jlower, jupper)

    for key, value in plot_spec_dict.items():
        m = value["m"]
        plot_args = value["plot_args"]
        plot_args["s"] = 2
        plt.scatter(freq_trans[pqr_mask[m]], jlower[pqr_mask[m]], **plot_args)

    return


def create_mask_for_vib_level(vu, vl, species):
    """
    Create sorted mask to apply to transitions

    :param vu:
    :param vl:
    :param species: (str) "12C16O" or "13C16O"
    :return:
    """
    vlow, jl_all, vhigh, jh, A, El_all, freq_tr = cfg.co_data(species)
    mask = np.zeros(len(freq_tr), dtype=bool)
    mask += np.logical_and(vhigh == vu, vlow == vl)
    ind = np.argsort((freq_tr[mask]))

    return mask, ind


def apply_mask_to_transitions(mask, ind, species):
    """
    From the transition arrays in config make a selection using mask, sort it according the frequency using ind.

    :param mask:
    :param ind:
    :param species: "12C16O" or "13C16O"
    :return:
    """
    vlow, jl_all, vhigh, jh, A, El_all, freq_tr = cfg.co_data(species)
    vl, jlower, vh, jupper, a_einstein, elow, freq_trans = \
        [(a[mask][ind]) for a in [vlow, jl_all, vhigh, jh, A, El_all, freq_tr]]
    return vl, jlower, vh, jupper, a_einstein, elow, freq_trans


def rotational_series_mask(freq_tr, jl, ju):
    """
    Returns three masks (in one 2D array) of length freq_tr that mark the P, Q, and R rotational series respectively.

    :param freq_tr:
    :param jl:
    :param ju:
    :return:
    """

    pqr_mask = np.empty((3, len(freq_tr)), dtype=bool)
    np.greater(jl, ju, out=pqr_mask[0])
    np.equal(jl, ju, out=pqr_mask[1])
    np.greater(ju, jl, out=pqr_mask[2])

    return pqr_mask


if __name__ == "__main__":
    run()
