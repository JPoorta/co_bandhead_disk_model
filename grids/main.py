"""
Module from which to calculate, save, plot grids of models.
"""

import grids.list_of_grids as gr_list
import grids.run_grids as run_gr


def run():

    full_calc_13co(gr_list.fid_model_B268(), iso_ratios=[89])
    # run_gr.run_quick_test_grid(grid=gr_list.fid_model_B275_no_dust(), quick_plots=True, plot_B275_checks=False)
    # run_gr.run_grid_for_intro()
    return


def full_calc_13co(grid, iso_ratios=None):
    """
    Run the full cycle to calculate and save 13CO included models for a given grid and given list of abundance ratios.

    :param grid:
    :param iso_ratios:
    :return:
    """
    # First run the grid without 13CO (is now required, but this should be fixed).
    run_gr.save_run_variation_around_one_model(grid)
    run_gr.save_grid_including_13co(grid, iso_ratios)
    run_gr.add_saved_isotope_grid(grid, iso_ratios)

    return


if __name__ == "__main__":
    run()
