import grids.list_of_grids as gr_list
import grids.run_grids as run_gr
import plotting_routines as pltr


def run():

    # run_gr.save_run_variation_around_one_model(grid=gr_list.grid_for_main_figure_p4())
    # run_gr.save_grid_including_13co(gr_list.grid_for_main_figure_p4())
    # run_gr.add_saved_isotope_grid(gr_list.grid_for_main_figure_p4())
    # pltr.plot_cum_flux_grid(gr_list.grid_for_main_figure_p4())
    run_gr.run_grid_for_intro()

    return


if __name__ == "__main__":
    run()
