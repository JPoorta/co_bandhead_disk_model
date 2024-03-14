import grids.list_of_grids as gr_list
import grids.run_grids as run_gr
import plotting_routines as pltr


def run():

    # run_gr.run_quick_test_grid(gr_list.common_test_grid(), **{"quick_plots": True})
    pltr.plot_cum_flux_grid(gr_list.common_test_grid())

    return


if __name__ == "__main__":
    run()
