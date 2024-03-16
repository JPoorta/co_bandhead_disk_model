import grids.run_grids as run_gr
import grids.list_of_grids as gr_list


def make_figs():
    test_dict = {"t1": [800],
                 "p": [-2]}
    fid_model = "fiducial_model"
    fid_model_t_strct = "t_strct"
    run_gr.run_quick_test_grid(gr_list.common_test_grid_original_t(test_dict), save=fid_model,
                               save_t_plot=fid_model_t_strct)
    return


if __name__ == "__main__":
    make_figs()
