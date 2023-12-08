import numpy as np
import model.config as cfg


class Gridpoint:
    """
    Object with methods to apply to gridpoints in the parameter space of a model grid.
    """

    def __init__(self, sub_folder="",
                 star=None,
                 results_folder=cfg.results_folder,
                 dv0=None,
                 ri=None,
                 q=None,
                 ni=None,
                 p=None,
                 ti=None,
                 inc_array=np.array([10, 20, 30, 40, 50, 60, 70, 80]),
                 wvl_array="wvl_re",
                 ):

        # define default attributes
        self.sub_folder = sub_folder
        self.star = star
        self.results_folder = results_folder
        self.dv0 = dv0
        self.ri = ri
        self.q = q
        self.ni = ni
        self.p = p
        self.ti = ti
        self.inc_array = inc_array
        self.wvl_array = wvl_array

    def filename_co(self):
        """
        Needs ri (in AU),ti,p,ni and q.
        :return: The concatenated strings of the gridpoint defining parameters.
        """
        try:
            return cfg.filename_co_grid_point(self.ti, self.p, self.ni, self.q, self.ri, self.dv0)

        except TypeError:
            print("WARNING: Variables ri (in AU),ti,p,ni,q of the gridpoint object are not (all) defined.")

            return

        # define combined attributes

    def path_co(self):

        return str(self.results_folder / self.star / self.sub_folder) + "/"
