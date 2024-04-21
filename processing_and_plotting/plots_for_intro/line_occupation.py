import numpy as np
import matplotlib.pyplot as plt

import model.config as cfg

# Folders
base_folder = cfg.results_folder / "B275/intro_plots/line_occupation"

# File names
alpha = "alpha_co.npy"

# Ring arrays
wave, alpha_co, t_gas = np.load(base_folder/alpha)

def create_figure(plot_count):

    ratios = np.ones(plot_count, )
    width = 8
    height = width * 1.34  # ~A5 proportions
    gridspec = dict(height_ratios=ratios, hspace=0, wspace=0, top=0.975, bottom=0.055,left=0.05, right=0.99,)
    f, ax = plt.subplots(plot_count,3, facecolor='w', figsize=(width,height), sharey=True, sharex=False,
                     gridspec_kw=gridspec)

    # remove x labels
    for i in range(plot_count - 1):
        for j in range(3):
            ax[i, j].set_xticks([])

    # Series titles
    title_font_size = 9.5
    ax[0, 0].set_title(r"Second overtone ($\Delta v=3$)", fontsize=title_font_size)
    ax[0, 1].set_title(r"First overtone ($\Delta v=2$)", fontsize=title_font_size)
    ax[0, 2].set_title(r"Fundamental ($\Delta v=1$)", fontsize=title_font_size)

    return f, ax


def plot_on_divided_axes(x, y, fig_ax, title=None, **kwargs):
    """
    Plot the given x,y on a plot showing only the wavelength regions of interest, that is, second, first overtone and
    fundamental. Also plots the legend.

    :param x: (array) x data, should be wavelength in micron.
    :param y: (array) y data, should be normalized flux.
    :param fig_ax: (fig, ax) objects. If not provided will be created using 'create_3_in_1_figure'.
    :param title: (str) title of the figure.
    :param kwargs: arguments to pass on to plot, color, zorder etc.
    :return: The fig and ax objects for further plotting.
    """

    fig, ax = fig_ax

    f_mask = return_mask(x, min=4.28, max=5.3)
    ot1_mask = return_mask(x, min=2.285, max=2.6)
    ot2_mask = return_mask(x, min=1.55, max=1.70)

    mask_dict = {0:ot2_mask, 1:ot1_mask, 2:f_mask}

    fig.suptitle(title)
    for i, axi in enumerate(ax):
        axi.plot(x, y/np.max(y[mask_dict[i]]), **kwargs) #
        axi.set_xlabel(r"$\lambda (\mu$m)")


    ax[0].set_xlim(1.55, 1.70)  # 1.75
    ax[0].set_xticklabels(['1.550', '1.575', '1.600', '1.625','1.650','1.675'])
    ax[1].set_xlim(2.275, 2.6)  # 2.9
    ax[2].set_xlim(4.25, 5.3)
    ax[1].legend()
    ax[0].set_ylim(0, 1.1)

    return fig, ax

def return_mask(array, min, max):
    return np.where(np.logical_and(array>min, array<max))

fig, axes = create_figure(5)

for i, t in enumerate(t_gas):
    plot_on_divided_axes(wave, alpha_co[0][i], (fig, axes[i]), **dict(label=r"$T_{\rm ex}=$"+str(int(t+1))+" K",
                                                                      lw=0.2, c= 'k', alpha=0.3))
    plot_on_divided_axes(wave, alpha_co[0][i], (fig, axes[i]), **dict(lw=0.2, c='k', alpha=0.3))
fig.savefig(cfg.plot_folder/"line_occupation.pdf")

plt.show()