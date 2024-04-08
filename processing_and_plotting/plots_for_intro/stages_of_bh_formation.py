import numpy as np
import matplotlib.pyplot as plt

import model.config as cfg

# Folders
base_folder = cfg.results_folder / "B275/intro_plots"
ring_folder = base_folder / "ring"
disk_folder = base_folder / "disk"

# File names
tau = "tau_co.npy"
intesity = "I_co.npy"
flux = "flux.npy"

# Ring arrays
wvl, intensity, S = np.load(ring_folder/intesity)
wave, tau_CO, t_gas = np.load(ring_folder/tau)
flux_ring = np.load(ring_folder/flux)[0]
flux_disk = np.load(disk_folder/flux)[0]

def create_figure(plot_count):

    ratios = np.ones(plot_count)
    width = 8
    height = width / 1.54  # ~A5 proportions
    gridspec = dict(height_ratios=ratios, hspace=0.0,top=0.985,
                                                    bottom=0.09,
                                                    left=0.055,
                                                    right=0.97,
                                                    )
    f, ax = plt.subplots(plot_count, facecolor='w', figsize=(width,height), sharey=False, sharex=True,
                     gridspec_kw=gridspec)

    return f, ax

fig1, axes1 = create_figure(plot_count=4)
fig2, axes2 = create_figure(plot_count=4)

for ax in [axes1, axes2]:
    ax[0].plot(wvl, tau_CO[0][0], label = r"$\tau_{\rm CO}$")
    ax[1].plot(wvl, intensity[0][0]/np.max(intensity[0][0]), label = r"$I_{\lambda}$")
    ax[1].plot(wvl, S[0]/np.max(intensity[0][0]), label = r"$\rm S_{CO}$")
    ax[2].plot(wvl, flux_ring/np.max(flux_ring), label = r"$(F_{\lambda})_{\rm ring}$")
    ax[-1].set_xlabel(r"$\lambda \rm (\mu m)$")
    ax[-1].plot(wvl, flux_disk/np.max(flux_disk), label = r"$(F_{\lambda})_{\rm disk}$")

zoom = 2.308
for ax in axes1:
    ax.set_xlim(2.29, 2.6)
    ax.axvline(zoom, color='r', linewidth=0.5)
    ax.legend(loc="upper right")
for ax in axes2:
    ax.set_xlim(2.2926, zoom)
    ax.legend(loc="upper right")


fig1.savefig(cfg.plot_folder/"bandhead_formation.pdf")
fig2.savefig(cfg.plot_folder/"bandhead_formation_zoom.pdf")


plt.show()
