import numpy as np
import matplotlib.pyplot as plt


rom = np.load(r"C:\GIT_Fork\ROMify\examples\shock_tube\Adaptive ROM 15000 snapshots Explicit - FD Euler GalerkinGappy POD cons.npy")/0.002
fom = np.load(r"C:\GIT_Fork\ROMify\examples\shock_tube\FOM 15000 snapshots Explicit - FD Euler cons.npy")/0.002

# rusanov     = np.load(r"C:\GIT_Fork\ROMify\examples\shock_tube\RusanovFOM 3000 snapshots Explicit - FD Euler cons.npy")/0.002
# roe = np.load(r"C:\GIT_Fork\ROMify\examples\shock_tube\RoeFOM 3000 snapshots Explicit - FD Euler cons.npy")/0.002

fig,ax = plt.subplots(1,1)

for iter in range(0,15000,50):

    rom_snapshot = rom[0,:,iter].ravel()
    fom_snapghot = fom[0,:,iter].ravel()

    ax.cla()
    ax.plot(rom_snapshot,label='rom')
    ax.plot(fom_snapghot,label='fom')
    ax.legend()
    ax.set_title(iter)
    plt.pause(0.01)

plt.show()