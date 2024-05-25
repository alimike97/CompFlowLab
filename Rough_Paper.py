import numpy as np
import matplotlib.pyplot as plt


# rom = np.load(r"C:\GIT_Fork\ROMify\examples\shock_tube\Adaptive ROM 15000 snapshots Explicit - FD Euler GalerkinGappy POD cons.npy")/0.002
fom = np.load(r"C:\GIT_Fork\ROMify\examples\sedov_blast_wave\FOM 50000 snapshots Explicit - FD Euler prim.npy")

# rusanov     = np.load(r"C:\GIT_Fork\ROMify\examples\shock_tube\RusanovFOM 3000 snapshots Explicit - FD Euler cons.npy")/0.002
# roe = np.load(r"C:\GIT_Fork\ROMify\examples\shock_tube\RoeFOM 3000 snapshots Explicit - FD Euler cons.npy")/0.002

fig,ax = plt.subplots(3,1)

x=range(500)

for iter in range(0,50000,1000):

    # rom_snapshot = rom[0,:,iter].ravel()
    rho = fom[0,:,iter]
    vx = fom[1,:,iter]
    press = fom[2,:,iter]

    ax[0].cla()
    ax[1].cla()
    ax[2].cla()

    ax[0].plot(x,rho  ,color='tab:blue')
    ax[1].plot(x,vx   ,color='tab:red')
    ax[2].plot(x,press,color='tab:green')

    ax[0].set_ylabel('Density')
    ax[1].set_ylabel('Velocity')
    ax[2].set_ylabel('Pressure')

    ax[0].set_xlabel('x')
    ax[1].set_xlabel('x')
    ax[2].set_xlabel('x')

    ax[0].relim()
    ax[0].autoscale()

    ax[1].relim()
    ax[1].autoscale()

    ax[2].relim()
    ax[2].autoscale()



    # ax.legend()
    # ax.set_title(iter)
    print(iter)
    plt.pause(0.01)

plt.show()