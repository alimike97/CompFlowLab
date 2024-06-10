import numpy as np
import matplotlib.pyplot as plt


full_basis    = np.load(r"C:\Users\mohag\OneDrive - University of Kansas\KU - Aerospace PhD\Research Related\basis_update_sampled_only_comparsion\colliding_shocks\full_basis_update\Adaptive ROM 3000 snapshots Explicit - FD Euler GalerkinGappy POD + Shock basis.npy")
sampled_basis = np.load(r"C:\Users\mohag\OneDrive - University of Kansas\KU - Aerospace PhD\Research Related\basis_update_sampled_only_comparsion\colliding_shocks\sampled_basis_update\Adaptive ROM 3000 snapshots Explicit - FD Euler GalerkinGappy POD + Shock basis.npy")
svd_basis     = np.load(r"C:\Users\mohag\OneDrive - University of Kansas\KU - Aerospace PhD\Research Related\basis_update_sampled_only_comparsion\colliding_shocks\svd_basis_update\Adaptive ROM 3000 snapshots Explicit - FD Euler GalerkinGappy POD + Shock basis.npy")


fig,ax = plt.subplots(3,3,sharex=True)

ax = ax.flatten()

iter_list=np.array([11,1000,2000,2999])

for iter in iter_list:

    for mode in range(9):

        ax[mode].cla()

        ax[mode].plot(full_basis[:,mode,iter],color='tab:blue'  ,label='full')
        ax[mode].plot(sampled_basis[:,mode,iter],color='tab:red',label='sampled')
        ax[mode].plot(svd_basis[:,mode,iter],color='black',label='svd')

        ax[mode].relim()
        ax[mode].autoscale()

        ax[mode].legend()

        ax[mode].set_title('mode = ' + str(mode))


    print(iter)
    plt.pause(1)

plt.show(block=True)
