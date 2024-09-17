import sys
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt


import numpy as np


# res_hist        = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\FOM 3500 snapshots Explicit - FD Euler res_hist.npy")
# basis           = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\basis_save.npy")
# S_indx_solver   = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\S_indx_solver_save_all.npy")


# res_fom = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\res_fom.npy")
# res_rom = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\res_rom.npy")


fom = np.load(r"C:\GIT_Fork\ROMify\examples\transient_flame\FOM 10000 snapshots Explicit - FD Euler prim.npy")
arom = np.load(r"C:\GIT_Fork\ROMify\examples\transient_flame\Adaptive ROM 100000 snapshots Explicit - FD Euler GalerkinQDEIM prim.npy")
samples = np.load(r"C:\GIT_Fork\ROMify\examples\transient_flame\Adaptive ROM 100000 snapshots Explicit - FD Euler GalerkinQDEIM samples_user.npy")
# arom = np.load(r"C:\GIT_Fork\ROMify\examples\classic_shock_tube\Adaptive ROM 3000 snapshots Explicit - FD Euler GalerkinQDEIM prim.npy")

# basis = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\basis.npy")
# norm = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\norm.npy")
# denorm = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\denorm.npy")
# qref = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\q_ref.npy")

# T = fom[3,:,6000].ravel()
# P = fom[2,:,6000].ravel()
# u = fom[1,:,6000].ravel()

# slice_source = fom[:,201,6000]

# fom[:,0:201,6000] = slice_source[:,np.newaxis]

# np.save(r"C:\GIT_Fork\ROMify\examples\transient_flame\steady_IC_trans.npy",fom[:,:,6000])

# T = fom[3,:,6000].ravel()
# P = fom[2,:,6000].ravel()
# u = fom[1,:,6000].ravel()

fig , ax = plt.subplots(3,1)

# fig.suptitle('Adaptive ROM dt=5e-5 vs. FOM dt=5e-4')

# ax[0].plot(T,label='T')
# ax[1].plot(P,label='P')
# ax[2].plot(u,label='u')
# # ax.plot(q_replica,linestyle='--',label='ROM no HR')
# ax[0].legend()
# ax[1].legend()
# ax[2].legend()

x=np.linspace(0,0.03,1000)

# plt.show()
counter = 0

for iter in range(0,100000,1000):

    s_indx = np.nonzero(samples[:,iter])[0]

    ax[0].cla()
    ax[1].cla()
    ax[2].cla()

    rho = fom[0,:,counter].ravel()
    u   = fom[1,:,counter].ravel()
    T   = fom[3,:,counter].ravel()

    a_rho = arom[0,:,iter].ravel()
    a_u   = arom[1,:,iter].ravel()
    a_T   = arom[3,:,iter].ravel()

    ax[0].plot(x,rho      ,color='black',label='FOM-rho')
    ax[1].plot(x,u        ,color='blue',label='FOM-u')
    ax[2].plot(x,T        ,color='red',label='FOM-T')

    ax[0].plot(x,rho      ,color='black',linestyle='--',label='AROM-rho')
    ax[1].plot(x,u        ,color='blue',linestyle='--',label='AROM-u')
    ax[2].plot(x,a_T      ,color='red',linestyle='--',label='AROM-T')

    ax[0].scatter(x[s_indx] , rho[s_indx]      ,color='black')
    ax[1].scatter(x[s_indx] , u[s_indx]        ,color='black')
    ax[2].scatter(x[s_indx] , a_T[s_indx]      ,color='black')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    ax[0].set_xlabel('x[m]')
    ax[1].set_xlabel('x[m]')
    ax[2].set_xlabel('x[m]')

    ax[0].set_ylabel('rho[kg/m3]')
    ax[1].set_ylabel('u[m/s]')
    ax[2].set_ylabel('T[K]')

    counter = counter + 100

    plt.pause(1e-6)
    

#     # q_replica = qref+(denorm*(basis@(basis.T@(norm*(q-qref)))))



#     print(iter)

# basis = basis[:,0:11]

# for iter in range(0,3500,100):

#     ax.cla()

#     f = res_hist[:,iter]

#     f = np.reshape(f,(12,1004))

#     f = f[:,2:-2].ravel()

#     f_replica = basis @ np.linalg.pinv(basis[S_indx_solver,:]) @ f[S_indx_solver]

#     ax.plot(f,label='FOM res')
#     ax.plot(f_replica,linestyle='--',label='Hyper-Reduced res')

#     ax.legend()

#     plt.pause(1e-6)



#     print(iter)




