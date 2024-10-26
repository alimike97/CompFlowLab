import sys
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt


import numpy as np


# res_hist        = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\FOM 3500 snapshots Explicit - FD Euler res_hist.npy")
# basis           = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\basis_save.npy")
# S_indx_solver   = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\S_indx_solver_save_all.npy")

# samples = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame_with_perturbation\Adaptive ROM 35000 snapshots Explicit - FD Euler GalerkinQDEIM samples_user.npy")

# res_fom = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\res_fom.npy")
# res_rom = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\res_rom.npy")

# fom_super_small = np.load(r"C:\GIT_Fork\ROMify\examples\classic_shock_tube\FOM 30000 snapshots Explicit - FD Euler prim_dt_1en6.npy")


fom_roe1           = np.load(r"C:\Users\mohag\Desktop\iteration11500_prim.npy")
# fom_roe2           = np.load(r"C:\GIT_Fork\ROMify\examples\classic_shock_tube\second_order_roe_FOM 1500 snapshots Explicit - SSPRK2 prim.npy")
# arom_original = np.load(r"C:\Users\mohag\OneDrive - University of Kansas\KU - Aerospace PhD\Research Related\error_analysis_of_ROM\shock_tube_initial_arom\Adaptive ROM 3000 snapshots Explicit - FD Euler GalerkinQDEIM cons.npy")
# arom_improved = np.load(r"C:\Users\mohag\OneDrive - University of Kansas\KU - Aerospace PhD\Research Related\error_analysis_of_ROM\shock_tube_improved_arom\Adaptive ROM 3000 snapshots Explicit - FD Euler GalerkinQDEIM cons.npy")

# fom = np.dstack((fom1,fom2))

# fom_res = np.load(r"C:\Users\mohag\Desktop\fom_res.npy")
# rom_res = np.load(r"C:\Users\mohag\Desktop\rom_res.npy")

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(fom_res)
# plt.plot(rom_res,linestyle='--')

# plt.show()

# np.save(r"C:\Users\mohag\OneDrive - University of Kansas\KU - Aerospace PhD\Research Related\Initial_Detonation_Wave_Wall_Reflected\FOM 61000 fine mesh snapshots Explicit - FD Euler prim.npy",fom)

# print(fom[:,0])

# ic_data = fom[:-1,:,-1]

# np.save(r"C:\GIT_Fork\ROMify\examples\wall_reflected_detonation\steady_IC.npy",ic_data)

# fom_small = np.load(r"C:\GIT_Fork\ROMify\examples\classic_shock_tube\FOM 3000 snapshots Explicit - FD Euler prim_dt_1en5.npy")
# arom = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame_with_perturbation\Adaptive ROM 50000 snapshots Explicit - FD Euler GalerkinQDEIM prim.npy")


# fom_large = np.load(r"C:\GIT_Fork\ROMify\examples\classic_shock_tube\FOM 300 snapshots Explicit - FD Euler prim_dt_1en4.npy")


# # arom = np.load(r"C:\GIT_Fork\ROMify\examples\classic_shock_tube\Adaptive ROM 3000 snapshots Explicit - FD Euler GalerkinQDEIM prim.npy")

# # basis = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\basis.npy")
# # norm = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\norm.npy")
# # denorm = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\denorm.npy")
# # qref = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame\q_ref.npy")

# # T = fom[3,:,6000].ravel()
# # P = fom[2,:,6000].ravel()
# # u = fom[1,:,6000].ravel()

# # slice_source = fom[:,201,6000]

# # fom[:,0:201,6000] = slice_source[:,np.newaxis]

# # np.save(r"C:\GIT_Fork\ROMify\examples\transient_flame\steady_IC_trans.npy",fom[:,:,6000])

# # T = fom[3,:,6000].ravel()
# # P = fom[2,:,6000].ravel()
# # u = fom[1,:,6000].ravel()

fig , ax = plt.subplots(2,2)

# fig.set_size_inches(15,6)

# # fig.suptitle('Adaptive ROM dt=5e-5 vs. FOM dt=5e-4')

# # ax[0].plot(T,label='T')
# # ax[1].plot(P,label='P')
# # ax[2].plot(u,label='u')
# # # ax.plot(q_replica,linestyle='--',label='ROM no HR')
# # ax[0].legend()
# # ax[1].legend()
# # ax[2].legend()


x=np.linspace(0,0.12,500)
# iter_axis = np.linspace(0,0.35,300)

# # plt.show()
# counter = 0
# counter2 = 0

# e1 = np.zeros(300)
# e2 = np.zeros(300)

# for iter in range(0,3000,10):

#     # error between AROM and FOM (same dt)
#     e1[counter] = np.linalg.norm( (arom[:,:,iter].ravel()-fom_small[:,:,iter].ravel()) ) / np.linalg.norm( fom_small[:,:,iter].ravel() )
#     # error between AROM and FOM (same dt)
#     e2[counter] = np.linalg.norm( (fom_large[:,:,counter].ravel()-fom_small[:,:,iter].ravel()) ) / np.linalg.norm( fom_small[:,:,iter].ravel() )

#     counter = counter + 1

# ax.plot(iter_axis,e1      ,label='AROM vs FOM (equal dt)')
# ax.plot(iter_axis,e2      ,label='FOM (large dt) vs FOM (small dt)')

# ax.set_ylabel('normalized error')
# ax.set_xlabel('t')


# ax.legend()



# plt.show()

# old_error = np.zeros(3000)
# new_error = np.zeros(3000)


# for iter in range(0):

    # s_indx = np.nonzero(samples[:,iter])[0]

ax[0,0].cla()
ax[1,0].cla()
ax[0,1].cla()
ax[1,1].cla()


rho1    = fom_roe1[-1,:].ravel()
u1      = fom_roe1[1,:].ravel()
P1      = fom_roe1[2,:].ravel()
T1      = fom_roe1[3,:].ravel()

# rho2    = fom_roe2[0,:,iter].ravel()
# u2      = fom_roe2[1,:,iter].ravel()
# P2      = fom_roe2[2,:,iter].ravel()
# T2      = fom_roe2[3,:,iter].ravel()

#     P_fom_super_small    = fom_super_small[2,:,counter2].ravel()
# P_fom_small    = fom_small[2,:,iter].ravel()
#     P_arom_small   = arom[2,:,iter].ravel()
#     P_fom_large    = fom_large[2,:,counter].ravel()

#     ax.plot(x,P_fom_super_small      ,label='FOM-super-small')
# ax.plot(x,P_fom_small      ,label='FOM-small')
#     ax.plot(x,P_arom_small     ,label='AROM')
#     ax.plot(x,P_fom_large      ,label='FOM_large')
ax[0,0].plot(x,rho1      ,color='tab:blue'   ,label='1st Order Roe')
ax[1,0].plot(x,u1         ,color='tab:green' ,label='1st Order Roe')
ax[0,1].plot(x,P1         ,color='tab:orange',label='1st Order Roe')
ax[1,1].plot(x,T1         ,color='tab:red'   ,label='1st Order Roe')

# ax[0,0].plot(x,rho2      ,linestyle='--',color='tab:blue'  ,label='2nd Order Roe')
# ax[1,0].plot(x,u2        ,linestyle='--',color='tab:green' ,label='2nd Order Roe')
# ax[0,1].plot(x,P2        ,linestyle='--',color='tab:orange',label='2nd Order Roe')
# ax[1,1].plot(x,T2        ,linestyle='--',color='tab:red'   ,label='2nd Order Roe')

# ax[0,0].scatter(x[s_indx] , rho[s_indx]      ,color='black')
# ax[1,0].scatter(x[s_indx] , u[s_indx]        ,color='black')
# ax[0,1].scatter(x[s_indx] , a_P[s_indx]      ,color='black')
# ax[1,1].scatter(x[s_indx] , a_T[s_indx]      ,color='black')

ax[0,0].legend()
ax[1,0].legend()
ax[0,1].legend()
ax[1,1].legend()

#     ax.set_xlabel('x[m]')
# ax[0,0].set_xlabel('x[m]')
# ax[1,0].set_xlabel('x[m]')
# ax[0,1].set_xlabel('x[m]')
# ax[1,1].set_xlabel('x[m]')

# ax[0,0].set_ylabel('Heat Release[W/m3]')
# ax[1,0].set_ylabel('Velocity[m/s]')
# ax[0,1].set_ylabel('Pressure[Pa]')
# ax[1,1].set_ylabel('Temperature[K]')
#     ax.set_ylabel('P')

# ax[0,1].set_ylim(95e3,107e3)
# ax[1,0].set_ylim(-10,10)

#     counter = counter + 1
#     counter2 = counter2 + 100

# plt.pause(1e-6)


#     # q_replica = qref+(denorm*(basis@(basis.T@(norm*(q-qref)))))


# print(iter)

    # arom_original_data = arom_original[:,:,iter].ravel()
    # arom_improved_data = arom_improved[:,:,iter].ravel()
    # fom_data           = fom[:,:,iter].ravel()

    # old_error[iter] = np.linalg.norm(arom_original_data - fom_data)/np.linalg.norm(fom_data)
    # new_error[iter] = np.linalg.norm(arom_improved_data - fom_data)/np.linalg.norm(fom_data)


# ax.plot(new_error,label='improved a-rom')
# ax.plot(old_error,label='initial  a-rom')

# ax.set_ylabel('normalized error')
# ax.set_xlabel('iteration')

# ax.set_yscale('log')

# ax.legend()


plt.show()

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


# trans_indx = np.where(x>=0.002)[0][0]

# new_data = fom[:,trans_indx,2000]

# fom[:,0:trans_indx,2000] = new_data[:,np.newaxis]

# new_ic = fom[:,:,2000]

# np.save(r"C:\GIT_Fork\ROMify\examples\free_flame\new_ic.npy",new_ic)




