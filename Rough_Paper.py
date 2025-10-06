import sys
import os
import numpy as np
import cantera as ct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


# import numpy as np

# prim   = np.load(r"C:\Users\mohag\OneDrive\Desktop\RDE_ROMify_AROM\examples\1D_RDE\Adaptive ROM_results\cons_prim\609000iteration_prim.npy")
# sample = np.load(r"C:\Users\mohag\OneDrive\Desktop\RDE_ROMify_AROM\examples\1D_RDE\Adaptive ROM_results\samples_user\609000iteration_samples_user.npy")


# x = np.linspace(0,0.0288,500)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(x,prim[2,:],color='tab:blue')
# plt.scatter(x[sample],prim[2,sample],color='black')
# # plt.plot(implicit_new[2,:],color='black')


# plt.show()


# prim_ic = np.load(r'C:\Users\mohag\OneDrive\Desktop\RDE_ROMify_FOM_AROM2\RDE_ROMify_FOM_AROM2\examples\1D_RDE\Adaptive ROM_results\cons_prim\2500000iteration_prim.npy')
# prim_ic = prim_ic[:-1,:]
# np.save(r"C:\GIT_Fork\ROMify\examples\1D_RDE\ROM_IC.npy",prim_ic)

dir_name_FOM = r"C:\GIT_Fork\ROMify\examples\supersonic_flow\AROM_results\cons_prim"
dir_name_ROM = r"C:\GIT_Fork\ROMify\examples\supersonic_flow\SAROM_results\cons_prim"
# dir_name = r"C:\GIT_Fork\ROMify\examples\supersonic_flow\Hybrid ROM_results\cons_prim"
# dir_name_fom = r'C:\GIT_Fork\ROMify\examples\1D_RDE\FOM_results'
# dir_name_fom = r'C:\GIT_Fork\ROMify\examples\wall_reflected_detonation\FOM_results'
# dir_name_fom = r'C:\GIT_Fork\ROMify\examples\1D_RDE\Adaptive ROM_results\cons_prim'
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\RDE_ROMify_clean\examples\1D_RDE\FOM_results"
# dir_name_ali = r"C:\Users\mohag\OneDrive\Desktop\RDE_ROMify_clean\examples\1D_RDE\FOM_results"
# dir_name_ali = r'C:\GIT_Fork\ROMify\examples\1D_RDE\Adaptive ROM_results\cons_prim'
# dir_name_ku  = r"C:\Users\mohag\OneDrive\Desktop\cons_prim"
# dir_name_ku_sample  = r"C:\Users\mohag\OneDrive\Desktop\samples_user"
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\1D_RDE_Simulation_5\examples\1D_RDE\FOM_results"
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\1D_RDE_5Inject\examples\1D_RDE\FOM_results"
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\RDE_ROMify_Cantera_Inj_250_Final\examples\1D_RDE\FOM_results"
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\RDE_ROMify_Final_Test_Larger_Domain\examples\1D_RDE\FOM_results"
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\non_intrusive_rde\1D_RDE_results\FOM_results"
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\cons_prim"
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\RDE_ROMify_FOM_AROM2\RDE_ROMify_FOM_AROM2\examples\1D_RDE\Adaptive ROM_results\cons_prim"
# dir_name_fom = r'C:\Users\mohag\OneDrive\Desktop\RDE_ROMify_Adaptive_ROM_Inj\examples\1D_RDE\Adaptive ROM_results\cons_prim'
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\cons_prim"
# dir_name_fom = r'C:\Users\mohag\OneDrive\Desktop\FGS_Journal_Paper_Graphs\detonation_data\FOM_results_4en9'
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\Detonation_Reflected\examples\wall_reflected_detonation\Adaptive ROM_results\cons_prim"
# dir_name_fom = r"C:\Users\mohag\OneDrive\Desktop\Detonation_Reflected\examples\wall_reflected_detonation\Adaptive ROM_results\cons_prim"
# dir_name_fom = r"C:\GIT_Fork\ROMify\examples\free_flame_with_perturbation\FOM_results"
# dir_name = r"C:\GIT_Fork\ROMify\examples\free_flame_with_perturbation\FOM_results"

# fig ,ax = plt.subplots(1,2)
# fig.set_size_inches(15,8)

# x= np.linspace(0,0.01,500)

# prim = np.load(os.path.join(dir_name,str(0)+'iteration'+'_prim.npy'))
# fom  = np.load(os.path.join(dir_name_fom,str(0)+'iteration'+'_prim.npy'))

# line11, = ax[0].plot(x,fom[2,:],c='black',lw=4,label='FOM')
# line12, = ax[0].plot(x,prim[2,:],c='tab:blue',ls='--',lw=3,label='Adaptive ROM')

# line21, = ax[1].plot(x,fom[3,:],c='black',lw=4,label='FOM')
# line22, = ax[1].plot(x,prim[3,:],c='tab:blue',ls='--',lw=3,label='Adaptive ROM')


# ax[0].set_ylabel('Pressure [Pa]',fontsize=14)
# ax[1].set_ylabel('Temperature [K]',fontsize=14)

# ax[0].set_xlabel('x [m]',fontsize=14)
# ax[1].set_xlabel('x [m]',fontsize=14)

# ax[0].tick_params(axis='both', labelsize=12)
# ax[1].tick_params(axis='both', labelsize=12)

# ax[0].grid(True,which='both')
# ax[1].grid(True,which='both')



# press_error = np.zeros(len(range(0,113040,100)))

# writer = FFMpegWriter(fps=30, metadata=dict(artist='Your Name'), bitrate=1800)

# with writer.saving(fig, "output.mp4", dpi=100):

#     for iter in range(0,113040,100):

#         prim = np.load(os.path.join(dir_name,str(iter)+'iteration'+'_prim.npy'))
#         fom  = np.load(os.path.join(dir_name_fom,str(int(iter/10))+'iteration'+'_prim.npy'))

#         line11.set_data(x,fom[2,:])
#         line12.set_data(x,prim[2,:])

#         if iter>10:
#             line12.set_data(x,prim[2,:]*1.0005)
        

#         line21.set_data(x,fom[3,:])
#         line22.set_data(x,prim[3,:])
#         # ax.plot(x,prim[2,:],label=str(iter/12560))

#         # ax.relim()
#         # ax.autoscale()
#         ax[0].set_xlim([0, 0.01])
#         ax[0].set_ylim([99000, 104000])

#         ax[0].set_title('Simulation Time: ' + str(int(iter)) + ' ns')
#         ax[1].set_title('Simulation Time: ' + str(int(iter)) + ' ns')

#         ax[1].set_xlim([0, 0.01])
#         ax[1].set_ylim([250, 2000])

#         if iter >= 50240:
            
#             # line11.set_color('tab:red')
#             line12.set_color('tab:red')
#             line12.set_label('Conventional ROM')

#             # line21.set_color('tab:red')
#             line22.set_color('tab:red')
#             line22.set_label('Conventional ROM')
        

#         ax[0].legend()
#         ax[1].legend()

#         plt.pause(0.001)
#         plt.draw()

#         print(iter)

#         writer.grab_frame()

#     # press_error[int(iter/100)] = np.linalg.norm(fom[2,:]-prim[2,:])/np.linalg.norm(fom[2,:])

# prim = np.load(os.path.join(dir_name,str(138159)+'iteration'+'_prim.npy'))

# line1.set_data(x,prim[2,:])
# ax.plot(x,prim[2,:],label=str(iter/12560))

# fig,ax = plt.subplots(1,1)

# t = np.linspace(0,113040,len(range(0,113040,100)))

# ax.plot(t,press_error,c='black',label='')

# ax.axvline(25120,color='grey',ls='--',label='Start Training')
# ax.axvline(50240,color='tab:red',ls='--',label='End Training (Switch)')

# ax.set_yscale('log')
# ax.set_ylabel('Pressure Error')
# ax.set_xlabel('Time [ns]')
# ax.grid(True,which='both')
# ax.legend()

# plt.show()


# iter_50k = np.load(r"C:\Users\mohag\Desktop\FOM_Results\50000iteration_prim.npy")
# temp_ic  = iter_50k[:-1,:]
# np.save(r"C:\Users\mohag\Desktop\after_reflection_ic.npy",temp_ic)


# dir_results_fom = r"C:\Users\mohag\Desktop\FOM_results"
# # dir_results_arom = r"C:\Users\mohag\Desktop\Adaptive ROM_results\cons_prim"
# # dir_results_arom_basis = r"C:\Users\mohag\Desktop\Adaptive ROM_results\basis"
# # dir_results_arom_sample = r"C:\Users\mohag\Desktop\Adaptive ROM_results\samples_solver"
# max_iter    = 51000
# freq_vis    = 1000


# sample_cons_snapshot = np.load(os.path.join(dir_results_fom, '0iteration_cons.npy'))
# sample_cons_shape    = np.shape(sample_cons_snapshot)

# sample_prim_snapshot = np.load(os.path.join(dir_results_fom, '0iteration_prim.npy'))
# sample_prim_shape    = np.shape(sample_prim_snapshot)

# # sample_basis_snapshot = np.load(r"C:\Users\mohag\Desktop\Adaptive ROM_results\basis\10iteration_basis.npy")
# # sample_basis_shape    = np.shape(sample_basis_snapshot)

# # sample_samples_solver_snapshot = np.load(r"C:\Users\mohag\Desktop\Adaptive ROM_results\samples_solver\10iteration_samples_solver.npy")
# # sample_samples_solver_shape    = np.shape(sample_samples_solver_snapshot)

# data_cons_fom = np.zeros((sample_cons_shape[0],sample_cons_shape[1],int(max_iter/freq_vis)))
# data_prim_fom = np.zeros((sample_prim_shape[0],sample_prim_shape[1],int(max_iter/freq_vis)))

# # data_cons_arom = np.zeros((sample_cons_shape[0],sample_cons_shape[1],int(max_iter/freq_vis)))
# # data_prim_arom = np.zeros((sample_prim_shape[0],sample_prim_shape[1],int(max_iter/freq_vis)))

# # basis_matrix   = np.zeros((sample_basis_shape[0],sample_basis_shape[1],int(max_iter/freq_vis)))

# # sample_matrix   = np.zeros((sample_samples_solver_shape[0],int(max_iter/freq_vis)),dtype=int)

# print('Assembling Snapshots!')

# counter = 0

# # bring all of snapshots into one matrix
# for i in range(0,max_iter,freq_vis):

#     file_name_cons = str(i)+'iteration'+'_cons.npy'
#     file_name_prim = str(i)+'iteration'+'_prim.npy'

#     data_cons_fom[:, :, counter] = np.load(os.path.join(dir_results_fom, file_name_cons))
#     data_prim_fom[:, :, counter] = np.load(os.path.join(dir_results_fom, file_name_prim))

#     # data_cons_arom[:, :, counter] = np.load(os.path.join(dir_results_arom, file_name_cons))
#     # data_prim_arom[:, :, counter] = np.load(os.path.join(dir_results_arom, file_name_prim))

#     counter=counter+1


# counter = 0

# # bring all of snapshots into one matrix
# for i in range(1000,max_iter,freq_vis):

#     file_name_basis = str(i)+'iteration'+'_basis.npy'

#     basis_matrix[:, :, counter] = np.load(os.path.join(dir_results_arom_basis,file_name_basis))

#     counter=counter+1

# counter = 0
# # bring all of snapshots into one matrix
# for i in range(1000,max_iter,freq_vis):

#     file_name_sample = str(i)+'iteration'+'_samples_solver.npy'

#     sample_matrix[:, counter] = np.load(os.path.join(dir_results_arom_sample,file_name_sample))

#     counter=counter+1


# # calculate proj error

# proj_error = np.zeros((int(max_iter/freq_vis)))

# for i in range(0,int(max_iter/freq_vis)):

#     I = np.eye(sample_basis_shape[0],sample_basis_shape[0])

#     proj_matrix = I - (basis_matrix[:,:,i]@basis_matrix[:,:,i].T)

#     proj_error[i] = np.linalg.norm(proj_matrix)


# counter=0

# sample_error = np.zeros((int(max_iter/freq_vis)))

# for i in range(0,int(max_iter/freq_vis)):

#     pinv_ST_V = np.linalg.pinv(basis_matrix[sample_matrix[:, counter],:,i])

#     sample_error[i] = np.linalg.norm(pinv_ST_V)


# plt.figure()
# plt.plot(proj_error)
# # plt.yscale('log')

# plt.figure()
# plt.plot(sample_error)
# # plt.yscale('log')

# data_prim_fom = np.load(r'examples\free_flame_with_perturbation\FOM_results\assembled_prim.npy')
# data_prim_fom = np.load(r"C:\GIT_Fork\ROMify\examples\free_flame_with_perturbation\Adaptive ROM_results\cons_prim\assembled_prim.npy")
# data_prim_rom1 = np.load(r'C:\GIT_Fork\ROMify\examples\free_flame_with_perturbation\ROM_no_hyper_results\cons_prim\0iteration_prim.npy')
# data_prim_rom2 = np.load(r'C:\GIT_Fork\ROMify\examples\free_flame_with_perturbation\ROM_no_hyper_results\cons_prim\12560iteration_prim.npy')
# data_prim_rom3 = np.load(r'C:\GIT_Fork\ROMify\examples\free_flame_with_perturbation\ROM_no_hyper_results\cons_prim\25120iteration_prim.npy')
# data_prim_rom4 = np.load(r'C:\GIT_Fork\ROMify\examples\free_flame_with_perturbation\ROM_no_hyper_results\cons_prim\37679iteration_prim.npy')

# fig , ax = plt.subplots(1,1)

# x=np.linspace(0,0.01,500)

# ax.plot(x,data_prim_fom[2,:,0*1256],label='IC')
# ax.plot(x,data_prim_fom[2,:,10*1256],label='Cycle 1-A-ROM')
# ax.plot(x,data_prim_fom[2,:,20*1256],label='Cycle 2-A-ROM')
# ax.plot(x,data_prim_fom[2,:,30*1256],label='Cycle 3-A-ROM')
# ax.plot(x,data_prim_fom[2,:,40*1256],label='Cycle 4-A-ROM')
# ax.plot(x,data_prim_fom[2,:,50*1256],label='Cycle 5-A-ROM')
# ax.plot(x,data_prim_fom[2,:,60*1256],label='Cycle 6-A-ROM')
# ax.plot(x,data_prim_fom[2,:,70*1256],label='Cycle 7-A-ROM')
# ax.plot(x,data_prim_fom[2,:,80*1256-1],label='Cycle 8-A-ROM')
# ax.plot(x,data_prim_rom1[2,:],label='Cycle 6-ROM',ls='--')
# ax.plot(x,data_prim_rom2[2,:],label='Cycle 7-ROM',ls='--')
# ax.plot(x,data_prim_rom3[2,:],label='Cycle 8-ROM',ls='--')
# ax.plot(x,data_prim_rom4[2,:],label='Cycle 9-ROM',ls='--')

# ax.legend()

# plt.show()




# for iter in range(0,int(max_iter/freq_vis)):


#     ax[0,0].cla()
#     ax[1,0].cla()
#     ax[0,1].cla()
#     ax[1,1].cla()


#     hr1     = data_prim_fom[0,: ,iter].ravel()
#     u1      = data_prim_fom[1 ,: ,iter].ravel()
#     P1      = data_prim_fom[2 ,: ,iter].ravel()
#     T1      = data_prim_fom[3 ,: ,iter].ravel()

#     # rho2    = data_prim_arom[0,: ,iter].ravel()
#     # u2      = data_prim_arom[1 ,: ,iter].ravel()
#     # P2      = data_prim_arom[2 ,: ,iter].ravel()
#     # T2      = data_prim_arom[3 ,: ,iter].ravel()

#     ax[0,0].plot(x,hr1        ,color='tab:blue'  , label='FOM')
#     ax[1,0].plot(x,u1         ,color='tab:green' , label='FOM')
#     ax[0,1].plot(x,P1         ,color='tab:orange', label='FOM')
#     ax[1,1].plot(x,T1         ,color='tab:red'   , label='FOM')

#     # ax[0,0].plot(x,rho2      ,linestyle='--',color='tab:blue'  ,label='AROM')
#     # ax[1,0].plot(x,u2        ,linestyle='--',color='tab:green' ,label='AROM')
#     # ax[0,1].plot(x,P2        ,linestyle='--',color='tab:orange',label='AROM')
#     # ax[1,1].plot(x,T2        ,linestyle='--',color='tab:red'   ,label='AROM')

#     # ax[0,0].scatter(x[s_indx] , rho[s_indx]      ,color='black')
#     # ax[1,0].scatter(x[s_indx] , u[s_indx]        ,color='black')
#     # ax[0,1].scatter(x[s_indx] , a_P[s_indx]      ,color='black')
#     # ax[1,1].scatter(x[s_indx] , a_T[s_indx]      ,color='black')

#     # ax[0,0].legend()
#     # ax[1,0].legend()
#     # ax[0,1].legend()
#     # ax[1,1].legend()

#     #     ax.set_xlabel('x[m]')
#     # ax[0,0].set_xlabel('x[m]')
#     # ax[1,0].set_xlabel('x[m]')
#     # ax[0,1].set_xlabel('x[m]')
#     # ax[1,1].set_xlabel('x[m]')

#     # ax[0,0].set_ylabel('Heat Release[W/m3]')
#     # ax[1,0].set_ylabel('Velocity[m/s]')
#     # ax[0,1].set_ylabel('Pressure[Pa]')
#     # ax[1,1].set_ylabel('Temperature[K]')
#     #     ax.set_ylabel('P')

#     # ax[0,1].set_ylim(95e3,107e3)
#     # ax[1,0].set_ylim(-10,10)

#     #     counter = counter + 1
#     #     counter2 = counter2 + 100

#     print(iter*freq_vis)

#     plt.pause(1e-6)



# plt.show()


# training_data_cons = np.load(r"C:\GIT_Fork\ROMify\examples\supersonic_flow\Hybrid ROM_results\cons_prim\assembled_cons.npy")
# # training_data_prim = np.load()

# training_data_cons = training_data_cons[:,:,25120:50240:1000]

# first_snapshot     = training_data_cons[:,:,-1]


# POD_energy_limit   = 100-99.999
# # first_snapshot     = np.mean(training_data,axis=2)

# state_var_num      = len(training_data_cons[:,0,0])
# snapshot_num       = len(training_data_cons[0,0,:])
# cell_num           = 500

# centered_data      = np.empty((state_var_num,cell_num,snapshot_num))

# for indx in range(0,snapshot_num):

#     centered_data[:,:,indx] = training_data_cons[:,:,indx] - first_snapshot

# l2_factors         = np.sqrt(np.sum(centered_data**2, axis=2))
# norm_factor        = np.mean(l2_factors, axis=1)

# cen_norm_data      = np.zeros((state_var_num,cell_num,snapshot_num))

# for i in range(0,state_var_num):

#     for j in range(0,snapshot_num):

#         cen_norm_data[i,:,j] = centered_data[i,:,j] / norm_factor[i]


# tall_thin_data = np.zeros((state_var_num * cell_num, snapshot_num))
    
# for indx in range(0,snapshot_num):

#     tall_thin_data[:, indx] = cen_norm_data[:,:,indx].ravel(order='C')

# V, S, U = np.linalg.svd(tall_thin_data, full_matrices=False)

# square_sum_singular_values = np.sum(S**2)

# POD_res_energy = [(1 - np.sum(S[:indx]**2) / square_sum_singular_values) * 100 for indx in range(len(S))]

# truncation_indx = np.where(np.array(POD_res_energy) < POD_energy_limit )


# basis         = V[:,0:truncation_indx[0][0]]


# normalizor_size = 3*500

# denormalizor = np.zeros(normalizor_size)

# for indx in range(normalizor_size):
    
#     norm_factor_index = int(np.floor(indx/500))

#     denormalizor[indx] = norm_factor[norm_factor_index]

# normalizor      = 1/denormalizor
# q_ref           = first_snapshot.ravel(order='C')



fig,ax = plt.subplots(1,1)

ax2 = ax.twinx()

iter_list = np.arange(0,100000,200)
# iter_list = [79999]
# iter_list = [628,1256,1884,2512,3140,3768,4396,5024,5652,6280,
#              6908,7536]

x=np.linspace(0,1,500)

# prim_data_rom   = np.load(os.path.join(dir_name, str(0)+'iteration_prim.npy'))
prim_data_ali     = np.load(os.path.join(dir_name_FOM, str(int(iter_list[0]))+'iteration_prim.npy'))
prim_data_ali2    = np.load(os.path.join(dir_name_ROM, str(int(iter_list[0]))+'iteration_prim.npy'))
# prim_data_ku     = np.load(os.path.join(dir_name_ku, str(int(10))+'iteration_prim.npy'))
# sample_data_ku   = np.load(os.path.join(dir_name_ku_sample, str(int(10))+'iteration_samples_user.npy'))
# cons_data_fom   = np.load(os.path.join(dir_name_fom, str(int(0))+'iteration_prim.npy'))

# ax.axhline(97000,0,0.288)
line1,=ax.plot(x,prim_data_ali[0,:],label=str(iter)+'iteration',c='blue',ls='-')
# line2,=ax2.plot(x,prim_data_ali2[-1,:],label=str(iter)+'iteration',c='black',ls='--')
line2,=ax.plot(x,prim_data_ali2[0,:],label=str(iter)+'iteration',c='black',ls='--')
# line3,=ax.plot(x[sample_data_ku],prim_data_ku[-1,sample_data_ku],label=str(iter)+'iteration',c='tab:red',ls='',marker='o')

ax.set_ylim([1,2])
# ax.set_ylabel('Pressure [Pa]')
# ax.set_ylim([-1500,1500])
# ax2.set_ylim([0,5e6])
# ax.set_ylim([0,3500])
# ax.set_ylim([0,0.02])
for iter in iter_list:

    # q_solver = q_user.ravel()

    # qr = basis.T @ (normalizor*(q_solver-q_ref))

    # q_rep_solver = q_ref + denormalizor * (basis @ qr)

    # q_rep_user = np.reshape(q_rep_solver,(3,500))

    # prim_data_rom   = np.load(os.path.join(dir_name, str(iter)+'iteration_prim.npy'))
    prim_data_ali   = np.load(os.path.join(dir_name_FOM, str(int(iter))+'iteration_prim.npy'))
    prim_data_ali2   = np.load(os.path.join(dir_name_ROM, str(int(iter))+'iteration_prim.npy'))
    # prim_data_ku    = np.load(os.path.join(dir_name_ku, str(int(iter))+'iteration_prim.npy'))
    # sample_data_ku  = np.load(os.path.join(dir_name_ku_sample, str(int(iter))+'iteration_samples_user.npy'))


    line1.set_data(x,prim_data_ali[0,:])
    line2.set_data(x,prim_data_ali2[0,:])
    # line3.set_data(x[sample_data_ku],prim_data_ku[2,sample_data_ku])

    # if iter == 204000:
    #     breakpoint()
    #     # line1.set_color('tab:red')

    print(iter)

    plt.pause(0.001)


ax.legend()
plt.show()





# import numpy as np

# rho_in      = 1.2823549145222093*2
# # u_in        = Q_prim_user[1,:] * 0 + 0.01
# v_in        = 100
# T_in        = 300
# P_in        = 101325*2
# Y_in        = np.array([0.01277243, 0.        , 0.        , 0.10136214, 0.        ,
#                         0.        , 0.        , 0.        , 0.88586543, 0.            ]).reshape(-1,1)

# prim_state_inject = np.zeros((14,1))

# prim_state_inject[0] =rho_in
# prim_state_inject[1] =v_in
# prim_state_inject[2] =P_in
# prim_state_inject[3] =T_in
# prim_state_inject[4:]=Y_in

# np.save(r'C:\GIT_Fork\ROMify\examples\1D_RDE\inject_prim_state.npy',prim_state_inject)