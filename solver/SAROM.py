import os
import numpy as np
import scipy.linalg as sp_al


from utils import init_func
from utils import reshape_func
from boundary_condition import bc_func

def repeat_pattern_finder(solver_param,rom_param,state):

    old_basis = rom_param['old_basis']
    new_basis = rom_param['basis']

    sub_angle = np.linalg.norm(
                    np.rad2deg(
                    sp_al.subspace_angles(new_basis[:,0:2],old_basis[:,0:2])
                    )
                    )
    
    window_size = int(solver_param['init_training_win'])
    iter        = solver_param['iter']

    state['subspace_angle'][iter]     = sub_angle
    mean_d                            = np.mean(state['subspace_angle'][0:iter])
    cusum_d                           = np.cumsum(state['subspace_angle'][0:iter]-mean_d)
    state['cum_sum'][0:iter]          = (cusum_d - np.min(cusum_d)) / (np.max(cusum_d) - np.min(cusum_d))
    state['moving_avg'][0:iter]       = np.convolve(state['subspace_angle'][0:iter], np.ones(window_size)/window_size, mode='same')

    return state

def init_inc_SVD(solver_param,state,rom_param):

    # This function is initializing the left singular, singular and right singular matrices of incremnetal SVD
    # in the form of QSR = X 

    "Incremental SVD"
    Q_cons_current_step   = state['Q_cons']

    Q_cons_current_step   = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],
                                                                solver_param['num_state_var'],
                                                                state['Q_cons'])
    
    Q_cons_current_step   = rom_param['norm']*(Q_cons_current_step-rom_param['q_ref'])

    # initalize the SVD (u0 is first snapshot)

    u0              = Q_cons_current_step
    m               = np.shape(rom_param['basis'])[0]
    W               = np.eye(m,m)

    S = np.sqrt(u0.T @ W @ u0)
    Q = u0 / S
    R = np.eye(1, 1)

    S *= np.eye(1, 1)
    Q = Q.reshape((Q.shape[0], 1))

    rom_param['W'] = W
    rom_param['Q'] = Q
    rom_param['S'] = S
    rom_param['R'] = R
    rom_param['SROM_training_win'] = np.reshape(Q_cons_current_step,(-1,1))

    return rom_param

def update_inc_SVD(solver_param,state,rom_param):

    Q_cons_current_step      = state['Q_cons']

    Q_cons_current_step      = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],
                                                                    solver_param['num_state_var'],
                                                                    state['Q_cons'])

    cent_norm                = rom_param['norm']*(Q_cons_current_step-rom_param['q_ref'])

    W=rom_param['W'] 
    Q=rom_param['Q'] 
    S=rom_param['S']
    R=rom_param['R'] 

    tol = 1e-15

    m               = np.shape(rom_param['basis'])[0]

    rom_param['SROM_training_win'] = np.hstack((rom_param['SROM_training_win'],cent_norm.reshape(-1,1)))

    u_l = np.reshape(cent_norm,(m,1))

    d = Q.T @ W @ u_l
    if not np.shape(d):
        d *= np.eye(1, 1)
    e = u_l - Q @ d
    p = np.sqrt(e.T @ W @ e) * np.eye(1, 1)

    if p < tol:
        p = np.zeros((1, 1))
    else:
        e = e / p[0, 0].item()

    k = np.shape(S)[0] if np.shape(S) else 1
    Y = np.vstack((np.hstack((S, d)), np.hstack((np.zeros((1, k)), p))))
    Qy, Sy, Ry = np.linalg.svd(Y, full_matrices=True, compute_uv=True)
    Sy = np.diag(Sy)

    l = np.shape(R)[0]
    if p < tol:
        Q = Q @ Qy[:k, :k]
        S = Sy[:k, :k]
        R = np.vstack((np.hstack((R, np.zeros((l, 1)))), np.hstack(
            (np.zeros((1, k)), np.eye(1))))) @ Ry[:, :k]
    else:
        Q = np.hstack((Q, e)) @ Qy
        S = Sy
        R = np.vstack((np.hstack((R, np.zeros((l, 1)))),
                np.hstack((np.zeros((1, k)), np.eye(1))))) @ Ry
        
    if np.abs(Q[:, -1].T @ W @ Q[:, 0]) > tol:
        k = Q.shape[1]
        for i in range(k):
            a = Q[:, i]
            for j in range(i):
                Q[:, i] = Q[:, i] - ((a.T @ W @ Q[:, j]) /
                                    (Q[:, j].T @ W @ Q[:, j])) * Q[:, j]
            norm = np.sqrt(Q[:, i].T @ W @ Q[:, i])
            Q[:, i] = Q[:, i] / norm

    rom_param['W'] = W
    rom_param['Q'] = Q
    rom_param['S'] = S
    rom_param['R'] = R

    return rom_param

def advance_one_time_step(solver_param,state,physics,time_integration,rom_param=None):

    #############################################   
    # This function is taking one time step     #   
    # with self-adapting ROM framework. It uses #
    # the both adaptive ROM and static ROM      #
    # frameworks.                               #    
    #############################################


    iter = solver_param['iter'] 
       
    ###################################################
    #                                                 #
    #    Initial Training of Adaptive ROM (FOM)       #
    #                                                 #
    ###################################################

    if solver_param['SA_Mode_AROM_Training']:

        solver_param, state, rom_param = solver_param['AROM_solver'].advance_one_time_step(solver_param,state,physics,time_integration,rom_param)

        if iter == int(solver_param['FOM2ROM_trans_iter']-1):

            solver_param['SA_Mode_AROM_Training']      = False
            solver_param['SA_Mode_AROM_Transition']    = True

    ###################################################
    #                                                 #
    #    Transition from FOM to Adaptive ROM          #
    #                                                 #
    ###################################################

    elif solver_param['SA_Mode_AROM_Transition']:

        solver_param['solver_mode']      = 'Adaptive ROM'
        solver_param, state, rom_param   = solver_param['AROM_solver'].advance_one_time_step(solver_param,state,physics,time_integration,rom_param)

        state['cum_sum']                 = np.full(solver_param['num_step'],1.0)
        state['moving_avg']              = np.full(solver_param['num_step'],90.0)
        state['subspace_angle']          = np.full(solver_param['num_step'],90.0)

        solver_param['SA_Mode_AROM_Transition']    = False
        solver_param['SA_Mode_SROM_Search']        = True

    ###################################################
    #                                                 #
    #        Looking for Pattern in Solutions         #
    #                                                 #
    ###################################################

    elif solver_param['SA_Mode_SROM_Search']:

        solver_param['solver_mode']      = 'Adaptive ROM'
        solver_param, state, rom_param   = solver_param['AROM_solver'].advance_one_time_step(solver_param,state,physics,time_integration,rom_param)

        state                            = repeat_pattern_finder(solver_param,rom_param,state)

        cum_sum_flag    = np.any(state['cum_sum']<=solver_param['sarom_cumsum_tol'] )
        # moving_avg_flag = np.any(state['moving_avg']<=0.03)
        moving_avg_flag = np.any(state['moving_avg']<=solver_param['sarom_moving_avg_tol'])

        if cum_sum_flag and moving_avg_flag:

            solver_param['SA_Mode_SROM_Search']     = False
            solver_param['SA_Mode_SROM_Training']   = True

            rom_param =  init_inc_SVD(solver_param,state,rom_param)

    ###################################################
    #                                                 #
    #           Gather Data for Static ROM            #
    #                                                 #
    ###################################################

    elif solver_param['SA_Mode_SROM_Training']:

        solver_param, state, rom_param   = solver_param['AROM_solver'].advance_one_time_step(solver_param,state,physics,time_integration,rom_param)

        if iter % solver_param['sarom_training_step'] == 0:

            rom_param = update_inc_SVD(solver_param,state,rom_param)

            square_sum_singular_values = np.sum(np.diag(rom_param['S'])**2)

            # Cumulative energy from first k modes
            cumulative_energy = np.cumsum(np.diag(rom_param['S'])**2)

            # Residual energy percentage
            POD_res_energy = (1 - (cumulative_energy / square_sum_singular_values)) * 100

            truncation_indx = np.array([0])

            if len(POD_res_energy[0:-1]) > 1:

                POD_res_energy_grad = np.abs(np.gradient(POD_res_energy[0:-1]))

                # Find truncation index where residual energy drops below the limit
                truncation_indx = np.where(POD_res_energy_grad < 1e-15)[0]

            if np.shape(rom_param['Q'])[1]>1 and len(truncation_indx) > 1:

                solver_param['SA_Mode_SROM_Training']   = False
                solver_param['SA_Mode_SROM_Transition'] = True
                rom_param['basis']                      = rom_param['Q']

    ###################################################
    #                                                 #
    #               Switch to Static ROM              #
    #                                                 #
    ###################################################

    elif solver_param['SA_Mode_SROM_Transition']:
        
        solver_param['solver_mode'] = solver_param['SAROM_SROM_solver']
        solver , rom_param          = init_func.init_solver(solver_param,state)
        solver_param['solver_mode'] = 'SAROM'

        solver_param['ROM_solver']   = solver

        state['qr'] = rom_param['qr0']

        if solver_param['hyper']:

            from rom.sampling_func import hyper_precompute

            rom_param = hyper_precompute(solver_param,rom_param)

        solver_param, state, rom_param          = solver_param['ROM_solver'].advance_one_time_step(solver_param,state,physics,time_integration,rom_param)

        solver_param['SA_Mode_SROM_Transition'] = False
        solver_param['SA_Mode_SROM']            = True

    ###################################################
    #                                                 #
    #                   Run Static ROM                #
    #                                                 #
    ###################################################

    elif solver_param['SA_Mode_SROM']:

        solver_param, state, rom_param          = solver_param['ROM_solver'].advance_one_time_step(solver_param,state,physics,time_integration,rom_param)

    return solver_param, state, rom_param