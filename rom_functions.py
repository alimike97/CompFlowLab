import time
import numpy as np
import scipy as sc
import solver_functions
import time_integrator_functions
from matplotlib import pyplot as plt

def precomputer(solver_param,state):

    print('Initializing the ROM mode!')

    if solver_param['solver_mode'] == 'ROM':

        training_data_path_cons = solver_param['FOM_result_dir']
        training_data_path_prim = training_data_path_cons.replace(' cons.npy',' prim.npy')

        training_data_cons      = np.load(training_data_path_cons)
        training_data_prim      = np.load(training_data_path_prim)
        first_snapshot          = training_data_cons[:,:,0]        

    elif solver_param['solver_mode'] == 'Adaptive ROM':

        iter = solver_param['iter']

        training_data_cons = state['cons_results_save'][:,:,0:iter]
        training_data_prim = state['prim_results_save'][:,:,0:iter]
        first_snapshot     = training_data_cons[:,:,-1]
        # first_snapshot     = np.mean(training_data_cons,axis=2)


    POD_energy_limit   = 100-solver_param['pod_energy']
    # first_snapshot     = np.mean(training_data,axis=2)

    state_var_num      = len(training_data_cons[:,0,0])
    snapshot_num       = len(training_data_cons[0,0,:])
    cell_num           = solver_param['cell_number']

    centered_data      = np.empty((state_var_num,cell_num,snapshot_num))

    for indx in range(0,snapshot_num):

        centered_data[:,:,indx] = training_data_cons[:,:,indx] - first_snapshot

    l2_factors         = np.sqrt(np.sum(centered_data**2, axis=2))
    norm_factor        = np.mean(l2_factors, axis=1)

    cen_norm_data      = np.zeros((state_var_num,cell_num,snapshot_num))

    for i in range(0,state_var_num):

        for j in range(0,snapshot_num):

            cen_norm_data[i,:,j] = centered_data[i,:,j] / norm_factor[i]


    tall_thin_data = np.zeros((state_var_num * cell_num, snapshot_num))
    
    for indx in range(0,snapshot_num):

        tall_thin_data[:, indx] = cen_norm_data[:,:,indx].ravel(order='C')

    V, S, U = np.linalg.svd(tall_thin_data, full_matrices=False)

    square_sum_singular_values = np.sum(S**2)

    POD_res_energy = [(1 - np.sum(S[:indx]**2) / square_sum_singular_values) * 100 for indx in range(len(S))]

    truncation_indx = np.where(np.array(POD_res_energy) < POD_energy_limit )

    print(f'to capture '+ str(solver_param['pod_energy']) +' percent of energy '+ str(truncation_indx[0][0])+ ' mode must be used')

    time.sleep(0.5)

    if solver_param['solver_mode'] == 'Adaptive ROM':

        basis           = V

    else:

        basis         = V[:,0:truncation_indx[0][0]]
        # basis         = V[:,0:1]
        # basis           = V
    
    normalizor_size = 3*solver_param['cell_number']

    denormalizor = np.zeros(normalizor_size)

    for indx in range(normalizor_size):
        
        norm_factor_index = int(np.floor(indx/solver_param['cell_number']))

        denormalizor[indx] = norm_factor[norm_factor_index]

    normalizor      = 1/denormalizor
    q_ref           = first_snapshot.ravel(order='C')

    print(str(len(basis[0,:])) + ' modes are going to be used in this simulation')

    time.sleep(1.5)

    rom_param     = {}

    rom_param['basis']              = basis
    rom_param['normalizor']         = normalizor
    rom_param['denormalizor']       = denormalizor
    rom_param['q_ref']              = q_ref
    rom_param['training_data_prim'] = training_data_prim

    Q_cons_solver = state['Q_cons']
    Q_cons_user   = solver_functions.results_solver2user_converter(cell_num,Q_cons_solver)
    Q_cons_interior_user = Q_cons_user[:,2:-2]
    Q_cons_interior_solver = solver_functions.results_user2solver_converter(Q_cons_interior_user) 
    rom_param['q_red0'] = basis.T @ Q_cons_interior_solver

    num_consv_var = 3
    
    S_indx_user   = np.arange(0,solver_param['cell_number'])
    pcc           = 0
    S_indx_solver = user2solver_indx_converter(S_indx_user,num_consv_var,solver_param['cell_number'])

    rom_param['S_indx_user']      = S_indx_user
    rom_param['S_indx_solver']    = S_indx_solver
    rom_param['hyper_precompute'] = pcc

    return rom_param

def hyper_precomputer(basis,S_indx_solver):

    pcc = basis @ np.linalg.pinv(basis[S_indx_solver,:])

    return pcc

def red2full_state_calculator(solver_param,rom_param,state):

    if solver_param['rom_method'] == 'Galerkin':

        if solver_param['hyper'] == False:

            RES_solver       = state['d_flux_dx']
            RES_user         = solver_functions.results_solver2user_converter(solver_param['cell_number'],RES_solver)
            RES              = RES_user[:,2:-2].ravel()

        else:
            pcc              = rom_param['hyper_precompute']
            RES_solver       = state['d_flux_dx']   
            RES              = pcc @ RES_solver


        Q0_red           = rom_param['q_red0']
        q_ref            = rom_param['q_ref']
        normalizor       = rom_param['normalizor']
        denormalizor     = rom_param['denormalizor']
        basis            = rom_param['basis']

        dQ_red_dt        = basis.T @ (normalizor * RES)

        state['Q_cons']  = Q0_red

        state['d_flux_dx']= dQ_red_dt

        state            = time_integrator_functions.advance_time(solver_param,state)

        Q_red            = state['Q_cons']

        Q_full_order_solver= q_ref + (denormalizor * (basis @ Q_red)) 

        Q_full_order_user  = solver_functions.results_solver2user_converter(solver_param['cell_number']-4,Q_full_order_solver)

        Q_full_order_user  = np.column_stack((Q_full_order_user[:,0], Q_full_order_user[:,0], Q_full_order_user , Q_full_order_user[:,-1] , Q_full_order_user[:,-1]))

        state['Q_cons']    = solver_functions.results_user2solver_converter(Q_full_order_user)

        rom_param['q_red0']    = Q_red

        state['d_flux_dx'] = RES


    return state , rom_param

def user2solver_indx_converter(S_indx_user,num_consv_var,num_cell):

    num_selected_cell = np.size(S_indx_user)

    S_indx_solver     = np.zeros((num_consv_var*num_selected_cell),dtype=int)
    
    for j in np.arange(0,num_consv_var):
        
        start = j*num_selected_cell
        end   = (j+1)*num_selected_cell

        S_indx_solver[start:end] = S_indx_user + j * num_cell

    return S_indx_solver

def solver2user_indx_converter(S_indx_solver,num_cell):

    all_vars_vector_size = np.size(S_indx_solver)
    
    S_indx_user = np.array([])

    for j in range(0,all_vars_vector_size):

        if S_indx_solver[j] >= num_cell:

            coeff = np.floor(S_indx_solver[j]/num_cell)

            new_indx = S_indx_solver[j] - coeff * num_cell

        else: 

            new_indx = S_indx_solver[j]

        S_indx_user = np.append(S_indx_user, new_indx)  

    S_indx_user = S_indx_user.astype(int)

    return S_indx_user

def sample_point_finder(solver_param,rom_param):

    basis = rom_param['basis']

    if solver_param['sampling_method'] == 'DEIM':

            S_indx_user = DEIM_sample_point_finder(basis,solver_param['cell_number'])

    elif solver_param['sampling_method'] == 'QDEIM':
            
            S_indx_user = QDEIM_sample_point_finder(basis,solver_param['cell_number'])

    elif solver_param['sampling_method'] == 'Gappy POD':
            

            center = np.arange(200,300,1,dtype=int)
            left        = np.arange(0,50,1,dtype=int)
            # right       = np.arange(490,499,1,dtype=int)

            # S_indx_user = np.concatenate((left,center,right))
            S_indx_user = np.concatenate((left,center))
            # S_indx_user = center



    elif solver_param['sampling_method'] == 'Gappy POD + E':
            
            num_samples = 150

            S_indx_user = GappyPODE_sample_point_finder(basis,num_samples,solver_param['cell_number'])
    
    num_consv_var  = 3
    S_indx_solver = user2solver_indx_converter(S_indx_user,num_consv_var,solver_param['cell_number'])
    pcc           = hyper_precomputer(basis,S_indx_solver)

    rom_param['hyper_precompute'] = pcc
    rom_param['S_indx_user']      = S_indx_user
    rom_param['S_indx_solver']    = S_indx_solver

    return rom_param

def DEIM_sample_point_finder(basis,num_cell):

    max_mode_value = np.max(np.abs(basis[:,0]))
    max_mode_indx  = np.argmax(np.abs(basis[:,0]),axis=0)

    S_indx_temp = max_mode_indx

    num_grids = np.shape(basis)[0]
    num_modes = np.shape(basis)[1]
    
    

    U = basis[:,0]
    U = U.reshape(-1, 1)
    P = np.zeros((num_grids,1))
    P[max_mode_indx] = 1


    for mode in range(1,num_modes):
        
        u_l = basis[:,mode].reshape(-1,1)
        c = np.linalg.solve(np.transpose(P) @ U , np.transpose(P) @ u_l)
        r = u_l - U @ c
        new_max_value = np.max(np.abs(r))
        new_indx      = np.argmax(np.abs(r))
        new_P_vector  = np.zeros((num_grids,1))
        new_P_vector[new_indx] = 1 
        U = np.hstack([U,u_l])
        P = np.hstack([P,new_P_vector])
        S_indx_temp = np.hstack([S_indx_temp,new_indx])

    S_indx_solver = S_indx_temp

    S_indx_user   = solver2user_indx_converter(S_indx_solver,num_cell)

    return S_indx_user

def QDEIM_sample_point_finder(U,num_cell):

    n, m = U.shape

    num_selected_samples = 0
    counter = 0

    Q, R, P = sc.linalg.qr(U.T, mode='full',pivoting=True)

    S_indx_solver = P[:m]

    S_indx_user   = solver2user_indx_converter(S_indx_solver,num_cell)

    S_indx_user   = np.unique(S_indx_user)

    num_selected_samples = len(S_indx_user)
    
    while num_selected_samples < m:
        
        start_indx = m + counter
        end_indx   = m + counter + 1

        new_sample = P[start_indx:end_indx]

        S_indx_solver = np.append(S_indx_solver,new_sample)

        S_indx_user   = solver2user_indx_converter(S_indx_solver,num_cell)

        S_indx_user   = np.unique(S_indx_user)

        num_selected_samples = len(S_indx_user)

        counter = counter + 1

    S_indx_user = np.sort(S_indx_user)

    S_indx_user = S_indx_user.astype(int)

    return S_indx_user

def GappyPODE_sample_point_finder(basis,num_samples,num_cell):

    _, _, p = sc.linalg.qr(basis.T, mode='full', pivoting=True)

    p = p[:basis.shape[1]]

    for i in range(len(p) + 1, num_samples + 1):

        _, S, W = np.linalg.svd(basis[p, :], full_matrices=False)

        g = S[-2]**2 - S[-1]**2

        Ub = np.dot(W.T, basis.T)

        r = g + np.sum(Ub**2, axis=0)

        r = r - np.sqrt((g + np.sum(Ub**2, axis=0))**2 - 4 * g * Ub[-1, :]**2)

        I = np.argsort(r)[::-1]

        e = 0

        while I[e] in p:

            e += 1

        p = np.append(p, I[e])


    S_indx_solver = p

    S_indx_user   = solver2user_indx_converter(S_indx_solver,num_cell)

    return S_indx_user

def single_adaptive_rom_progress(solver_param,rom_param,state,iter):

    if iter <= int(solver_param['init_training_win']):

        if iter != int(solver_param['init_training_win']):
            
            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state = time_integrator_functions.advance_time(solver_param,state)

        elif iter == int(solver_param['init_training_win']):

            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state = time_integrator_functions.advance_time(solver_param,state)

            rom_param = precomputer(solver_param,state)
            rom_param = sample_point_finder(solver_param,rom_param)
            solver_param['hyper'] = True
            state['Q_bar'] = state['Q_cons']


    else:   
            
            # read basics parameters
            q_ref            = rom_param['q_ref']
            normalizor       = rom_param['normalizor']
            denormalizor     = rom_param['denormalizor']


            # Q tilda (ROM) before any update
            Q_tilda_old = state['Q_cons']
            Q_tilda_old_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_tilda_old)

            # Find new Q tilda using old basis (prediction)
            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state , rom_param = red2full_state_calculator(solver_param,rom_param,state)
            Q_bar_residual_unsampled= state['d_flux_dx']
            

            # new Q_r
            Q_red_new = rom_param['q_red0']

            Q_tilda_predict = state['Q_cons']
            Q_tilda_predict_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_tilda_predict)

            # Find FOM at sampled points

            sampling_adapt_freq = solver_param['unsampled_update_freq']

            if sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0:
                
                Q_bar_star_old = state['Q_bar']
                solver_param['hyper'] = False
                solver_param['dt'] = sampling_adapt_freq * solver_param['dt']

                state['Q_cons'] = Q_bar_star_old

                state = solver_functions.residual_calculator(solver_param,rom_param,state)
                Q_bar_residual_sampled = state['d_flux_dx'][rom_param['S_indx_solver']]
                state = time_integrator_functions.advance_time(solver_param,state)

                Q_bar_star_new = state['Q_cons']
                Q_bar_star_new_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_bar_star_new)

                state['Q_bar'] = Q_bar_star_new

                Q_bar_new_sampling = Q_bar_star_new_solver_int[rom_param['S_indx_solver']]
                Q_bar_new_solver_int = Q_bar_star_new_solver_int
                solver_param['hyper'] = True
                solver_param['dt'] = solver_param['dt'] / sampling_adapt_freq

                                
            else:
                 
                state              = solver_functions.residual_calculator(solver_param,rom_param,state)
                Q_bar_residual_sampled     = state['d_flux_dx']
                state['Q_cons']    = Q_tilda_old_solver_int[rom_param['S_indx_solver']]
                state              = time_integrator_functions.advance_time(solver_param,state)
                Q_bar_new_sampling = state['Q_cons']

                # Estimate FOM at unsampled points using old basis
                Q_bar_new_solver_int                             = q_ref + (denormalizor * (rom_param['basis'] @ Q_red_new))

                # Combine FOM sampled points with FOM Estimation at umsampled
                Q_bar_new_solver_int[rom_param['S_indx_solver']] = Q_bar_new_sampling 


            # Update the basis
            del_basis = ( (normalizor.reshape(-1,1)*(Q_bar_new_solver_int-Q_tilda_predict_solver_int).reshape(-1,1))@(Q_red_new.reshape(1,-1)) ) / (np.linalg.norm(Q_red_new)**2)
            rom_param['basis'] = rom_param['basis'] + del_basis

                
            # Find Q tilda (ROM) with new basis(correction)
            Q_tilda_correct_solver_int= q_ref + (denormalizor * (rom_param['basis'] @ Q_red_new)) 
            Q_tilda_correct_solver_full= solver_functions.solver_add_ghost(solver_param,Q_tilda_correct_solver_int)

            state['Q_cons'] = Q_tilda_correct_solver_full

            rom_param = sample_point_finder(solver_param,rom_param)

            # Update the precompute term of HyperReduction with new basis and samples
            

            # Q_bar_residual = Q_bar_residual_unsampled
            # Q_bar_residual[rom_param['S_indx_solver']] = Q_bar_residual_sampled

            # num_req_samples = len(rom_param['S_indx_user'])

            # error = 0
            # while error < 1e-6:

            #     interp_error      = (Q_bar_residual).reshape(-1,1) - (rom_param['basis']@np.linalg.pinv(rom_param['basis'][rom_param['S_indx_solver']]))@(Q_bar_residual)[rom_param['S_indx_solver']].reshape(-1,1)
            #     interp_error_indx = np.argsort(np.squeeze(interp_error))[::-1]
            #     S_indx_solver     = interp_error_indx[0:num_req_samples]
            #     S_indx_user       = solver2user_indx_converter(S_indx_solver,solver_param['cell_number'])
            #     S_indx_user       = np.sort(np.unique(S_indx_user))
            #     S_indx_solver     = user2solver_indx_converter(S_indx_user,3,solver_param['cell_number'])

            #     num_selected_samples = len(S_indx_user)
            #     counter = 0

            #     while num_selected_samples < num_req_samples:

            #         start_indx = num_req_samples + counter
            #         end_indx   = num_req_samples + counter + 1

            #         new_indx = interp_error_indx[start_indx:end_indx]
            #         S_indx_solver=np.append(S_indx_solver,new_indx)

            #         S_indx_user       = solver2user_indx_converter(S_indx_solver,solver_param['cell_number'])
            #         S_indx_user       = np.sort(np.unique(S_indx_user))
            #         S_indx_solver     = user2solver_indx_converter(S_indx_user,3,solver_param['cell_number'])
            #         num_selected_samples = len(S_indx_user)
            #         counter = counter + 1

            #     rom_param['S_indx_solver'] = S_indx_solver
            #     rom_param['S_indx_user']   = S_indx_user

            #     interp_error_new    = (normalizor*Q_bar_residual).reshape(-1,1) - (rom_param['basis']@np.linalg.pinv(rom_param['basis'][rom_param['S_indx_solver']]))@(normalizor*Q_bar_residual)[rom_param['S_indx_solver']].reshape(-1,1)
            #     error = np.linalg.norm(interp_error_new - interp_error)
            #     print('re-sampling....')
            # pcc                        = hyper_precomputer(rom_param['basis'],S_indx_solver)
            # rom_param['hyper_precompute'] = pcc

            # breakpoint()


    return state, solver_param , rom_param

def multi_adaptive_rom_progress(solver_param,rom_param,state,iter):
    
    if iter <= int(solver_param['init_training_win']):

        if iter != int(solver_param['init_training_win']):
            
            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state = time_integrator_functions.advance_time(solver_param,state)

        elif iter == int(solver_param['init_training_win']):

            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state = time_integrator_functions.advance_time(solver_param,state)

            rom_param = precomputer(solver_param,state)
            rom_param = sample_point_finder(solver_param,rom_param)
            solver_param['hyper'] = False
            # state['Q_bar'] = state['Q_cons']


    else:   
            
            # read basics parameters
            # breakpoint()
            q_ref            = rom_param['q_ref']
            normalizor       = rom_param['normalizor']
            denormalizor     = rom_param['denormalizor']

            # breakpoint()
            # np.save(r'C:\Users\mohag\Desktop\initial_basis.npy',rom_param['basis'])
            # np.save(r'C:\Users\mohag\Desktop\normalizor.npy',normalizor)
            # np.save(r'C:\Users\mohag\Desktop\q_ref.npy',q_ref)

            # Q tilda (ROM) before any update
            Q_tilda_old = state['Q_cons']
            Q_tilda_old_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_tilda_old)

            # Find new Q tilda using old basis (prediction)
            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state , rom_param = red2full_state_calculator(solver_param,rom_param,state)
            # Q_bar_residual_unsampled= state['d_flux_dx']
            

            # new Q_r
            Q_red_new = rom_param['q_red0']

            Q_tilda_predict = state['Q_cons']
            Q_tilda_predict_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_tilda_predict)

            # Find FOM at sampled points

            state['Q_cons'] = Q_tilda_old
            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            # solver_param['dt'] = 10 * solver_param['dt']
            state = time_integrator_functions.advance_time(solver_param,state)
            # solver_param['dt'] = (1/10) * solver_param['dt']
            Q_bar_new_solver_full = state['Q_cons']
            Q_bar_new_solver_int  = solver_functions.solver_eliminate_ghost(solver_param,Q_bar_new_solver_full)

            # sampling_adapt_freq = solver_param['unsampled_update_freq']

            # if sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0:
                
            #     Q_bar_star_old = state['Q_bar']
            #     solver_param['hyper'] = False
            #     solver_param['dt'] = sampling_adapt_freq * solver_param['dt']

            #     state['Q_cons'] = Q_bar_star_old

            #     state = solver_functions.residual_calculator(solver_param,rom_param,state)
            #     Q_bar_residual_sampled = state['d_flux_dx'][rom_param['S_indx_solver']]
            #     state = time_integrator_functions.advance_time(solver_param,state)

            #     Q_bar_star_new = state['Q_cons']
            #     Q_bar_star_new_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_bar_star_new)

            #     state['Q_bar'] = Q_bar_star_new

            #     Q_bar_new_sampling = Q_bar_star_new_solver_int[rom_param['S_indx_solver']]
            #     Q_bar_new_solver_int = Q_bar_star_new_solver_int
            #     solver_param['hyper'] = True
            #     solver_param['dt'] = solver_param['dt'] / sampling_adapt_freq

                                
            # else:
                 
            #     state              = solver_functions.residual_calculator(solver_param,rom_param,state)
            #     Q_bar_residual_sampled     = state['d_flux_dx']
            #     state['Q_cons']    = Q_tilda_old_solver_int[rom_param['S_indx_solver']]
            #     state              = time_integrator_functions.advance_time(solver_param,state)
            #     Q_bar_new_sampling = state['Q_cons']

            #     # Estimate FOM at unsampled points using old basis
            #     Q_bar_new_solver_int                             = q_ref + (denormalizor * (rom_param['basis'] @ Q_red_new))

            #     # Combine FOM sampled points with FOM Estimation at umsampled
            #     Q_bar_new_solver_int[rom_param['S_indx_solver']] = Q_bar_new_sampling 
            # breakpoint()

            # Create a matrix of snapshots with results from past and q bar
            basis_num_row,basis_num_col = np.shape(rom_param['basis'])
            
            window_size = 10

            snapshot_to_taken = np.arange(iter-window_size+1,iter)

            F  = np.zeros((basis_num_row,window_size))

            for indx in range(window_size-1):
                
                # past snapshots
                snapshot_temp_int = solver_functions.results_user2solver_converter(state['cons_results_save'][:,:,snapshot_to_taken[indx]])
                F [:,indx]          = (snapshot_temp_int-q_ref)*normalizor

            F [:,-1]          = (Q_bar_new_solver_int-q_ref)*normalizor

            C = np.linalg.pinv(rom_param['basis']) @ F 
                    
            R = (rom_param['basis'] @ C) - F

            _ , Sv , SrT = np.linalg.svd(R,full_matrices=False)

            Sr = SrT

            CT_inv = np.linalg.pinv(C.T)

            r = window_size

            r = min(r, len(Sv))

            for indx in np.arange(0,r):

                alpha = - R @ Sr[:,indx].reshape(-1,1)
                beta  = CT_inv @ Sr[:,indx].reshape(-1,1)
                del_basis = alpha @ beta.T

                rom_param['basis'] = rom_param['basis'] + del_basis

            rom_param['basis'] , _ = np.linalg.qr(rom_param['basis'])
            
            # Find Q tilda (ROM) with new basis(correction)
            Q_tilda_correct_solver_int= q_ref + (denormalizor * (rom_param['basis'] @ rom_param['basis'].T @ F [:,-1] ))
            Q_tilda_correct_solver_full= solver_functions.solver_add_ghost(solver_param,Q_tilda_correct_solver_int)

            # state['Q_cons'] = Q_tilda_old
            # state = solver_functions.residual_calculator(solver_param,rom_param,state)
            # state , rom_param = red2full_state_calculator(solver_param,rom_param,state)

            state['Q_cons'] = Q_tilda_correct_solver_full
            
            rom_param = sample_point_finder(solver_param,rom_param)

            # Update the precompute term of HyperReduction with new basis and samples
            

            # Q_bar_residual = Q_bar_residual_unsampled
            # Q_bar_residual[rom_param['S_indx_solver']] = Q_bar_residual_sampled

            # num_req_samples = len(rom_param['S_indx_user'])

            # error = 0
            # while error < 1e-6:

            #     interp_error      = (Q_bar_residual).reshape(-1,1) - (rom_param['basis']@np.linalg.pinv(rom_param['basis'][rom_param['S_indx_solver']]))@(Q_bar_residual)[rom_param['S_indx_solver']].reshape(-1,1)
            #     interp_error_indx = np.argsort(np.squeeze(interp_error))[::-1]
            #     S_indx_solver     = interp_error_indx[0:num_req_samples]
            #     S_indx_user       = solver2user_indx_converter(S_indx_solver,solver_param['cell_number'])
            #     S_indx_user       = np.sort(np.unique(S_indx_user))
            #     S_indx_solver     = user2solver_indx_converter(S_indx_user,3,solver_param['cell_number'])

            #     num_selected_samples = len(S_indx_user)
            #     counter = 0

            #     while num_selected_samples < num_req_samples:

            #         start_indx = num_req_samples + counter
            #         end_indx   = num_req_samples + counter + 1

            #         new_indx = interp_error_indx[start_indx:end_indx]
            #         S_indx_solver=np.append(S_indx_solver,new_indx)

            #         S_indx_user       = solver2user_indx_converter(S_indx_solver,solver_param['cell_number'])
            #         S_indx_user       = np.sort(np.unique(S_indx_user))
            #         S_indx_solver     = user2solver_indx_converter(S_indx_user,3,solver_param['cell_number'])
            #         num_selected_samples = len(S_indx_user)
            #         counter = counter + 1

            #     rom_param['S_indx_solver'] = S_indx_solver
            #     rom_param['S_indx_user']   = S_indx_user

            #     interp_error_new    = (normalizor*Q_bar_residual).reshape(-1,1) - (rom_param['basis']@np.linalg.pinv(rom_param['basis'][rom_param['S_indx_solver']]))@(normalizor*Q_bar_residual)[rom_param['S_indx_solver']].reshape(-1,1)
            #     error = np.linalg.norm(interp_error_new - interp_error)
            #     print('re-sampling....')
            # pcc                        = hyper_precomputer(rom_param['basis'],S_indx_solver)
            # rom_param['hyper_precompute'] = pcc

            # breakpoint()


    return state, solver_param , rom_param






        

    