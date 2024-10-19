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

        if solver_param['plot_fom_flag'] == True:

            training_data_path_cons = solver_param['FOM_result_dir']
            training_data_path_prim = training_data_path_cons.replace(' cons.npy',' prim.npy')

            training_data_prim      = np.load(training_data_path_prim)

    elif solver_param['solver_mode'] == 'Hybrid ROM':

        iter = solver_param['iter']

        training_data_cons = state['cons_results_save'][:,:,0:iter]
        training_data_prim = state['prim_results_save'][:,:,0:iter]
        first_snapshot     = training_data_cons[:,:,-1]

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


    if solver_param['solver_mode'] == 'Adaptive ROM':

        basis           = V[:,0:-1]

    else:

        basis         = V[:,0:truncation_indx[0][0]]
        # basis         = V[:,0:3]
        # basis           = V
    
    normalizor_size = solver_param['num_state_var']*solver_param['cell_number']

    denormalizor = np.zeros(normalizor_size)

    for indx in range(normalizor_size):
        
        norm_factor_index = int(np.floor(indx/solver_param['cell_number']))

        denormalizor[indx] = norm_factor[norm_factor_index]

    normalizor      = 1/denormalizor
    q_ref           = first_snapshot.ravel(order='C')

    rom_param     = {}

    rom_param['basis']              = basis
    rom_param['normalizor']         = normalizor
    rom_param['denormalizor']       = denormalizor
    rom_param['q_ref']              = q_ref
    rom_param['training_data_prim'] = training_data_prim

    Q_cons_solver = state['Q_cons']
    Q_cons_user   = solver_functions.results_solver2user_converter(solver_param['num_state_var'],cell_num,Q_cons_solver)
    Q_cons_interior_user = Q_cons_user[:,2:-2]
    Q_cons_interior_solver = solver_functions.results_user2solver_converter(Q_cons_interior_user) 
    rom_param['q_red0'] = basis.T @ Q_cons_interior_solver

    num_consv_var = solver_param['num_state_var']
    
    S_indx_user   = np.arange(0,solver_param['cell_number'])
    pcc           = 0
    S_indx_solver = user2solver_indx_converter(S_indx_user,num_consv_var,solver_param['cell_number'])

    rom_param['S_indx_user']      = S_indx_user
    rom_param['S_indx_solver']    = S_indx_solver
    rom_param['hyper_precompute'] = pcc

    state['q_red_save']        = np.zeros((np.shape(rom_param['q_red0'])[0],solver_param['num_step']))
    state['basis_save']        = np.zeros((np.shape(rom_param['basis'])[0],np.shape(rom_param['basis'])[1],solver_param['num_step']))
    state['q_ref_save']        = q_ref
    state['normalizor_save']   = normalizor
    state['denormalizor_save'] = denormalizor
    state['S_indx_user_save']  = np.zeros((np.shape(rom_param['S_indx_user'])[0],solver_param['num_step']))
    state['S_indx_solver_save']= np.zeros((np.shape(rom_param['S_indx_solver'])[0],solver_param['num_step']))

    return rom_param

def hyper_precomputer(basis,S_indx_solver):

    pcc = basis @ np.linalg.pinv(basis[S_indx_solver,:])

    return pcc

def red2full_state_calculator(solver_param,rom_param,state):

    if solver_param['rom_method'] == 'Galerkin':
        
        Q0_red           = rom_param['q_red0']
        q_ref            = rom_param['q_ref']
        normalizor       = rom_param['normalizor']
        denormalizor     = rom_param['denormalizor']
        basis            = rom_param['basis']

        if solver_param['hyper'] == False:

            RES_solver       = state['d_flux_dx']
            RES_user         = solver_functions.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],RES_solver)
            RES              = RES_user[:,2:-2].ravel()

        else:

            pcc              = rom_param['hyper_precompute']
            RES_solver       = state['d_flux_dx']  
            cent_norm        = normalizor[rom_param['S_indx_solver']]*(RES_solver)
            RES_cent_norm    = pcc @ cent_norm
            RES              = (RES_cent_norm*denormalizor)

        dQ_red_dt        = basis.T @ (normalizor * RES)

        state['Q_cons']  = Q0_red

        state['d_flux_dx']= dQ_red_dt

        state            = time_integrator_functions.advance_time(solver_param,state)

        Q_red            = state['Q_cons']

        Q_full_order_solver= q_ref + (denormalizor * (basis @ Q_red)) 

        Q_full_order_user  = solver_functions.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number']-4,Q_full_order_solver)

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

def sampled2unsampled_indx_converter(S_indx_user,num_consv_var,num_cell):

    full_S_indx_user              = np.arange(0,num_cell)
    
    S_star_indx_user              = np.setdiff1d(full_S_indx_user,S_indx_user)

    S_star_indx_solver            = user2solver_indx_converter(S_star_indx_user,num_consv_var,num_cell)

    return S_star_indx_user , S_star_indx_solver

def sample_point_finder(solver_param,rom_param):

    basis = rom_param['basis']

    if solver_param['sampling_method'] == 'DEIM':

            S_indx_user = DEIM_sample_point_finder(basis,solver_param['cell_number'])

    elif solver_param['sampling_method'] == 'QDEIM':
            
            S_indx_user = QDEIM_sample_point_finder(basis,solver_param['cell_number'])

    elif solver_param['sampling_method'] == 'Gappy POD' or solver_param['sampling_method'] == 'Gappy POD + Shock':
            

            # center = np.arange(245,255,1,dtype=int)
            # left        = np.arange(0,10,1,dtype=int)
            # right       = np.arange(490,499,1,dtype=int)

            # S_indx_user = np.concatenate((left,center,right))
            # S_indx_user = np.concatenate((left,center))
            S_indx_user = np.arange(0,solver_param['cell_number'],1)
            # S_indx_user = center

    elif solver_param['sampling_method'] == 'Gappy POD + E':
            
            num_samples = 800

            S_indx_user = GappyPODE_sample_point_finder(basis,num_samples,solver_param['cell_number'])
    
    num_consv_var  = solver_param['num_state_var']
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

def adaptive_rom_progress(solver_param,rom_param,state,iter):

    if solver_param['adaptive_rom_method'] == 'Single-Snapshot':

        state, solver_param , rom_param = single_snapshot_adaptive_rom_progress(solver_param,rom_param,state,iter)

    else:

        state, solver_param , rom_param = multi_snapshot_adaptive_rom_progress(solver_param,rom_param,state,iter)

    return state, solver_param , rom_param

def single_snapshot_adaptive_rom_progress(solver_param,rom_param,state,iter):

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

            # read basic parameters
            q_ref            = rom_param['q_ref']
            normalizor       = rom_param['normalizor']
            denormalizor     = rom_param['denormalizor']

            # Q tilda (ROM) before any update
            Q_tilda_old = state['Q_cons']
            Q_tilda_old_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_tilda_old)

            # Find new Q tilda using old basis (prediction)
            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state , rom_param = red2full_state_calculator(solver_param,rom_param,state)
            
            # new Q_r
            Q_red_new = rom_param['q_red0']

            Q_tilda_predict = state['Q_cons']
            Q_tilda_predict_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_tilda_predict)

            # Find FOM at sampled points
            sampling_adapt_freq = solver_param['unsampled_update_freq']

            # Run a FOM at whole domain to update unsampled points at a specific freq
            if (sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0):

                Q_bar_star_old = state['Q_bar']
                solver_param['hyper'] = False
                solver_param['dt'] = sampling_adapt_freq * solver_param['dt']

                state['Q_cons'] = Q_bar_star_old

                state = solver_functions.residual_calculator(solver_param,rom_param,state)
                state = time_integrator_functions.advance_time(solver_param,state)

                Q_bar_star_new = state['Q_cons']
                Q_bar_star_new_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_bar_star_new)

                state['Q_bar'] = Q_bar_star_new

                Q_bar_new_sampling = Q_bar_star_new_solver_int[rom_param['S_indx_solver']]
                Q_bar_new_solver_int = Q_bar_star_new_solver_int
                solver_param['hyper'] = True
                solver_param['dt'] = solver_param['dt'] / sampling_adapt_freq

                rom_param , F = adapt_basis(solver_param,state,rom_param,Q_bar_new_solver_int,iter,Q_red_new,Q_tilda_predict_solver_int)
            
                # Find Q tilda (ROM) with new basis(correction)

                Q_tilda_correct_solver_int= q_ref + (denormalizor * (rom_param['basis']  @ Q_red_new ))
                
                Q_tilda_correct_solver_full= solver_functions.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_correct_solver_int)
                
                state['Q_cons'] = Q_tilda_correct_solver_full


            # Run ROM at sampled points
            else:

                state['Q_cons']    = Q_tilda_old
                state              = solver_functions.residual_calculator(solver_param,rom_param,state)
                state['Q_cons']    = Q_tilda_old_solver_int[rom_param['S_indx_solver']]
                state              = time_integrator_functions.advance_time(solver_param,state)
                Q_bar_new_sampling = state['Q_cons']

                # Estimate FOM at unsampled points using old basis
                Q_bar_new_solver_int                    = q_ref + (denormalizor * (rom_param['basis'] @ Q_red_new))

                # Combine FOM sampled points with FOM Estimation at umsampled
                Q_bar_new_solver_int[rom_param['S_indx_solver']] = Q_bar_new_sampling

                rom_param , F = adapt_basis(solver_param,state,rom_param,Q_bar_new_solver_int,iter,Q_red_new,Q_tilda_predict_solver_int)
            
                # Find Q tilda (ROM) with new basis(correction)

                Q_tilda_correct_solver_int= q_ref + (denormalizor * (rom_param['basis']  @ Q_red_new ))

                Q_tilda_correct_solver_full= solver_functions.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_correct_solver_int)

                state['Q_cons'] = Q_tilda_correct_solver_full
            
        
            # Update Samples

            if (sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0) or (iter == int(solver_param['init_training_win'])+1):

                ### adapt sample ###

                rom_param,state = adapt_sample(solver_param,rom_param,F,state)

    return state, solver_param , rom_param

def multi_snapshot_adaptive_rom_progress(solver_param,rom_param,state,iter):

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

            # read basic parameters
            q_ref            = rom_param['q_ref']
            normalizor       = rom_param['normalizor']
            denormalizor     = rom_param['denormalizor']

            # Q tilda (ROM) before any update
            Q_tilda_old = state['Q_cons']
            Q_tilda_old_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_tilda_old)
            
            # Find FOM at sampled points
            sampling_adapt_freq = solver_param['unsampled_update_freq']

            # Run a FOM at whole domain to update unsampled points at a specific freq
            # if (sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0) or (iter == int(solver_param['init_training_win'])+1):
            if (sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0):

                Q_bar_star_old = state['Q_bar']
                solver_param['hyper'] = False
                solver_param['dt'] = sampling_adapt_freq * solver_param['dt']

                state['Q_cons'] = Q_bar_star_old

                state = solver_functions.residual_calculator(solver_param,rom_param,state)
                state = time_integrator_functions.advance_time(solver_param,state)

                Q_bar_star_new = state['Q_cons']
                Q_bar_star_new_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_bar_star_new)

                state['Q_bar'] = Q_bar_star_new

                Q_bar_new_sampling = Q_bar_star_new_solver_int[rom_param['S_indx_solver']]
                Q_bar_new_solver_int = Q_bar_star_new_solver_int
                solver_param['hyper'] = True
                solver_param['dt'] = solver_param['dt'] / sampling_adapt_freq

                Q_red_new = rom_param['basis'].T @ Q_bar_star_new_solver_int

                rom_param , F = adapt_basis(solver_param,state,rom_param,Q_bar_new_solver_int,iter,Q_red_new,Q_tilda_predict_solver_int=0)
            
                # Find Q tilda (ROM) with new basis(correction)

                rom_param['q_red0'] = np.linalg.pinv(rom_param['basis']) @ F [:,-1]

                Q_tilda_correct_solver_int= q_ref + (denormalizor * (rom_param['basis'] @ rom_param['q_red0'] ))

                Q_tilda_correct_solver_full= solver_functions.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_correct_solver_int)

                state['Q_cons'] = Q_tilda_correct_solver_full


            # Run ROM at sampled points
            else:

                state['Q_cons']    = Q_tilda_old
                state              = solver_functions.residual_calculator(solver_param,rom_param,state)
                state['Q_cons']    = Q_tilda_old_solver_int[rom_param['S_indx_solver']]
                state              = time_integrator_functions.advance_time(solver_param,state)
                Q_bar_new_sampling = state['Q_cons']

                # Estimate FOM at unsampled points using old basis (DEIM Equation)

                decen_norm_Q_bar_new_sampling           = normalizor[rom_param['S_indx_solver']]*(Q_bar_new_sampling-q_ref[rom_param['S_indx_solver']])
                C                                       = np.linalg.pinv(rom_param['basis'][rom_param['S_indx_solver']]) @ decen_norm_Q_bar_new_sampling
                Q_bar_new_solver_int                    = q_ref + (denormalizor * (rom_param['basis'] @ C ))

                rom_param , F = adapt_basis(solver_param,state,rom_param,Q_bar_new_solver_int,iter,C,Q_tilda_predict_solver_int=0)
            
                # Find Q tilda (ROM) with new basis(correction)

                rom_param['q_red0'] = np.linalg.pinv(rom_param['basis']) @ F [:,-1]

                Q_tilda_correct_solver_int= q_ref + (denormalizor * (rom_param['basis'] @ rom_param['q_red0'] ))
                
                Q_tilda_correct_solver_full= solver_functions.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_correct_solver_int)

                state['Q_cons'] = Q_tilda_correct_solver_full
            
            # Update Samples

            if (sampling_adapt_freq != 0 and solver_param['iter'] % sampling_adapt_freq == 0) or (iter == int(solver_param['init_training_win'])+1):

                ### adapt sample ###

                rom_param,state = adapt_sample(solver_param,rom_param,F,state)

    return state, solver_param , rom_param

def adapt_basis(solver_param,state,rom_param,Q_bar_new_solver_int,iter,Q_red_new,Q_tilda_predict_solver_int):

    q_ref = rom_param['q_ref']
    normalizor = rom_param['normalizor']

    if solver_param['adaptive_rom_method'] == 'Single-Snapshot':

        # Update the basis
        del_basis = ( (normalizor.reshape(-1,1)*(Q_bar_new_solver_int-Q_tilda_predict_solver_int).reshape(-1,1))@(Q_red_new.reshape(1,-1)) ) / (np.linalg.norm(Q_red_new)**2)
        rom_param['basis'] = rom_param['basis'] + del_basis

        F   = np.reshape((Q_bar_new_solver_int-q_ref)*normalizor,(-1,1))

                            
    elif solver_param['adaptive_rom_method'] == 'Multi-Snapshot':

        # Create a matrix of snapshots with results from past and q bar

        basis_num_row,basis_num_col = np.shape(rom_param['basis'])
        
        window_size = basis_num_col+1

        snapshot_to_taken = np.arange(iter-window_size+1,iter)

        F  = np.zeros((basis_num_row,window_size))

        for indx in range(window_size-1):
            
            # past snapshots
            snapshot_temp_int = solver_functions.results_user2solver_converter(state['cons_results_save'][:,:,snapshot_to_taken[indx]])
            F [:,indx]          = (snapshot_temp_int-q_ref)*normalizor


        # future snapshot
        F [:,-1]          = (Q_bar_new_solver_int-q_ref)*normalizor

        # use a window of snapshots to update basis
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

        # orthogonalize the basis
        rom_param['basis'] , _ = np.linalg.qr(rom_param['basis'])        
        

    elif solver_param['adaptive_rom_method'] == 'Direct Adapt':

        # Create a matrix of snapshots with results from past and q bar

        basis_num_row,basis_num_col = np.shape(rom_param['basis'])
        window_size = basis_num_col+1

        snapshot_to_taken = np.arange(iter-window_size+1,iter)

        F  = np.zeros((basis_num_row,window_size))
        Q_R = np.zeros((basis_num_col,window_size))

        for indx in range(window_size-1):
            
            # past snapshots
            snapshot_temp_int = solver_functions.results_user2solver_converter(state['cons_results_save'][:,:,snapshot_to_taken[indx]])
            F [:,indx]          = (snapshot_temp_int-q_ref)*normalizor

            Q_R[:,indx]         = state['q_red_save'][:,snapshot_to_taken[indx]]
        
        # future snapshot
        F [:,-1]          = (Q_bar_new_solver_int-q_ref)*normalizor

        Q_R[:,-1]         = rom_param['basis'].T @ F [:,-1]

        # Q_R = np.linalg.pinv(rom_param['basis'][rom_param['S_indx_solver'],:]) @ F[rom_param['S_indx_solver'],:]

        pinv_Q_R = np.linalg.pinv(Q_R)

        rom_param['basis'] = F @ pinv_Q_R

        # rom_param['basis'][rom_param['S_indx_solver'],:] = F[rom_param['S_indx_solver'],:] @ pinv_Q_R

        new_basis = rom_param['basis']

        # orthogonalize the basis
        # rom_param['basis'] , _ = np.linalg.qr(rom_param['basis'])
        rom_param['basis'] , _ , _ = np.linalg.svd(rom_param['basis'],full_matrices=False)


    elif solver_param['adaptive_rom_method'] == 'Initiative Adapt':

        # Create a matrix of snapshots with results from past and q bar

        basis_num_row,basis_num_col = np.shape(rom_param['basis'])
        
        window_size = basis_num_col+1

        snapshot_to_taken = np.arange(iter-window_size+1,iter)

        F  = np.zeros((basis_num_row,window_size))

        for indx in range(window_size-1):
            
            # past snapshots
            snapshot_temp_int = solver_functions.results_user2solver_converter(state['cons_results_save'][:,:,snapshot_to_taken[indx]])
            F [:,indx]          = (snapshot_temp_int-q_ref)*normalizor

        # future snapshot
        F [:,-1]          = (Q_bar_new_solver_int-q_ref)*normalizor

        rom_param['basis'],_,_ = np.linalg.svd(F , full_matrices=False)

        rom_param['basis'] = rom_param['basis'][:,0:basis_num_col]
        
    
    return rom_param , F

def adapt_sample(solver_param,rom_param,F,state):

    basis = rom_param['basis']

    if solver_param['sampling_method'] == 'DEIM':

            S_indx_user  = DEIM_sample_point_finder(basis,solver_param['cell_number'])
            S_indx_solver= user2solver_indx_converter(S_indx_user,solver_param['num_state_var'],solver_param['cell_number'])

    elif solver_param['sampling_method'] == 'QDEIM':
            
            S_indx_user = QDEIM_sample_point_finder(basis,solver_param['cell_number'])
            S_indx_solver= user2solver_indx_converter(S_indx_user,solver_param['num_state_var'],solver_param['cell_number'])

    elif solver_param['sampling_method'] == 'Gappy POD':

        num_req_samples = len(rom_param['S_indx_user'])

        basis_pinv = np.linalg.pinv(rom_param['basis'][rom_param['S_indx_solver'],:])

        interp_error      = np.abs(F[:,-1] - (rom_param['basis']@basis_pinv)@F[rom_param['S_indx_solver'],-1])
        interp_error_indx = np.argsort(np.squeeze(interp_error))[::-1]
        
        S_indx_solver     = interp_error_indx[0:num_req_samples]
        S_indx_user       = solver2user_indx_converter(S_indx_solver,solver_param['cell_number'])
        S_indx_user       = np.sort(np.unique(S_indx_user))
        S_indx_solver     = user2solver_indx_converter(S_indx_user,solver_param['num_state_var'],solver_param['cell_number'])

        num_selected_samples = len(S_indx_user)
        counter = 0

        while num_selected_samples < num_req_samples:

            start_indx = num_req_samples + counter
            end_indx   = num_req_samples + counter + 1

            new_indx = interp_error_indx[start_indx:end_indx]
            S_indx_solver=np.append(S_indx_solver,new_indx)

            S_indx_user       = solver2user_indx_converter(S_indx_solver,solver_param['cell_number'])
            S_indx_user       = np.sort(np.unique(S_indx_user))
            S_indx_solver     = user2solver_indx_converter(S_indx_user,solver_param['num_state_var'],solver_param['cell_number'])

            num_selected_samples = len(S_indx_user)
            counter = counter + 1

    elif solver_param['sampling_method'] == 'Gappy POD + Shock':

        Q_tilda_correct_solver_full = state['Q_cons']

        # if solver_param['iter'] == 300:

        # breakpoint()

        solver_param['hyper'] = False
        solver_param['dt'] = solver_param['unsampled_update_freq'] * solver_param['dt']

        state['Q_cons'] = state['Q_bar']

        state = solver_functions.residual_calculator(solver_param,rom_param,state)
        state = time_integrator_functions.advance_time(solver_param,state)

        Q_bar_shock = state['Q_cons']
        Q_bar_shock_solver_int = solver_functions.solver_eliminate_ghost(solver_param,Q_bar_shock)
        
        ### future shock ###

        rho_bar_shock = Q_bar_shock_solver_int[0:solver_param['cell_number']]

        # plt.figure()
        # plt.plot(rho_bar_shock,label='future shock')


        second_derv_density = np.gradient(np.gradient(rho_bar_shock))

        deflection_points = np.argsort(second_derv_density)[-2:]

        first_shock   = deflection_points[0]
        shorck_range1 = np.arange(first_shock-5,first_shock+5,1)

        second_shock   = deflection_points[1]
        shorck_range2 = np.arange(second_shock-5,second_shock+5,1)

        shock_range = np.sort(np.unique(np.append(shorck_range1,shorck_range2)))

        # sorted_diff_rho_bar_shock = np.sort(np.where(diff_rho_bar_shock != 0))[0]

        # deflection_points = sorted_diff_rho_bar_shock[0:3]

        # future_shock_head = sorted_diff_rho_bar_shock[-1]
        # future_shock_tail = future_shock_head - 3

        # ### current shock ###

        # current_rho_bar_shock = Q_tilda_correct_solver_int[0:500]

        # current_diff_rho_bar_shock = np.abs(np.diff(current_rho_bar_shock))

        # current_sorted_diff_rho_bar_shock = np.flip(np.argsort(current_diff_rho_bar_shock))

        # current_deflection_points = np.sort(current_sorted_diff_rho_bar_shock[0:5])

        # current_shock_head = current_deflection_points[4]
        # current_shock_tail = current_deflection_points[3]

        ### shock capturing sampling ###

        # shock_range = np.arange(mid_shock_indx-10,mid_shock_indx+10,1)
        # num_req_samples_shock = len(shock_range)

    
        ### normal sampling ###
        # num_req_samples = len(rom_param['S_indx_user'])-num_req_samples_shock
        num_req_samples = 30
        basis_pinv = np.linalg.pinv(rom_param['basis'][rom_param['S_indx_solver'],:])

        interp_error      = np.abs(F[:,-1] - (rom_param['basis']@basis_pinv)@F[rom_param['S_indx_solver'],-1])
        interp_error_indx = np.argsort(np.squeeze(interp_error))[::-1]

        
        S_indx_solver     = interp_error_indx[0:num_req_samples]
        S_indx_user       = solver2user_indx_converter(S_indx_solver,solver_param['cell_number'])
        S_indx_user       = np.sort(np.unique(S_indx_user))

        for indx in S_indx_user:

            if indx in shock_range:

                indices_to_delete = np.where(S_indx_user == indx)
                S_indx_user = np.delete(S_indx_user, indices_to_delete)

        S_indx_solver     = user2solver_indx_converter(S_indx_user,solver_param['num_state_var'],solver_param['cell_number'])

        num_selected_samples = len(S_indx_user)
        counter = 0

        while num_selected_samples < num_req_samples:

            start_indx = num_req_samples + counter
            end_indx   = num_req_samples + counter + 1

            new_indx = interp_error_indx[start_indx:end_indx]
            S_indx_solver=np.append(S_indx_solver,new_indx)

            S_indx_user       = solver2user_indx_converter(S_indx_solver,solver_param['cell_number'])
            S_indx_user       = np.sort(np.unique(S_indx_user))

            for indx in S_indx_user:

                if indx in shock_range:

                    indices_to_delete = np.where(S_indx_user == indx)
                    S_indx_user = np.delete(S_indx_user, indices_to_delete)

            S_indx_solver     = user2solver_indx_converter(S_indx_user,solver_param['num_state_var'],solver_param['cell_number'])

            num_selected_samples = len(S_indx_user)
            counter = counter + 1

        S_indx_user       = np.sort(np.append(S_indx_user,shock_range))
        S_indx_solver     = user2solver_indx_converter(S_indx_user,solver_param['num_state_var'],solver_param['cell_number'])

        solver_param['hyper'] = True
        solver_param['dt'] = solver_param['dt'] / solver_param['unsampled_update_freq']

        state['Q_cons'] = Q_tilda_correct_solver_full

    elif solver_param['sampling_method'] == 'Gappy POD + E':
            
        num_samples = 100

        S_indx_user = GappyPODE_sample_point_finder(basis,num_samples,solver_param['cell_number'])
        S_indx_solver= user2solver_indx_converter(S_indx_user,solver_param['num_state_var'],solver_param['cell_number'])


    rom_param['S_indx_solver'] = S_indx_solver
    rom_param['S_indx_user']   = S_indx_user

    pcc                        = hyper_precomputer(rom_param['basis'],S_indx_solver)
    rom_param['hyper_precompute'] = pcc

    return rom_param , state

def hybrid_rom_progress(solver_param,rom_param,state,iter):

    switch_iter = 30000

    # initially starting with adaptive ROM
    if iter < switch_iter:

        solver_param['solver_mode'] = 'Adaptive ROM'

        state, solver_param , rom_param  = adaptive_rom_progress(solver_param,rom_param,state,iter)

    # transition step - some preprations is needed for switching
    elif iter == switch_iter:
        
        solver_param['solver_mode'] = 'Hybrid ROM'

        rom_param = precomputer(solver_param,state)
        rom_param = sample_point_finder(solver_param,rom_param)

        solver_param['solver_mode'] = 'ROM'

        state = solver_functions.residual_calculator(solver_param,rom_param,state)
        state , rom_param = red2full_state_calculator(solver_param,rom_param,state)

    # keep continue with classical ROM
    elif iter > switch_iter:

        solver_param['solver_mode'] = 'ROM'

        state = solver_functions.residual_calculator(solver_param,rom_param,state)
        state , rom_param = red2full_state_calculator(solver_param,rom_param,state)

    solver_param['solver_mode'] = 'Hybrid ROM'

    return state, solver_param , rom_param 