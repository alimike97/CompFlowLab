import os
import numpy as np


from utils import reshape_func
from boundary_condition import bc_func

def precomputer(solver_param):

    print('Initializing ROM')

    rom_param = {}

    training_data_cons = reshape_func.assemble_snapshots(solver_param)

    # number of snapshot
    num_snapshot = len(training_data_cons[0,0,:])

    # reference profile
    q_ref = training_data_cons[:,:,0]

    # center data 
    centered_data = training_data_cons - q_ref[:,:,np.newaxis]

    # normalizing factors
    l2_factors         = np.sqrt(np.sum(centered_data**2, axis=2))
    norm_factor        = np.mean(l2_factors, axis=1)

    # centered_normalized data
    cen_norm_data = centered_data / norm_factor[:, np.newaxis, np.newaxis]

    # data matrix
    tall_thin_data = cen_norm_data.reshape(-1, num_snapshot)

    # perform SVD
    V, S, U = np.linalg.svd(tall_thin_data, full_matrices=False)

    # POD residual energy check
    square_sum_singular_values = np.sum(S**2)
    cumulative_energy          = np.cumsum(S**2)
    POD_res_energy = (1 - (cumulative_energy / square_sum_singular_values)) * 100


    POD_energy_limit_retain  = 100-99.9
    POD_energy_limit_discard = 100-solver_param['pod_energy'] 

    truncation_indx_retain  = np.where(np.array(POD_res_energy) < POD_energy_limit_retain )[0][0]
    truncation_indx_discard = np.where(np.array(POD_res_energy) < POD_energy_limit_discard)[0][0]

    # finalize the basis
    basis_retain  = V[:,0:truncation_indx_retain]
    basis_discard = V[:,truncation_indx_retain:truncation_indx_discard]

    # wrap up and exit the function
    denormalizor = np.repeat(norm_factor, solver_param['cell_number'])
    normalizor   = 1/denormalizor

    rom_param['basis']                = basis_retain
    rom_param['basis_discard']        = basis_discard
    rom_param['q_ref']                = q_ref.ravel()
    rom_param['norm']                 = normalizor
    rom_param['denorm']               = denormalizor
    rom_param['cent_norm_train_data'] = tall_thin_data
    rom_param['qr0']                  = basis_retain.T @ tall_thin_data[:,0]
    rom_param['qr_train_retain']      = basis_retain.T @ tall_thin_data
    rom_param['qr_train_discard']     = basis_discard.T @ tall_thin_data

    rom_param = non_linear_rom_model_trainer(solver_param,rom_param)    

    return rom_param

def prepare_to_store(solver_param,state,rom_param):

    # prepare data to save
    state['cons_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],state['Q_cons'])[:,2:-2]
    state['res_save']          = np.zeros_like(state['cons_results_save'].ravel())-1
    state['res_save'][rom_param['S_indx_solver']] = state['d_flux_dx']
    state['res_save']          = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number']-4,state['res_save'])
    
    if solver_param['gas_model'] == 'Air':

        state['prim_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]

    else :
        state['prim_results_save'][:-1,:] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]
        state['prim_results_save'][-1,:]  = state['heat_release'][2:-2]

    return state

def results_recorder(solver_param,state,rom_param=None):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param['dir_results'])
    iter = solver_param['iter']
    save_title = str(iter)+'iteration'

    # Save the results and end the simulation
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_cons.npy"), state['cons_results_save'])
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_prim.npy"), state['prim_results_save'])
    np.save(os.path.join(dir_results,'res'       ,f"{save_title}_res.npy") , state['res_save'])

    # Save rom related parameters
    if solver_param['iter'] == 0:

        np.save(os.path.join(dir_results,'basis'         ,f"{save_title}_basis.npy")          , rom_param['basis'])
        np.save(os.path.join(dir_results,'samples_user'  ,f"{save_title}_samples_user.npy")   , rom_param['S_indx_user'])
        np.save(os.path.join(dir_results,'samples_solver',f"{save_title}_samples_solver.npy") , rom_param['S_indx_solver'])
        np.save(os.path.join(dir_results,'q_r'           ,f"{save_title}_q_r.npy")            , state['qr'])
        np.save(os.path.join(dir_results,'q_ref'         ,f"{save_title}_q_ref.npy")          , rom_param['q_ref'])
        np.save(os.path.join(dir_results,'norm'          ,f"{save_title}_norm.npy")           , rom_param['norm'])
        np.save(os.path.join(dir_results,'denorm'        ,f"{save_title}_denorm.npy")         , rom_param['denorm'])

def non_linear_rom_model_trainer(solver_param,rom_param):

    # the scikit seems to be very slow when it comes to predict because of multipleoutput regressor
    # its needed to implement everything myself later

    print('Training Nonlinear Error Disclosure!')

    if solver_param['nl_rom_model'] == 'ridge':

        from sklearn.linear_model import Ridge
        from sklearn.multioutput import MultiOutputRegressor

        model = MultiOutputRegressor(Ridge())
        model.fit(rom_param['qr_train_retain'].T
                ,rom_param['qr_train_discard'].T)
        
        rom_param['nl_rom_model'] = model

    elif solver_param['nl_rom_model'] == 'gpr':

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern
        from sklearn.multioutput import MultiOutputRegressor

        kernel = Matern(length_scale=1.0, nu=1.5)
        gpr    = GaussianProcessRegressor(kernel=kernel, random_state=0)
        model  = MultiOutputRegressor(gpr)
        model.fit(rom_param['qr_train_retain'].T
                 ,rom_param['qr_train_discard'].T)
                
        rom_param['nl_rom_model'] = model

    elif solver_param['nl_rom_model'] == 'rbf':

        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, Matern
        from sklearn.multioutput import MultiOutputRegressor

        kernel    = RBF(length_scale=1e-9)
        rbf       = GaussianProcessRegressor(kernel=kernel, random_state=0)
        model     = MultiOutputRegressor(rbf)
        model.fit(rom_param['qr_train_retain'].T
                 ,rom_param['qr_train_discard'].T)
                
        rom_param['nl_rom_model'] = model

    print('Training Nonlinear Error Disclosure Completed!')

    return rom_param

def advance_one_time_step(solver_param,state,physics,time_integration,rom_param=None):

    q_ref            = rom_param['q_ref']
    norm             = rom_param['norm']
    denorm           = rom_param['denorm']
    basis            = rom_param['basis']
    basis_disc       = rom_param['basis_discard']

    # store the previous time step
    Q_tilda_old            = state['Q_cons']
    Q_tilda_old_solver_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_old)

    # find the residual only at sampled points
    state              = physics.residual_calculator(solver_param,rom_param,state)
    state['Q_cons']    = Q_tilda_old_solver_int[rom_param['S_indx_solver']]

    # find the solution only at sampled points
    state              = time_integration.advance_time(solver_param,rom_param,state,physics)
    Q_tilda_new        = state['Q_cons']

    # compute retained qr
    decen_norm_Q_bar_new_sampling = norm[rom_param['S_indx_solver']]*(Q_tilda_new-q_ref[rom_param['S_indx_solver']])
    # qr_retain                     = np.linalg.pinv(basis[rom_param['S_indx_solver'],:]) @ decen_norm_Q_bar_new_sampling
    qr_retain                     = rom_param['hyper_precompute'] @ decen_norm_Q_bar_new_sampling

    # compute discarded qr
    qr_discard                    = rom_param['nl_rom_model'].predict(qr_retain.reshape(1,-1)).T
    qr_discard                    = np.squeeze(qr_discard)

    # find the full-state
    C                             = (basis @ qr_retain) + (basis_disc @ qr_discard)
    Q_bar_new_solver_int          = q_ref + (denorm * C)

    state['Q_cons'] = reshape_func.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_bar_new_solver_int)

    # post process part
    if solver_param['injection']:

        state = physics.injection_correction(solver_param,state)

    # update prim state
    state = physics.cons2prim_converter(solver_param,state)

    # update the ghost cells
    state = bc_func.update_ghost_cell(solver_param,state)

    # update prim state
    state = physics.prim2cons_converter(solver_param,state)

    # prepare results to save
    state = prepare_to_store(solver_param,state,rom_param)

    # save the data 
    if solver_param['iter'] % solver_param['save_interval'] == 0:

        results_recorder(solver_param,state,rom_param)    

    return state