import os
import copy
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

from compflowlab.utils import reshape_func
from compflowlab.boundary_condition import bc_func
from compflowlab.rom.sampling_func import hyper_precompute

def assemble_ensemble_snapshots(solver_param):

    print('Assembling Ensemble Snapshots!')

    list_snapshots = np.arange(solver_param['training_start_iter'],
                                solver_param['training_end_iter'  ],
                                solver_param['training_step_iter' ])

    num_snapshot = len(list_snapshots)
    num_ensmbl   = solver_param['num_ensmbl']

    # one (num_state_var, cell_number, num_snapshot) matrix per ensemble member
    member_snapshots = []

    for k in range(num_ensmbl):

        training_data_cons_k = np.zeros((solver_param['num_state_var'],
                                          solver_param['cell_number'],
                                          num_snapshot))

        for indx, iter in enumerate(list_snapshots):

            file_name_cons = 'ensemble' + str(k) + 'time_step' + str(iter) + '_cons.npy'
            training_data_cons_k[:, :, indx] = np.load(os.path.join(solver_param['training_data_dir'], file_name_cons))

        member_snapshots.append(training_data_cons_k)

    print('Assembling Ensemble Snapshots Completed!')

    return member_snapshots

def precomputer(solver_param):

    print('Initializing Adaptive ROM')

    rom_param = {}

    # list of num_ensmbl matrices, each (num_state_var, cell_number, num_snapshot)
    member_snapshots = assemble_ensemble_snapshots(solver_param)

    # reference profile: mean, across the ensemble, of each member's solution at the transition
    q_ref = np.mean([snapshots[:,:,-1] for snapshots in member_snapshots], axis=0)

    # center each member's own snapshots around the shared ensemble-mean reference
    centered_member_snapshots = [snapshots - q_ref[:,:,np.newaxis] for snapshots in member_snapshots]

    # normalizing factor: mean, across the ensemble, of each member's own normalization factor
    norm_factor_per_member = [np.mean(np.sqrt(np.sum(centered**2, axis=2)), axis=1) for centered in centered_member_snapshots]
    norm_factor             = np.mean(norm_factor_per_member, axis=0)

    # centered_normalized data, per member, then stacked (along the snapshot axis) across the ensemble
    cen_norm_member_snapshots = [centered / norm_factor[:,np.newaxis,np.newaxis] for centered in centered_member_snapshots]
    cen_norm_data              = np.concatenate(cen_norm_member_snapshots, axis=2)

    # data matrix: every ensemble member's snapshots stacked as columns of one tall-thin matrix
    tall_thin_data = cen_norm_data.reshape(-1, cen_norm_data.shape[-1])

    # perform SVD -- one single basis shared across the whole ensemble
    V, S, U = np.linalg.svd(tall_thin_data, full_matrices=False)

    # finalize the basis
    basis = V[:,0:-1]

    # wrap up and exit the function
    denormalizor = np.repeat(norm_factor, solver_param['cell_number'])
    normalizor   = 1/denormalizor

    rom_param['basis']                = basis
    rom_param['q_ref']                = q_ref.ravel()
    rom_param['norm']                 = normalizor
    rom_param['denorm']               = denormalizor
    rom_param['F']                    = tall_thin_data
    rom_param['Q_R']                  = rom_param['basis'].T @ rom_param['F']
    rom_param['qr0']                  = basis.T @ tall_thin_data[:,0]

    return rom_param

def prepare_to_store_FOM(solver_param,state,rom_param):

    # prepare data to save
    state['cons_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],state['Q_cons'])[:,2:-2]
    state['res_save']          = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],state['d_flux_dx'])[:,2:-2]

    if solver_param['gas_model'] == 'Air':

        state['prim_results_save'] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]

    else :
        state['prim_results_save'][:-1,:] = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]
        state['prim_results_save'][-1,:]  = state['heat_release'][2:-2]

    return state

def prepare_to_store_ROM(solver_param,state,rom_param):

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

def enforce_positivity(solver_param,state):

    # the linear POD/GMM-DO reconstruction (unlike the FOM) has no built-in physical
    # realizability constraint, and can hand back negative density/pressure; feeding
    # that into the next residual evaluation produces NaN (sqrt of a negative number
    # in the Roe average) that then poisons everything downstream. Floor both here.
    Q_prim_user = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],state['Q_prim'])

    floor = 1e-6

    Q_prim_user[0,:] = np.maximum(Q_prim_user[0,:], floor)   # density
    Q_prim_user[2,:] = np.maximum(Q_prim_user[2,:], floor)   # pressure

    state['Q_prim'] = reshape_func.results_user2solver_converter(Q_prim_user)

    return state

def results_recorder_FOM(solver_param,state,rom_param=None):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param['dir_results'])
    iter = solver_param['iter']
    save_title = 'ensemble'+ str(solver_param['ensmbl']) + 'time_step' + str(iter)

    # Save the results and end the simulation
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_cons.npy"), state['cons_results_save'])
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_prim.npy"), state['prim_results_save'])
    np.save(os.path.join(dir_results,'res'       ,f"{save_title}_res.npy") , state['res_save'])

def results_recorder_trans(solver_param,state,rom_param=None):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param['dir_results'])
    iter = solver_param['iter']
    save_title = str(iter)+'iteration'

    # Save the results and end the simulation
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_cons.npy"), state['cons_results_save'])
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_prim.npy"), state['prim_results_save'])
    np.save(os.path.join(dir_results,'res'       ,f"{save_title}_res.npy") , state['res_save'])

    # Save rom related parameters
    np.save(os.path.join(dir_results,'basis'         ,f"{save_title}_basis.npy")          , rom_param['basis'])
    np.save(os.path.join(dir_results,'samples_user'  ,f"{save_title}_samples_user.npy")   , rom_param['S_indx_user'])
    np.save(os.path.join(dir_results,'samples_solver',f"{save_title}_samples_solver.npy") , rom_param['S_indx_solver'])
    np.save(os.path.join(dir_results,'q_ref'         ,f"{save_title}_q_ref.npy")          , rom_param['q_ref'])
    np.save(os.path.join(dir_results,'norm'          ,f"{save_title}_norm.npy")           , rom_param['norm'])
    np.save(os.path.join(dir_results,'denorm'        ,f"{save_title}_denorm.npy")         , rom_param['denorm'])

def results_recorder_ROM(solver_param,state,rom_param=None):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param['dir_results'])
    iter = solver_param['iter']
    save_title = str(iter)+'iteration'

    # Save the results and end the simulation
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_cons.npy"), state['cons_results_save'])
    np.save(os.path.join(dir_results,'cons_prim' ,f"{save_title}_prim.npy"), state['prim_results_save'])
    np.save(os.path.join(dir_results,'res'       ,f"{save_title}_res.npy") , state['res_save'])

    # Save rom related parameters
    np.save(os.path.join(dir_results,'basis'         ,f"{save_title}_basis.npy")          , rom_param['basis'])
    np.save(os.path.join(dir_results,'samples_user'  ,f"{save_title}_samples_user.npy")   , rom_param['S_indx_user'])
    np.save(os.path.join(dir_results,'samples_solver',f"{save_title}_samples_solver.npy") , rom_param['S_indx_solver'])

def init_ensemble(solver_param,state,physics):

    #####################################################################
    # Draws one random realization per ensemble member EXACTLY ONCE and #
    # gives every member its own independent copy of the state dict, so #
    # each member can keep advancing from its own previous time step.   #
    #####################################################################

    # Mean Sod initial condition
    rho_mean   = [solver_param['ic_data'][0][2], solver_param['ic_data'][1][2]]
    vel_mean   = [solver_param['ic_data'][0][3], solver_param['ic_data'][1][3]]
    press_mean = [solver_param['ic_data'][0][4], solver_param['ic_data'][1][4]]
    temp_mean  = [solver_param['ic_data'][0][5], solver_param['ic_data'][1][5]]

    # Standard deviations
    rho_std   = [solver_param['rho_std'][0]  , solver_param['rho_std'][1]]   # ~5% perturbation
    vel_std   = [solver_param['vel_std'][0]  , solver_param['vel_std'][1]]
    press_std = [solver_param['press_std'][0], solver_param['press_std'][1]]
    temp_std  = [solver_param['temp_std'][0] , solver_param['temp_std'][1]]

    x = np.linspace(solver_param['x_initial'], solver_param['x_final'], solver_param['cell_number'])

    rng = np.random.default_rng(42)

    ensemble = []

    for k in range(solver_param['num_ensmbl']):

        solver_param['ensmbl'] = k

        # Draw one realization of left/right states
        rho_L, rho_R = rng.normal(rho_mean  , rho_std)
        u_L,   u_R   = rng.normal(vel_mean  , vel_std)
        p_L,   p_R   = rng.normal(press_mean, press_std)
        T_L,   T_R   = rng.normal(temp_mean , temp_std)

        # Construct spatial profiles
        rho_profile   = np.where(x < float(solver_param['ic_data'][0][1]), rho_L, rho_R)
        vel_profile   = np.where(x < float(solver_param['ic_data'][0][1]), u_L,   u_R)
        press_profile = np.where(x < float(solver_param['ic_data'][0][1]), p_L, p_R)
        temp_profile  = np.where(x < float(solver_param['ic_data'][0][1]), T_L, T_R)

        q_prim = np.vstack((rho_profile,
                             vel_profile,
                             press_profile,
                             temp_profile))

        # give this ensemble member its own independent copy of the state dict
        state_k = copy.copy(state)

        state_k['Q_prim'] = reshape_func.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var']+1,q_prim.ravel())
        state_k = physics.prim2cons_converter(solver_param,state_k)

        ensemble.append(state_k)

    return ensemble

def advance_one_time_step(solver_param,state,physics,time_integration,rom_param=None):

    #############################################
    # This function is taking one time step     #
    # using adaptive ROM algorithm. It will run #
    # FOM initially but then will turn into ROM #
    # with evolving basis and sample adaptation #
    #############################################

    # build the ensemble's initial conditions exactly once; every member then keeps
    # advancing from its own previous time step instead of being re-initialized here
    if 'ensemble' not in state:

        state['ensemble'] = init_ensemble(solver_param,state,physics)

    iter = solver_param['iter']

    # inputs to the basis time derivative V_dot: every member's starting state (normalized
    # and centered, to get its q_r) and its full-domain residual at that state (F)
    state_norm_list = []
    resid_norm_list = []

    # post-transition members only predict here; each member's correction (projecting its
    # new snapshot through the basis) is deferred until after the basis itself is updated
    # below, so it uses the fresh basis rather than the one from before this time step
    pass1_predictions = []

    # advance every ensemble member through this time step, each starting from its
    # own state left over from the previous time step, before iter is incremented
    for k in range(solver_param['num_ensmbl']):

        solver_param['ensmbl'] = k

        # this member's own state, carried over from the previous time step
        state_k = state['ensemble'][k]

        if iter <= int(solver_param['FOM2ROM_trans_iter']):

            if iter != int(solver_param['FOM2ROM_trans_iter']):

                # take FOM step for initial training
                state_k = physics.residual_calculator(solver_param,rom_param,state_k)
                state_k = time_integration.advance_time(solver_param,rom_param,state_k,physics)

                # post process part
                if solver_param['injection']:

                    state_k = physics.injection_correction(solver_param,state_k)

                # update prim state
                state_k = physics.cons2prim_converter(solver_param,state_k)

                # update the ghost cells
                state_k = bc_func.update_ghost_cell(solver_param,state_k)

                # update prim state
                state_k = physics.prim2cons_converter(solver_param,state_k)

                # prepare results to save
                state_k = prepare_to_store_FOM(solver_param,state_k,rom_param)

                # save solution
                results_recorder_FOM(solver_param,state_k,rom_param)

            elif iter == int(solver_param['FOM2ROM_trans_iter']):

                # take this member's final training-window FOM step
                state_k = physics.residual_calculator(solver_param,rom_param,state_k)
                state_k = time_integration.advance_time(solver_param,rom_param,state_k,physics)

                # post process part
                if solver_param['injection']:

                    state_k = physics.injection_correction(solver_param,state_k)

                # update prim state
                state_k = physics.cons2prim_converter(solver_param,state_k)

                # update the ghost cells
                state_k = bc_func.update_ghost_cell(solver_param,state_k)

                # update prim state
                state_k = physics.prim2cons_converter(solver_param,state_k)

                # prepare results to save
                state_k = prepare_to_store_FOM(solver_param,state_k,rom_param)

                # persist this member's transition-step snapshot alongside the rest of its
                # training window, so it feeds the shared ensemble basis built below
                results_recorder_FOM(solver_param,state_k,rom_param)

        else:

            # read basic parameters
            q_ref                  = rom_param['q_ref']
            normalizor             = rom_param['norm']
            denormalizor           = rom_param['denorm']

            # Q tilda (ROM) before any update
            Q_tilda_old            = state_k['Q_cons']
            Q_tilda_old_solver_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_old)

            # every member always takes its own small (hyper-reduced) ROM step; the large
            # FOM advance that used to happen here per member now happens ONCE, for a single
            # separate truth trajectory (see below) -- this ensemble is purely the forecast

            # find new solution only at sampled points
            state_k['Q_cons']  = Q_tilda_old
            state_k            = physics.residual_calculator(solver_param,rom_param,state_k)
            state_k['Q_cons']  = Q_tilda_old_solver_int[rom_param['S_indx_solver']]
            state_k            = time_integration.advance_time(solver_param,rom_param,state_k,physics)
            Q_bar_new_sampling = state_k['Q_cons']

            # Estimate full-state at unsampled points using old basis (DEIM Equation) -- PREDICTION STEP
            decen_norm_Q_bar_new_sampling           = normalizor[rom_param['S_indx_solver']]*(Q_bar_new_sampling-q_ref[rom_param['S_indx_solver']])
            C                                       = np.linalg.pinv(rom_param['basis'][rom_param['S_indx_solver']]) @ decen_norm_Q_bar_new_sampling
            Q_bar_new_solver_int                    = q_ref + (denormalizor * (rom_param['basis'] @ C ))

            if solver_param['injection']:

                Q_bar_new_solver_full               = reshape_func.solver_add_ghost(solver_param['cell_number'],
                                                                                    solver_param['num_state_var'],
                                                                                    Q_bar_new_solver_int)

                state_k['Q_cons'] = Q_bar_new_solver_full
                state_k = physics.injection_correction(solver_param,state_k)
                Q_bar_new_solver_full = state_k['Q_cons']

                Q_bar_new_solver_int  = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],
                                                                            solver_param['num_state_var'],
                                                                            Q_bar_new_solver_full)

            # defer the CORRECTION STEP (projecting this snapshot through the basis) until
            # every member has predicted and the basis has been updated (V_dot), below
            pass1_predictions.append((k, state_k, Q_bar_new_solver_int))

            continue

        # persist this member's advanced state so the next time step continues from here
        # (only reached for iter <= FOM2ROM_trans_iter; post-transition members `continue`d
        # above and are instead finalized by the correction step below, after V_dot)
        state['ensemble'][k] = state_k

    # Observation source: advance the single truth trajectory (not the ensemble) with one
    # large FOM step, and probe every member's current forecast state for its full-domain
    # residual -- the input the basis equation (eq. 27) needs. Fig. 2 has exactly one
    # observation per assimilation cycle; the ensemble is the forecast, not the measurement.
    if iter > int(solver_param['FOM2ROM_trans_iter']) and np.any(solver_param['iter'] == solver_param['resample_iter_list']):

        q_ref                = rom_param['q_ref']
        normalizor            = rom_param['norm']
        sampling_adapt_freq  = solver_param['unsampled_update_freq']

        # --- advance the truth trajectory, once, over the interval since its last advance ---
        solver_param['hyper'] = False
        solver_param['dt']    = sampling_adapt_freq * solver_param['dt']

        truth_state = state['truth']
        truth_state = physics.residual_calculator(solver_param,rom_param,truth_state)
        truth_state = time_integration.advance_time(solver_param,rom_param,truth_state,physics)

        if solver_param['injection']:

            truth_state = physics.injection_correction(solver_param,truth_state)

        truth_state = physics.cons2prim_converter(solver_param,truth_state)
        truth_state = enforce_positivity(solver_param,truth_state)
        truth_state = bc_func.update_ghost_cell(solver_param,truth_state)
        truth_state = physics.prim2cons_converter(solver_param,truth_state)

        state['truth']     = truth_state
        rom_param['Q_bar']  = truth_state['Q_cons']   # FGS sampling's reference state

        solver_param['dt'] = solver_param['dt'] / sampling_adapt_freq

        # --- probe every member's fresh PREDICTION (not last iteration's finalized state)
        # for its residual, for V_dot; a residual evaluation only, no time advance, and no
        # change to the member -- the correction step still has to happen after this ---
        for k, state_k, Q_bar_new_solver_int in pass1_predictions:

            solver_param['ensmbl'] = k

            probe_state           = copy.copy(state_k)
            probe_state['Q_cons'] = reshape_func.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_bar_new_solver_int)
            probe_state           = physics.residual_calculator(solver_param,rom_param,probe_state)

            probe_Q_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],probe_state['Q_cons'])

            state_norm_list.append((probe_Q_int - q_ref) * normalizor)
            resid_norm_list.append(reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],probe_state['d_flux_dx']) * normalizor)

        basis_dt = solver_param['dt']

        solver_param['hyper'] = True

    # Forecast step, basis part (paper, eq. 27): the time derivative of the basis is
    #     V_dot = (I - V V^T) (F - F_bar) Q_R^T (Q_R Q_R^T)^{-1}
    # with F the ensemble's residuals, F_bar their mean, and Q_R the ensemble's (centered)
    # q_r coefficients. It needs the residual over the whole domain, so it runs where that
    # exists (large-step FOM); the basis is fixed through the hyper-reduced steps between.
    if len(state_norm_list) > 0:

        V = rom_param['basis']

        F = np.array(resid_norm_list).T
        Z = np.array(state_norm_list).T

        Q_R = V.T @ Z
        Q_R = Q_R - np.mean(Q_R, axis=1, keepdims=True)

        # the s x s matrix Q_R Q_R^T has rank <= K-1 (fewer members than modes), so it is
        # inverted as a pseudo-inverse
        V_dot = (F - np.mean(F, axis=1, keepdims=True)) @ Q_R.T @ np.linalg.pinv(Q_R @ Q_R.T, hermitian=True)
        V_dot = V_dot - V @ (V.T @ V_dot)

        # forward Euler with the step the FOM just took, then re-orthonormalize via SVD
        # (guards against V drifting away from orthonormal -- and blowing up to NaN/Inf --
        # over repeated forward-Euler updates)
        V_new = V + basis_dt * V_dot
        U_new, _, _ = np.linalg.svd(V_new, full_matrices=False)
        rom_param['basis'] = U_new

    # Forecast step, correction part: NOW that the basis is at its final value for this
    # time step, project every post-transition member's predicted snapshot through it and
    # finalize that member's state -- deferred from the per-member loop above so a resample
    # iteration's correction uses the just-updated basis, not the one from before this step
    if len(pass1_predictions) > 0:

        q_ref        = rom_param['q_ref']
        normalizor   = rom_param['norm']
        denormalizor = rom_param['denorm']
        basis        = rom_param['basis']

        for k, state_k, Q_bar_new_solver_int in pass1_predictions:

            solver_param['ensmbl'] = k

            # project this member's new snapshot through the basis -- CORRECTION STEP
            new_snapshot_cent_norm      = (Q_bar_new_solver_int - q_ref) * normalizor
            new_qr                      = basis.T @ new_snapshot_cent_norm
            corrected_cent_norm         = basis @ new_qr
            Q_tilda_correct_solver_int  = q_ref + (denormalizor * corrected_cent_norm)
            Q_tilda_correct_solver_full = reshape_func.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_tilda_correct_solver_int)
            state_k['Q_cons'] = Q_tilda_correct_solver_full

            # post process part
            if solver_param['injection']:

                state_k = physics.injection_correction(solver_param,state_k)

            # update prim state
            state_k = physics.cons2prim_converter(solver_param,state_k)

            # guard against an unphysical reconstruction (see enforce_positivity)
            state_k = enforce_positivity(solver_param,state_k)

            # update the ghost cells
            state_k = bc_func.update_ghost_cell(solver_param,state_k)

            # update prim state
            state_k = physics.prim2cons_converter(solver_param,state_k)

            # prepare results to save
            state_k = prepare_to_store_ROM(solver_param,state_k,rom_param)

            # save the data
            if solver_param['iter'] % solver_param['save_interval'] == 0:

                results_recorder_ROM(solver_param,state_k,rom_param)

            # persist this member's advanced state so the next time step continues from here
            state['ensemble'][k] = state_k

    # once the truth trajectory has taken this iteration's large-step FOM run, assimilate
    # its single observation into the ensemble's GMM-DO stochastic-subspace representation
    if iter > int(solver_param['FOM2ROM_trans_iter']) and np.any(solver_param['iter'] == solver_param['resample_iter_list']):

        basis        = rom_param['basis']
        normalizor   = rom_param['norm']
        denormalizor = rom_param['denorm']

        # x_bar^f: the standing (pre-assimilation) mean field
        x_mean_forecast = rom_param['q_ref']

        # each member's own current forecast state (interior, no ghost) -- the regular
        # small-step ROM continuation just taken above, not a large-step excursion
        Q_forecast_members = [reshape_func.solver_eliminate_ghost(solver_param['cell_number'],
                                                                    solver_param['num_state_var'],
                                                                    state_k['Q_cons'])
                               for state_k in state['ensemble']]

        # forecast realizations in the stochastic subspace, {Phi^f_1, ..., Phi^f_K}
        Phi_forecast = np.array([basis.T @ ((Q_k - x_mean_forecast) * normalizor) for Q_k in Q_forecast_members])

        # fit a GMM to the forecast ensemble, selecting the mixture complexity by BIC
        gmm_seed       = solver_param['gmm_do_seed'] + solver_param['iter']
        max_complexity = min(solver_param['gmm_max_complexity'], solver_param['num_ensmbl'] - 1)

        best_gmm, best_bic = None, np.inf

        for M in range(1, max_complexity + 1):

            gmm = GaussianMixture(n_components=M, covariance_type='full', random_state=gmm_seed)
            gmm.fit(Phi_forecast)
            bic = gmm.bic(Phi_forecast)

            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        weights_f = best_gmm.weights_
        means_f   = best_gmm.means_
        covs_f    = best_gmm.covariances_

        # observation operator: every state variable in obs_state_var_index, at the cells
        # nearest each obs_x_locations -- one observation per (variable, location) pair
        obs_cell_idx = np.array([np.argmin(np.abs(solver_param['x'] - loc)) for loc in solver_param['obs_x_locations']])
        obs_flat_idx = np.array([var_idx * solver_param['cell_number'] + cell_idx
                                  for var_idx  in solver_param['obs_state_var_index']
                                  for cell_idx in obs_cell_idx])

        H_tilde = basis[obs_flat_idx, :]
        R       = (solver_param['obs_std']**2) * np.eye(len(obs_flat_idx))

        # observation: the single truth trajectory's fresh large-step FOM state
        x_mean_truth = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],solver_param['num_state_var'],state['truth']['Q_cons'])
        y            = x_mean_truth[obs_flat_idx]
        y_tilde      = y - x_mean_forecast[obs_flat_idx]

        # Bayesian (Kalman) update of every mixture component
        log_weights_a = np.zeros_like(weights_f)
        mu_hat_a      = np.zeros_like(means_f)
        covs_a        = np.zeros_like(covs_f)

        for j in range(len(weights_f)):

            S_j       = H_tilde @ covs_f[j] @ H_tilde.T + R
            K_tilde_j = covs_f[j] @ H_tilde.T @ np.linalg.inv(S_j)

            mu_hat_a[j] = means_f[j] + K_tilde_j @ (y_tilde - H_tilde @ means_f[j])

            # symmetrize and lightly regularize: floating-point error in the subtraction
            # below can leave the covariance just barely non-PSD, which multivariate_normal
            # sampling further down would otherwise silently mishandle
            covs_a_j    = covs_f[j] - K_tilde_j @ H_tilde @ covs_f[j]
            covs_a[j]   = 0.5*(covs_a_j + covs_a_j.T) + 1e-10*np.eye(covs_a_j.shape[0])

            # log-space: a component whose predicted observation is far from y_tilde can
            # underflow a plain pdf() to exactly 0 for every component at once (0/0 -> NaN
            # in the normalization below); logpdf + a log-sum-exp normalization avoids that
            log_weights_a[j] = np.log(weights_f[j]) + multivariate_normal.logpdf(y_tilde, mean=H_tilde @ means_f[j], cov=S_j, allow_singular=True)

        log_weights_a -= np.max(log_weights_a)
        weights_a       = np.exp(log_weights_a)
        weights_a       = weights_a / np.sum(weights_a)

        # reset intermediate means so the mixture stays zero-mean (DO constraint), and
        # fold the removed offset into the posterior mean field
        mean_offset     = np.sum(weights_a[:,np.newaxis] * mu_hat_a, axis=0)
        means_a         = mu_hat_a - mean_offset
        x_mean_analysis = x_mean_forecast + denormalizor * (basis @ mean_offset)

        rom_param['q_ref'] = x_mean_analysis

        # draw a fresh posterior realization for every ensemble member from the updated GMM
        rng               = np.random.default_rng(gmm_seed)
        component_choice  = rng.choice(len(weights_a), size=solver_param['num_ensmbl'], p=weights_a)

        for k in range(solver_param['num_ensmbl']):

            Phi_a_k = rng.multivariate_normal(means_a[component_choice[k]], covs_a[component_choice[k]])

            Q_cons_analysis_int  = x_mean_analysis + denormalizor * (basis @ Phi_a_k)
            Q_cons_analysis_full = reshape_func.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],Q_cons_analysis_int)

            state_k = state['ensemble'][k]
            state_k['Q_cons'] = Q_cons_analysis_full

            if solver_param['injection']:

                state_k = physics.injection_correction(solver_param,state_k)

            state_k = physics.cons2prim_converter(solver_param,state_k)

            # guard against an unphysical reconstruction (see enforce_positivity) -- the
            # posterior draw from the fitted GMM has no realizability constraint of its own
            state_k = enforce_positivity(solver_param,state_k)

            state_k = bc_func.update_ghost_cell(solver_param,state_k)
            state_k = physics.prim2cons_converter(solver_param,state_k)

            # this member never took a full-domain (hyper=False) residual step this
            # iteration -- its last d_flux_dx is the hyper-reduced one from its own ROM
            # step in pass 1/2, above -- so store it the same way the regular ROM path does
            state_k = prepare_to_store_ROM(solver_param,state_k,rom_param)

            if solver_param['iter'] % solver_param['save_interval'] == 0:

                results_recorder_ROM(solver_param,state_k,rom_param)

            state['ensemble'][k] = state_k

        # refresh hyper-reduction samples against the newly assimilated basis
        rom_param = hyper_precompute(solver_param,rom_param,static_basis=False)

    # once every ensemble member has taken its transition-step FOM step, build the
    # single shared ROM basis from all of their training snapshots together
    if iter == int(solver_param['FOM2ROM_trans_iter']):

        # adjust the training range (inclusive of the transition step just taken)
        solver_param['training_start_iter'] = int(iter-solver_param['init_training_win'])
        solver_param['training_end_iter'  ] = iter + 1
        solver_param['training_step_iter' ] = 1
        solver_param['training_data_dir']   = os.path.join(solver_param['dir_results'], 'cons_prim')

        # build ROM: single basis, ensemble-mean q_ref, ensemble-mean normalization
        rom_param = precomputer(solver_param)

        # create the single truth trajectory that will later provide the one observation
        # per assimilation cycle (Fig. 2), starting from the ensemble-mean transition state;
        # also FGS sampling's reference state
        state['truth']            = copy.copy(state)
        state['truth']['Q_cons']  = np.mean([member['Q_cons'] for member in state['ensemble']], axis=0)
        state['truth']            = physics.cons2prim_converter(solver_param,state['truth'])
        rom_param['Q_bar']        = state['truth']['Q_cons']

        # adjust sampling configuration
        solver_param['hyper'] = True

        # find initial samples
        rom_param = hyper_precompute(solver_param,rom_param,static_basis=False)

        # create list of resampling time steps
        solver_param['resample_iter_list'] = np.arange(solver_param['FOM2ROM_trans_iter'],
                                                    solver_param['num_step'],
                                                    solver_param['unsampled_update_freq'],dtype=int)

        # save ROM related param, reporting the ensemble-mean trajectory that fed it
        ensemble_mean_state = {
            'cons_results_save': np.mean([member['cons_results_save'] for member in state['ensemble']], axis=0),
            'prim_results_save': np.mean([member['prim_results_save'] for member in state['ensemble']], axis=0),
            'res_save':          np.mean([member['res_save']          for member in state['ensemble']], axis=0),
        }
        results_recorder_trans(solver_param,ensemble_mean_state,rom_param)

    # report the ensemble mean as the main state, for visualization/checkpointing code
    # (and the plotting code, which reads *_save, not Q_cons/Q_prim directly) that
    # expects a single field rather than the per-member state['ensemble']
    state['Q_cons']            = np.mean([member['Q_cons']            for member in state['ensemble']], axis=0)
    state['Q_prim']            = np.mean([member['Q_prim']            for member in state['ensemble']], axis=0)
    state['cons_results_save'] = np.mean([member['cons_results_save'] for member in state['ensemble']], axis=0)
    state['prim_results_save'] = np.mean([member['prim_results_save'] for member in state['ensemble']], axis=0)
    state['res_save']          = np.mean([member['res_save']          for member in state['ensemble']], axis=0)

    return solver_param, state, rom_param
