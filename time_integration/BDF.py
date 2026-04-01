from scipy.optimize import newton_krylov
from scipy.optimize import least_squares
from scipy.optimize import NoConvergence

import numpy as np


from utils import reshape_func
from time_integration import FDF

def implicit_bdf_residual_calculator(q,q_old,solver_param,rom_param,state,physics):

    # some ROM methods need time integratation only on the sampled elements 
    # this may create issue with computing residuals so the Q_cons is temporarily filled with
    # old Q_cons to prevent raising error but while residual is calculated under the hood, it only updates sampled ones, so no computation efficiency is lost
    # also Q_cons later on is converted back to q to not change the overall computational cost while updating solution

    len_Q_cons_full = (solver_param['cell_number']+4) * solver_param['num_state_var']

    # must be the case in FOM
    if len(q) == len_Q_cons_full:

        state           = physics.residual_calculator(solver_param,rom_param,state)
        flux_res        = state['d_flux_dx']
        dt              = solver_param['dt']

        res             = q - q_old - dt * flux_res

        # state['Q_cons'] = q

    # must be the case in ROM
    else:

        num_mode = len(rom_param['basis'][0,:])

        # must be the case when hyper-reduction is used in projected space
        if len(q) == num_mode:

            if solver_param['rom_method'] == 'galerkin':

                qr                = q
                Q_cons_solver_int = rom_param['q_ref'] + rom_param['denorm'] * (rom_param['basis']@qr)

                Q_cons_solver = reshape_func.solver_add_ghost(solver_param['cell_number'],
                                                                solver_param['num_state_var'],
                                                                Q_cons_solver_int)            

                state['Q_cons']   = Q_cons_solver
                state             = physics.residual_calculator(solver_param,rom_param,state)
                # state['Q_cons']   = qr
                cent_denorm_res   = rom_param['norm'][rom_param['S_indx_solver']]*(state['d_flux_dx']-rom_param['q_ref'][rom_param['S_indx_solver']])
                flux_res          = rom_param['hyper_precompute'] @ cent_denorm_res
                dt                = solver_param['dt']
                res               = q - q_old - dt * flux_res

            if solver_param['rom_method'] ==  'lspg':

                Q_cons_old_int    = state['cons_results_save'].ravel()
                qr                = q
                Q_cons_solver_int = rom_param['q_ref'] + rom_param['denorm'] * (rom_param['basis']@qr)

                Q_cons_solver = reshape_func.solver_add_ghost(solver_param['cell_number'],
                                                                solver_param['num_state_var'],
                                                                Q_cons_solver_int)            

                state['Q_cons']   = Q_cons_solver
                state             = physics.residual_calculator(solver_param,rom_param,state)
                dt                = solver_param['dt']
                res               = Q_cons_solver_int[rom_param['S_indx_solver']] - Q_cons_old_int[rom_param['S_indx_solver']] - dt * state['d_flux_dx']

        # must be the case when hyper-reduction is used in full space
        else:

            Q_cons_old_int                             = state['cons_results_save'].ravel()

            Q_cons_old_int[rom_param['S_indx_solver']] = q

            Q_cons_old_solver = reshape_func.solver_add_ghost(solver_param['cell_number'],
                                                            solver_param['num_state_var'],
                                                            Q_cons_old_int)
            
            state['Q_cons']   = Q_cons_old_solver


            state           = physics.residual_calculator(solver_param,rom_param,state)
            # state['Q_cons'] = q     # rolling back to original value to not increase the size of matrix if hyper-reduction is used
            flux_res        = state['d_flux_dx']
            dt              = solver_param['dt']

            res             = q - q_old - dt * flux_res

    return res

def advance_time(solver_param,rom_param,state,physics):

    q_n       = state['Q_cons']
    q_guess   = q_n 

    try:
        sol = newton_krylov(
            lambda q: implicit_bdf_residual_calculator(q, q_guess, solver_param, rom_param, state, physics),
            q_guess,
            maxiter=30,
            f_tol  = 6e-6,
            method ='gmres'
        )

        state['Q_cons'] = sol

    except NoConvergence as e:
        # The last iterate is stored in e.args[0]
        sol = e.args[0]
        print("Solver did not converge. Using last iterate instead.")
        state['Q_cons'] = sol

    return state

def advance_time_lspg(solver_param,rom_param,state,physics):

    q_n       = state['Q_cons']
    # q_guess   = state['Q_cons'] + solver_param['dt'] * state['d_flux_dx']
    q_guess   = q_n

    try:
        # sol = newton_krylov(
        #     lambda q: implicit_bdf_residual_calculator(q, q_guess, solver_param, rom_param, state, physics),
        #     q_guess,
        #     maxiter=30,
        #     f_tol  = 6e-6,
        #     method ='gmres'
        # )

        # state['Q_cons'] = sol

        sol = least_squares(
        lambda q: implicit_bdf_residual_calculator(q, q_guess, solver_param, rom_param, state, physics),
        q_guess,
        method='lm',
        ftol=1e-1,      # was 1e-3
        xtol=1e-1,      # step size — exit as soon as q stops moving
        gtol=1e-1,      # J^T R — most likely to trigger first in LSPG
        max_nfev=5,    # was 50, ~0.4x n_modes is often enough
        )

        state['Q_cons'] = sol.x

    except NoConvergence as e:
        # The last iterate is stored in e.args[0]
        sol = e.args[0]
        print("Solver did not converge. Using last iterate instead.")
        state['Q_cons'] = sol

    return state
