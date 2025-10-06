from scipy.optimize import newton_krylov
from scipy.optimize import NoConvergence

import numpy as np


from utils import reshape_func

def implicit_bdf_residual_calculator(q,q_old,solver_param,rom_param,state,physics):

    # some ROM methods need time integratation only on the sampled elements 
    # this may create issue with computing residuals so the Q_cons is temporarily filled with
    # old Q_cons to prevent raising error but while residual is calculated under the hood, it only updates sampled ones, so no computation efficiency is lost
    # also Q_cons later on is converted back to q to not change the overall computational cost while updating solution

    len_Q_cons_full = (solver_param['cell_number']+4) * solver_param['num_state_var']

    # must be the case when hyper-reduction is used

    if len(state['Q_cons']) != len_Q_cons_full:

        Q_cons_old_int                             = state['cons_results_save'].ravel()

        Q_cons_old_int[rom_param['S_indx_solver']] = q

        Q_cons_old_solver = reshape_func.solver_add_ghost(solver_param['cell_number'],
                                                          solver_param['num_state_var'],
                                                          Q_cons_old_int)
        
        state['Q_cons']   = Q_cons_old_solver

    # must be the case in FOM
    else:

        state['Q_cons'] = q

    
    state           = physics.residual_calculator(solver_param,rom_param,state)
    state['Q_cons'] = q     # rolling back to original value to not increase the size of matrix if hyper-reduction is used
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
            maxiter=20,
            f_tol  = 1e-3,
            method ='gmres'
        )

    except NoConvergence as e:
        # The last iterate is stored in e.args[0]
        sol = e.args[0]
        print("Solver did not converge. Using last iterate instead.")
        state['Q_cons'] = sol

    return state
