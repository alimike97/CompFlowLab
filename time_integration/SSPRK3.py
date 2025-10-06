from utils import reshape_func

def advance_time(solver_param,rom_param,state,physics=None):
    
    # some ROM methods need time integratation only on the sampled elements 
    # this may create issue with computing residuals so the Q_cons is temporarily filled with
    # old Q_cons to prevent raising error but while residual is calculated under the hood, it only updates sampled ones, so no computation efficiency is lost
    # also Q_cons later on is converted back to q to not change the overall computational cost while updating solution

    len_Q_cons_full = (solver_param['cell_number']+4) * solver_param['num_state_var']

    # must be the case when hyper-reduction is used

    if len(state['Q_cons']) != len_Q_cons_full:

        dt    = solver_param['dt']
        Q_old = state['Q_cons']
        dQ_dt = state['d_flux_dx']

        # first solution
        q1 = Q_old + dt * dQ_dt

        # prepare Q_cons for next residual calculations
        Q_cons_old_int                             = state['cons_results_save'].ravel()

        Q_cons_old_int[rom_param['S_indx_solver']] = q1

        Q_cons_old_solver = reshape_func.solver_add_ghost(solver_param['cell_number'],
                                                          solver_param['num_state_var'],
                                                          Q_cons_old_int)
        
        # second solution
        state['Q_cons'] = Q_cons_old_solver
        state           = physics.residual_calculator(solver_param,rom_param,state)
        res1            = state['d_flux_dx']
        q2              = q1 + dt * res1

        # prepare Q_cons for next residual calculations
        Q_cons_old_int                             = state['cons_results_save'].ravel()

        Q_cons_old_int[rom_param['S_indx_solver']] = q2

        Q_cons_old_solver = reshape_func.solver_add_ghost(solver_param['cell_number'],
                                                          solver_param['num_state_var'],
                                                          Q_cons_old_int)
        
        # third solution
        state['Q_cons'] = Q_cons_old_solver
        state           = physics.residual_calculator(solver_param,rom_param,state)
        res2            = state['d_flux_dx']
        q3              = (2*Q_old/3) + (1/3)*(q2 + 0.5 * dt * res2)

        # prepare Q_cons for next residual calculations
        Q_cons_old_int                             = state['cons_results_save'].ravel()

        Q_cons_old_int[rom_param['S_indx_solver']] = q3

        Q_cons_old_solver = reshape_func.solver_add_ghost(solver_param['cell_number'],
                                                          solver_param['num_state_var'],
                                                          Q_cons_old_int)

        # combine solutions (Q_new)
        state['Q_cons'] = Q_cons_old_solver
        state           = physics.residual_calculator(solver_param,rom_param,state)
        res3            = state['d_flux_dx']
        Q_new           = q3 + 0.5 * dt * res3

        state['Q_cons'] = Q_new

    # must be the case in FOM
    else:

        dt              = solver_param['dt']
        Q_old           = state['Q_cons']
        dQ_dt           = state['d_flux_dx']

        # first solution
        q1              = Q_old + 0.5 * dt * dQ_dt

        # second solution
        state['Q_cons'] = q1
        state           = physics.residual_calculator(solver_param,rom_param,state)
        res1            = state['d_flux_dx']
        q2              = q1 + 0.5 * dt * res1

        # third solution
        state['Q_cons'] = q2
        state           = physics.residual_calculator(solver_param,rom_param,state)
        res2            = state['d_flux_dx']
        q3              = (2*Q_old/3) + (1/3)*(q2 + 0.5 * dt * res2)

        # combine solutions (Q_new)
        state['Q_cons'] = q3
        state           = physics.residual_calculator(solver_param,rom_param,state)
        res3            = state['d_flux_dx']
        Q_new           = q3 + 0.5 * dt * res3

        state['Q_cons'] = Q_new

    return state