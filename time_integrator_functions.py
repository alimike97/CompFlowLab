import numpy as np
import solver_functions

def advance_time(solver_param,rom_param,state):
    # time integration
    if solver_param['time_scheme'] == 'Explicit - FD Euler':

        state   = explicit_fd_euler(solver_param,state)

    elif solver_param['time_scheme'] == 'Explicit - SSPRK2':

        state   = explicit_ssp_rk2(solver_param,rom_param,state)

    elif solver_param['time_scheme'] == 'Explicit - SSPRK3':

        state   = explicit_ssp_rk3(solver_param,rom_param,state)

    # elif solver_param['time_scheme'] == 'Explicit - SSPRK3':

    #     state   = explicit_ssp_rk3(solver_param,rom_param,state)

    return state

def explicit_fd_euler( solver_param,state ):

    dt    = solver_param['dt']
    Q_old = state['Q_cons']
    dQ_dt = state['d_flux_dx']
    d_Q   = dt * dQ_dt 
    Q_new = Q_old + d_Q

    state['Q_cons'] = Q_new

    return state

def explicit_rk4( solver_param,rom_param,state ):

    # if solver_param['hyper']==True:
        
    #     breakpoint()

    #     Q_old = np.zeros((solver_param['num_state_var']*int(solver_param['cell_number'])+(solver_param['num_state_var']*4)))

    #     Q_old[rom_param['S_indx_solver']] = state['Q_cons']

    #     d_flux_dx = np.zeros((solver_param['num_state_var']*int(solver_param['cell_number'])+(solver_param['num_state_var']*4)))


    #     d_flux_dx[rom_param['S_indx_solver']] = state['d_flux_dx']


    # else :

    Q_old  = state['Q_cons']

    d_flux_dx = np.zeros((solver_param['num_state_var'] , solver_param['cell_number'] + 4))

    d_flux_dx = np.zeros((solver_param['num_state_var']*int(solver_param['cell_number'])+(solver_param['num_state_var']*4)))

    d_flux_dx = state['d_flux_dx']

    # k1
    dt     = solver_param['dt']
    dQ_dt1 = d_flux_dx
    k1     = dt * dQ_dt1

    # k2
    state['Q_cons'] = Q_old + ((dQ_dt1/2) * (dt/2))
    state  = solver_functions.residual_calculator(solver_param,rom_param,state)
    dQ_dt2 = state['d_flux_dx']
    k2 = dt * dQ_dt2

    # k3
    state['Q_cons'] = Q_old + ((dQ_dt2/2) * (dt/2))
    state = solver_functions.residual_calculator(solver_param,rom_param,state)
    dQ_dt3 = state['d_flux_dx']
    k3 = dt * dQ_dt3

    # k4
    state['Q_cons'] = Q_old + ((dQ_dt3) * (dt))
    state = solver_functions.residual_calculator(solver_param,rom_param,state)
    dQ_dt4 = state['d_flux_dx']
    k4 = dt * dQ_dt4

    Q_new = Q_old + 1/6 * (k1 + 2*k2 + 2*k3 + k4)

    state['Q_cons'] = Q_new

    return state


def explicit_ssp_rk2( solver_param,rom_param,state ):

    if solver_param['solver_mode'] == 'Adaptive ROM' and solver_param['hyper'] == True:

        iter = solver_param['iter']

        q_old_full = state['cons_results_save'][:,:].ravel()

        q_old = q_old_full[rom_param['S_indx_solver']]

        q_old_full = solver_functions.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],q_old_full)


        d_flux_dx = state['d_flux_dx']

        dt     = solver_param['dt']

        # q1
        res_old = d_flux_dx
        q1      = q_old + dt * res_old

        # q2
        state['Q_cons'] = q_old_full
        state['Q_cons'][rom_param['S_indx_solver']] = q1
        state  = solver_functions.residual_calculator(solver_param,rom_param,state)
        res1 = state['d_flux_dx']
        q2 = q1 + dt * res1

        # q_new
        q_new = q_new = 0.5*(q1+q2)

        state['Q_cons'] = q_new

    else: 

        q_old = state['Q_cons']

        d_flux_dx = state['d_flux_dx']

        dt     = solver_param['dt']

        # q1
        res_old = d_flux_dx
        q1     = q_old + dt * res_old

        # q2
        state['Q_cons'] = q1
        state  = solver_functions.residual_calculator(solver_param,rom_param,state)
        res1 = state['d_flux_dx']
        q2 = q1 + dt * res1

        # q_new
        q_new = 0.5*(q1+q2)

        state['Q_cons'] = q_new

    return state

def explicit_ssp_rk3( solver_param,rom_param,state ):

    if solver_param['solver_mode'] == 'Adaptive ROM' and solver_param['hyper'] == True:

        iter = solver_param['iter']

        q_old_full = state['cons_results_save'][:,:].ravel()

        q_old = q_old_full[rom_param['S_indx_solver']]

        q_old_full = solver_functions.solver_add_ghost(solver_param['cell_number'],solver_param['num_state_var'],q_old_full)


        d_flux_dx = state['d_flux_dx']

        dt     = solver_param['dt']

        # q1
        res_old = d_flux_dx
        q1     = q_old + 0.5 * dt * res_old

        # q2
        state['Q_cons'] = q_old_full
        state['Q_cons'][rom_param['S_indx_solver']] = q1
        state  = solver_functions.residual_calculator(solver_param,rom_param,state)
        res1 = state['d_flux_dx']
        q2 = q1 + 0.5 * dt * res1

        # q3
        state['Q_cons'] = q_old_full
        state['Q_cons'][rom_param['S_indx_solver']] = q2
        state = solver_functions.residual_calculator(solver_param,rom_param,state)
        res2 = state['d_flux_dx']
        q3 = (2*q_old/3) + (1/3)*(q2 + 0.5 * dt * res2)

        # q_new
        state['Q_cons'] = q_old_full
        state['Q_cons'][rom_param['S_indx_solver']] = q3
        state = solver_functions.residual_calculator(solver_param,rom_param,state)
        res3 = state['d_flux_dx']
        q_new = q3 + 0.5 * dt * res3

        state['Q_cons'] = q_new

    else: 

        q_old = state['Q_cons']

        d_flux_dx = state['d_flux_dx']

        dt     = solver_param['dt']

        # q1
        res_old = d_flux_dx
        q1     = q_old + 0.5 * dt * res_old

        # q2
        state['Q_cons'] = q1
        state  = solver_functions.residual_calculator(solver_param,rom_param,state)
        res1 = state['d_flux_dx']
        q2 = q1 + 0.5 * dt * res1

        # q3
        state['Q_cons'] = q2
        state = solver_functions.residual_calculator(solver_param,rom_param,state)
        res2 = state['d_flux_dx']
        q3 = (2*q_old/3) + (1/3)*(q2 + 0.5 * dt * res2)

        # q_new
        state['Q_cons'] = q3
        state = solver_functions.residual_calculator(solver_param,rom_param,state)
        res3 = state['d_flux_dx']
        q_new = q3 + 0.5 * dt * res3

        state['Q_cons'] = q_new

    return state