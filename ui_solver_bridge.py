import solver_functions
import time_integrator_functions
import visualization_functions
import rom_functions
import numpy as np
import matplotlib.pyplot as plt
import time
import cantera as ct


def driver(self):

    breakpoint()
    
    # collect all of variables from user interface
    solver_param = solver_functions.solver_parameters_collector(self)

    # initialize main states 
    state = solver_functions.initialize_state(solver_param)

    # get the initial condition

    state = solver_functions.ic_generator(solver_param,state)

    # add ghost cells
    # state = solver_functions.update_ghost_cell(solver_param,state)

    # convert prim to cons
    state = solver_functions.prim2cons_converter(solver_param, state)

    # initialize rom if necessary
    if solver_param['solver_mode'] == 'FOM' or solver_param['solver_mode'] == 'Adaptive ROM' or solver_param['solver_mode'] == 'Hybrid ROM':

        S_indx_user               = np.arange(0,solver_param['cell_number'])
        S_indx_solver             = rom_functions.user2solver_indx_converter(S_indx_user,3,solver_param['cell_number'])
        pcc                       = 0
        solver_param['hyper']     = False

        rom_param = {}
        
        rom_param['S_indx_user']      = S_indx_user
        rom_param['S_indx_solver']    = S_indx_solver
        rom_param['hyper_precompute'] = pcc

    elif solver_param['solver_mode'] == 'ROM':

        rom_param = rom_functions.precomputer(solver_param,state)

        if solver_param['hyper'] == True:

            rom_param = rom_functions.sample_point_finder(solver_param,rom_param)

    # create plot
    visual_param = visualization_functions.visual_var_collector(solver_param)

    plt.close()
    fig , axs = plt.subplots(2,2)
    fig.set_size_inches(15,6)
    
    visual_param        = visualization_functions.initial_plot(axs,solver_param,visual_param)

    # begin simulation
    start_time = time.time()
    state['time'] = 0

    for iter in range(solver_param['num_step']):

        solver_param['iter'] = iter

        if solver_param['solver_mode'] == 'FOM':

            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state = time_integrator_functions.advance_time(solver_param,state)

        elif solver_param['solver_mode'] == 'ROM':
            
            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state , rom_param = rom_functions.red2full_state_calculator(solver_param,rom_param,state)

        elif solver_param['solver_mode'] == 'Adaptive ROM':

            state, solver_param , rom_param  = rom_functions.adaptive_rom_progress(solver_param,rom_param,state,iter)

        elif solver_param['solver_mode'] == 'Hybrid ROM':

            state, solver_param , rom_param  = rom_functions.hybrid_rom_progress(solver_param,rom_param,state,iter)

        # convert cons to prim
        state = solver_functions.cons2prim_converter(solver_param,state)

        # update the ghost cells
        state = solver_functions.update_ghost_cell(solver_param,state)
    
        state['time'] = state['time'] + solver_param['dt']

        print('Iteration: ' + str(iter+1))

        state['cons_results_save'][:,:,iter] = solver_functions.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],[state['Q_cons']])[:,2:-2]
        
        if solver_param['gas_model'] == 'Non-Reacting Air':

            state['prim_results_save'][:,:,iter] = solver_functions.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]

        else :
            
            state['prim_results_save'][:-1,:,iter] = solver_functions.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]
            state['prim_results_save'][-1,:,iter] = state['heat_release'][2:-2]


        if (solver_param['solver_mode'] != 'FOM' and iter > int(solver_param['init_training_win'])): 

            state['q_red_save'][:,iter]                                 = rom_param['q_red0']
            state['basis_save'][:,:,iter]                               = rom_param['basis']
            state['S_indx_user_save'][rom_param['S_indx_user'],iter]    = rom_param['S_indx_user']
            state['S_indx_solver_save'][rom_param['S_indx_solver'],iter]= rom_param['S_indx_solver']

        # visualization
        visualization_functions.in_progress_plot(fig,axs,iter,solver_param,rom_param,state,visual_param)
        
        plt.show(block=False)



    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

    # prepare the name for the files to be saved
    work_dir   = solver_param['working_dir']

    if solver_param['solver_mode'] == 'FOM':

        save_title = solver_param['solver_mode'] + ' ' + str(solver_param['num_step']) + ' ' + 'snapshots' + ' ' +solver_param['time_scheme'] 

    elif solver_param['solver_mode'] == 'ROM' or solver_param['solver_mode'] == 'Adaptive ROM' or solver_param['solver_mode'] == 'Hybrid ROM' :

        save_title = solver_param['solver_mode'] + ' ' + str(solver_param['num_step']) + ' ' + 'snapshots' + ' ' + solver_param['time_scheme'] + ' ' + solver_param['rom_method']

        if solver_param['hyper'] == True:

            save_title = save_title + '' + solver_param['sampling_method']

    print('Saving resutls into working directory')

    # save the results and end the simulation
    np.save( work_dir + '/' +save_title + ' cons.npy' ,state['cons_results_save'])
    np.save( work_dir + '/' +save_title + ' prim.npy' ,state['prim_results_save'])

    if solver_param['solver_mode'] != 'FOM':

        np.save( work_dir + '/' +save_title + ' q_r.npy'                  ,state['q_red_save']        )
        np.save( work_dir + '/' +save_title + ' basis.npy'                ,state['basis_save']        )
        np.save( work_dir + '/' +save_title + ' q_ref.npy'                ,state['q_ref_save']        )
        np.save( work_dir + '/' +save_title + ' norm.npy'                 ,state['normalizor_save']   )
        np.save( work_dir + '/' +save_title + ' denorm.npy'               ,state['denormalizor_save'] )
        np.save( work_dir + '/' +save_title + ' samples_user.npy'         ,state['S_indx_user_save']  )
        np.save( work_dir + '/' +save_title + ' samples_solver.npy'       ,state['S_indx_solver_save'])


    print('Simulation successfully completed !')



         

 
        






        







        


        
