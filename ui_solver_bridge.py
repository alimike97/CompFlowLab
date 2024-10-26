
import solver_functions
import time_integrator_functions
import visualization_functions
import rom_functions
import numpy as np
import matplotlib.pyplot as plt
import time
import cantera as ct
import os
import shutil


def driver(args,solver_param):

    # breakpoint()
    
    # initialize main states 
    state = solver_functions.initialize_state(solver_param)

    # get the initial condition

    state = solver_functions.ic_generator(solver_param,state)

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

    if solver_param['visual'] == True:

        # create plot
        visual_param = visualization_functions.visual_var_collector(solver_param)

        plt.close()
        fig , axs = plt.subplots(2,2)
        fig.set_size_inches(15,6)
        
        visual_param        = visualization_functions.initial_plot(axs,solver_param,visual_param)

    # create folders for storing data
    dir_results = solver_param['working_dir'] + '\\' + solver_param['solver_mode'] + '_results'

    # Check if the directory exists
    if os.path.exists(dir_results):

        # Remove the entire directory
        shutil.rmtree(dir_results)

    # Create a new directory
    os.makedirs(dir_results)

    # create rom related folders if we are running rom
    if solver_param['solver_mode'] != 'FOM':

        os.makedirs(dir_results+'\\'+'cons_prim'     )
        os.makedirs(dir_results+'\\'+'basis'         )
        os.makedirs(dir_results+'\\'+'samples_user'  )
        os.makedirs(dir_results+'\\'+'samples_solver')
        os.makedirs(dir_results+'\\'+'q_r'           )
        os.makedirs(dir_results+'\\'+'q_ref'         )
        os.makedirs(dir_results+'\\'+'norm'          )
        os.makedirs(dir_results+'\\'+'denorm'        )


    # begin simulation
    start_time = time.time()
    state['time'] = 0

    for iter in range(solver_param['num_step']):

        solver_param['iter'] = iter

        if solver_param['solver_mode'] == 'FOM':

            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state = time_integrator_functions.advance_time(solver_param,rom_param,state)

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

        # prepare data to save
        state['cons_results_save'] = solver_functions.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],[state['Q_cons']])[:,2:-2]
        
        if solver_param['gas_model'] == 'Non-Reacting Air':

            state['prim_results_save'] = solver_functions.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]

        else :
            
            state['prim_results_save'][:-1,:] = solver_functions.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],[state['Q_prim']])[:,2:-2]
            state['prim_results_save'][-1,:]  = state['heat_release'][2:-2]

        # save the data 

        if iter % solver_param['save_interval'] == 0:

            solver_functions.results_recorder(solver_param,rom_param,state)

        if solver_param['visual'] == True:

            # visualization
            visualization_functions.in_progress_plot(fig,axs,iter,solver_param,rom_param,state,visual_param)
            
            plt.show(block=False)

        print('Iteration: ' + str(iter))


    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

    print('Simulation successfully completed!')

    return state



         

 
        






        







        


        
