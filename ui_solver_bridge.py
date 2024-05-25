import solver_functions
import non_linear_terms
import time_integrator_functions
import visualization_functions
import rom_functions
import numpy as np
import matplotlib.pyplot as plt
import time


import cProfile, pstats, io
from pstats import SortKey

def driver(self):
    
    # collect all of variables from user interface
    solver_param = solver_functions.solver_parameters_collector(self)

    # initialize main states 
    state = solver_functions.initialize_state(solver_param)

    # get the initial condition
    state = solver_functions.ic_generator(solver_param,state)

    # add ghost cells
    state = solver_functions.update_ghost_cell(solver_param,state)

    # convert prim to cons
    state = solver_functions.prim2cons_converter(solver_param, state)

    # initialize rom if necessary
    if solver_param['solver_mode'] == 'FOM' or solver_param['solver_mode'] == 'Adaptive ROM':

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

    # # Create a new figure
    # fig_basis = plt.figure()

    # # Add axes to the figure
    # ax_basis = fig_basis.add_subplot(1, 1, 1)  # 1 row, 1 column, first plot

    # begin simulation
    start_time = time.time()

    for iter in range(solver_param['num_step']):

        solver_param['iter'] = iter

        if solver_param['solver_mode'] == 'FOM':

            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state = time_integrator_functions.advance_time(solver_param,state)

        elif solver_param['solver_mode'] == 'ROM':
            
            state = solver_functions.residual_calculator(solver_param,rom_param,state)
            state , rom_param = rom_functions.red2full_state_calculator(solver_param,rom_param,state)

        elif solver_param['solver_mode'] == 'Adaptive ROM':

            # pr = cProfile.Profile()
            # pr.enable()

            state, solver_param , rom_param  = rom_functions.adaptive_rom_progress(solver_param,rom_param,state,iter)
            # if iter >=10:
            #     ax_basis.cla()
            #     for i in range(9):
                    
            #         ax_basis.plot(rom_param['basis'][0:500,i],label='mode' + str(i))
            #         ax_basis.legend()
            #         ax_basis.set_ylim([-1,1])

            # pr.disable()
            # s = io.StringIO()
            # sortby = SortKey.CUMULATIVE
            # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            # ps.print_stats()
            # print(s.getvalue())
            
        # convert cons to prim
        state = solver_functions.cons2prim_converter(solver_param,state)

        state = solver_functions.update_ghost_cell(solver_param,state)

        # visualization
        visualization_functions.in_progress_plot(fig,axs,iter,solver_param,rom_param,state,visual_param)
        
        plt.show(block=False)

        # if iter == 59:

        #     breakpoint()
        #     for i in range(0,59):
        #         plt.plot(solver_param['x'],state['cons_results_save'][0,:,i],label=i)
        #         plt.legend()
        #         plt.show()
        #         time.sleep(0.01)
        print('Iteration: ' + str(iter+1))
        state['cons_results_save'][:,:,iter] = solver_functions.results_solver2user_converter(solver_param['cell_number'],[state['Q_cons']])[:,2:-2]
        state['prim_results_save'][:,:,iter] = solver_functions.results_solver2user_converter(solver_param['cell_number'],[state['Q_prim']])[:,2:-2]  

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

    # prepare the name for the files to be saved
    work_dir   = solver_param['working_dir']

    if solver_param['solver_mode'] == 'FOM':

        save_title = solver_param['solver_mode'] + ' ' + str(solver_param['num_step']) + ' ' + 'snapshots' + ' ' +solver_param['time_scheme'] 

    elif solver_param['solver_mode'] == 'ROM' or solver_param['solver_mode'] == 'Adaptive ROM':

        save_title = solver_param['solver_mode'] + ' ' + str(solver_param['num_step']) + ' ' + 'snapshots' + ' ' + solver_param['time_scheme'] + ' ' + solver_param['rom_method']

        if solver_param['hyper'] == True:

            save_title = save_title + '' + solver_param['sampling_method']

    print('Saving resutls into working directory')

    # save the results and end the simulation
    np.save( work_dir + '/' +save_title + ' cons.npy' ,state['cons_results_save'])
    np.save( work_dir + '/' +save_title + ' prim.npy' ,state['prim_results_save'])

    print('Simulation successfully completed !')



         

 
        






        







        


        
