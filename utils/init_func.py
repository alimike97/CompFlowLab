import os
import shutil
import numpy as np
import cantera as ct

from utils import reshape_func

def init_state(solver_param):

    # create state variable
    state = {}

    if solver_param['gas_model']  == 'Air':

        solver_param['num_species']   = 0
        solver_param['num_prim_var']  = 4
        solver_param['num_state_var'] = 3 

    else:

        solver_param['num_prim_var']  = 4 + solver_param['num_species']
        solver_param['num_state_var'] = solver_param['num_prim_var'] - 1 # no temp

        state['gas']         = ct.Solution(solver_param['gas_model'])
        state['gas'].basis   = 'mass'
        state['gas_array']   = ct.SolutionArray(state['gas'],(1,int(solver_param['cell_number'])+4))

    num_state_var = solver_param['num_state_var']
    num_prim_var  = solver_param['num_prim_var']

    state['Q_cons']                 = np.zeros((num_state_var*int(solver_param['cell_number'])+(num_state_var*4)))
    state['Q_prim']                 = np.zeros((num_prim_var *int(solver_param['cell_number'])+(num_prim_var*4)))

    state['Q_cons_old']             = state['Q_cons']

    state['cons_results_save']      = np.zeros(( num_state_var , int(solver_param['cell_number']) ))
    state['res_save']               = np.zeros(( num_state_var , int(solver_param['cell_number']) ))

    if solver_param['gas_model']  == 'Air': 

        state['prim_results_save']      = np.zeros(( num_prim_var  , int(solver_param['cell_number']) ))
    
    else : 

        # heat release is also plotted part of prim when there is a combustion case
        state['prim_results_save']      = np.zeros(( num_prim_var+1  , int(solver_param['cell_number'])))

    return state

def init_injection(solver_param,state):

    # prepare the number of cells that will be influenced by injection
    state['injection_add_final'] = int(solver_param['non_inj_portion'] * solver_param['cell_number'])
    state['injection_sub_init']  = int(solver_param['non_inj_tail_portion'] * solver_param['cell_number'])
    
    return state

def init_physics(solver_param):

    if solver_param['gas_model']  == 'Air':
        
        import physics.Ideal_Air as physics_module

    else:

        import physics.Reacting_Flow as physics_module

    return physics_module

def init_time_integration(solver_param):

    if solver_param['time_scheme'] == 'FDF':

        import time_integration.FDF as time_integration

    if solver_param['time_scheme'] == 'BDF':

        import time_integration.BDF as time_integration

    if solver_param['time_scheme'] == 'SSPRK2':

        import time_integration.SSPRK2 as time_integration

    if solver_param['time_scheme'] == 'SSPRK3':

        import time_integration.SSPRK3 as time_integration

    return time_integration

def init_solver(solver_param,state):

    if solver_param['solver_mode'] == 'FOM':

        import solver.FOM as solver_module

        rom_param = {}
        solver_param['hyper'] =  False

    elif solver_param['solver_mode'] == 'ROM':

        import solver.ROM as solver_module

        rom_param   = solver_module.precomputer(solver_param)
        state['qr'] = rom_param['qr0']

        if solver_param['hyper']:

            from rom.sampling_func import hyper_precompute

            rom_param = hyper_precompute(solver_param,rom_param)
            
    elif solver_param['solver_mode'] == 'PROM':

        import solver.PROM as solver_module

        rom_param   = solver_module.precomputer(solver_param)
        state['qr'] = rom_param['qr0']

        if solver_param['hyper']:

            from rom.sampling_func import hyper_precompute

            rom_param = hyper_precompute(solver_param,rom_param)
            rom_param['hyper_precompute'] = rom_param['basis'].T @ rom_param['hyper_precompute']

    elif solver_param['solver_mode'] == 'NLROM':

        import solver.NLROM as solver_module

        rom_param   = solver_module.precomputer(solver_param)
        state['qr'] = rom_param['qr0']

        if solver_param['hyper']:

            from rom.sampling_func import hyper_precompute

            rom_param = hyper_precompute(solver_param,rom_param)
            rom_param['hyper_precompute'] = rom_param['basis'].T@rom_param['hyper_precompute']

    elif solver_param['solver_mode'] == 'QROM':

        import solver.QROM as solver_module

        rom_param   = solver_module.precomputer(solver_param)
        state['qr'] = rom_param['qr0']

        if solver_param['hyper']:

            from rom.sampling_func import hyper_precompute

            rom_param = hyper_precompute(solver_param,rom_param)
            rom_param['hyper_precompute'] = rom_param['basis'].T@rom_param['hyper_precompute']

    elif solver_param['solver_mode'] == 'AROM':

        import solver.AROM as solver_module

        rom_param = {}
        solver_param['hyper']              =  False
        solver_param['FOM2ROM_trans_iter'] = solver_param['init_training_win']

    elif solver_param['solver_mode'] == 'SAROM':

        import solver.SAROM as solver_module
        import solver.AROM  as AROM_solver_module

        rom_param = {}
        solver_param['hyper']              = False
        solver_param['FOM2ROM_trans_iter'] = solver_param['init_training_win']

        solver_param['AROM_solver']        = AROM_solver_module

        # initialize all of the modes in this method
        solver_param['SA_Mode_AROM_Training']      = True
        solver_param['SA_Mode_AROM_Transition']    = False
        solver_param['SA_Mode_SROM_Search']        = False
        solver_param['SA_Mode_SROM_Training']      = False
        solver_param['SA_Mode_SROM_Transition']    = False
        solver_param['SA_Mode_SROM']               = False

    elif solver_param['solver_mode'] == 'DACAROM':

        import solver.DACAROM as solver_module

        import solver.AROM  as AROM_solver_module
        import solver.ROM   as ROM_solver_module

        rom_param = {}
        solver_param['hyper']              =  False
        solver_param['FOM2ROM_trans_iter'] = solver_param['init_training_win']

        solver_param['AROM_solver']        = AROM_solver_module
        solver_param['ROM_solver']         = ROM_solver_module

    return solver_module , rom_param

def ic_generator(solver_param,state):

    # already ic profile given
    if np.size(solver_param['ic_data']) == solver_param['num_prim_var'] * int(solver_param['cell_number']):

        full_ic_profile = reshape_func.solver_add_ghost(solver_param['cell_number'],solver_param['num_prim_var'],solver_param['ic_data'].ravel())

        state['Q_prim'] = full_ic_profile

    else:

        num_region = len(solver_param['ic_data'])
        num_cell   = solver_param['cell_number']
        x          =  np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number'])+4 )

        rho = np.zeros(num_cell+4)
        vx  = np.zeros(num_cell+4)
        P   = np.zeros(num_cell+4)
        T   = np.zeros(num_cell+4)
        
        for region in range(0,num_region):

            indx = np.where(  (x >= float(solver_param['ic_data'][region][0]))  &  (x <= float(solver_param['ic_data'][region][1]))  )
        
            rho[indx] = eval(solver_param['ic_data'][region][2])
            vx [indx] = eval(solver_param['ic_data'][region][3])
            P  [indx] = eval(solver_param['ic_data'][region][4])
            T  [indx] = eval(solver_param['ic_data'][region][5])

        state['Q_prim'] = np.vstack((rho,vx,P,T)).ravel()

        state['MW_gas'] = 28.97

        ### In Reacting Flows ###

        if solver_param['gas_model'] != 'Air':

            MF = np.zeros( (solver_param['num_species'],num_cell+4) )

            for region in range(0,num_region):

                indx = np.where(  (x >= float(solver_param['ic_data'][region][0]))  &  (x <= float(solver_param['ic_data'][region][1]))  )

                MF_list = eval(solver_param['ic_data'][region][6])
                MF_np   = np.array(MF_list)

                for cell in indx[0]:

                    MF[:,cell] = MF_np
                
            state['Q_prim'] = np.vstack((rho,vx,P,T,MF)).ravel()

    return state

def init_dir(solver_param):

    dir_results = os.path.join(solver_param['working_dir'], f"{solver_param['solver_mode']}_results")
    solver_param['dir_results'] = dir_results

    # Check if the directory exists
    if os.path.exists(dir_results):

        # Remove the entire directory
        shutil.rmtree(dir_results)

    # Create a new directory
    os.makedirs(dir_results)

    # create rom related folders if we are running rom
    if solver_param['solver_mode'] != 'FOM':

        os.makedirs( os.path.join(dir_results, 'cons_prim')     )
        os.makedirs( os.path.join(dir_results, 'res')           )
        os.makedirs( os.path.join(dir_results, 'basis')         )
        os.makedirs( os.path.join(dir_results, 'samples_user')  )
        os.makedirs( os.path.join(dir_results, 'samples_solver'))
        os.makedirs( os.path.join(dir_results, 'q_r')           )
        os.makedirs( os.path.join(dir_results, 'q_ref')         )
        os.makedirs( os.path.join(dir_results, 'norm')          )
        os.makedirs( os.path.join(dir_results, 'denorm')        )
        
    # create rom related folders if we are running rom
    if solver_param['save_visual']:

        os.makedirs( os.path.join(dir_results, 'plots')     )