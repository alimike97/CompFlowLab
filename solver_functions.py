import os
import numpy as np
import cantera as ct

def solver_parameters_collector(args,input_param):

    solver_param = {}

    ### solver_mode ###
    if input_param['solver_mode'] == 'FOM':
    
        solver_param['solver_mode'] = 'FOM'

    elif input_param['solver_mode'] == 'ROM':

        solver_param['solver_mode']           = 'ROM'
        solver_param['pod_energy']            = float(input_param['pod_energy'])
        solver_param['rom_method']            = input_param['rom_method']     
        solver_param['hyper']                 = eval(input_param['hyper'])         
        solver_param['sampling_method']       = input_param['hyper_method']
        solver_param['fom_results_max_iter']  = int(input_param['fom_results_max_iter'])

    elif  input_param['solver_mode'] == 'Adaptive ROM':

        solver_param['solver_mode']           = 'Adaptive ROM'
        solver_param['adaptive_rom_method']   = input_param['arom_method']
        solver_param['pod_energy']            = float(input_param['pod_energy'])
        solver_param['init_training_win']     = float(input_param['init_training_win'])
        solver_param['unsampled_update_freq'] = int(input_param['unsampled_update_freq'])
        solver_param['rom_method']            = input_param['rom_method']     
        solver_param['hyper']                 = eval(input_param['hyper'])          
        solver_param['sampling_method']       = input_param['hyper_method']


    elif input_param['solver_mode'] == 'Hybrid ROM':

        solver_param['solver_mode']           = 'Hybrid ROM'
        solver_param['adaptive_rom_method']   = input_param['arom_method']
        solver_param['pod_energy']            = float(input_param['pod_energy'])
        solver_param['init_training_win']     = float(input_param['init_training_win'])
        solver_param['unsampled_update_freq'] = int(input_param['unsampled_update_freq'])
        solver_param['rom_method']            = input_param['rom_method']     
        solver_param['hyper']                 = eval(input_param['hyper'])          
        solver_param['sampling_method']       = input_param['hyper_method']
        

    ### time discretization ###
    solver_param['dt']          = float(input_param['dt'])
    solver_param['num_step']    = int(input_param['num_steps'])
    solver_param['time_scheme'] = input_param['time_scheme']


    ### space discretization ###
    solver_param['x_initial']   = float(input_param['x_initial'])
    solver_param['x_final']     = float(input_param['x_final'])
    solver_param['cell_number'] = int(input_param['cell_number'])

    ### BC data ###
    solver_param['bc_data'] = np.empty((5,2) , dtype='object')
    
    solver_param['bc_data'][0,0] = input_param['rho_inlet']
    solver_param['bc_data'][1,0] = input_param['vel_inlet']
    solver_param['bc_data'][2,0] = input_param['press_inlet']
    solver_param['bc_data'][3,0] = input_param['temp_inlet']
    solver_param['bc_data'][4,0] = input_param['mass_frac_inlet']


    solver_param['bc_data'][0,1] = input_param['rho_outlet']
    solver_param['bc_data'][1,1] = input_param['vel_outlet']
    solver_param['bc_data'][2,1] = input_param['press_outlet']
    solver_param['bc_data'][3,1] = input_param['temp_outlet']
    solver_param['bc_data'][4,1] = input_param['mass_frac_outlet']

    ### ic data ###

    if 'ic_path' in input_param:

        solver_param['ic_data'] = np.load(input_param['ic_path'])

        solver_param['num_species']   = len(solver_param['ic_data'][:,0])-4

    else:

        # Number of rows
        num_rows = len(eval(input_param['x_interval_ic']))

        # Structuring the data
        solver_param['ic_data'] = []

        for i in range(num_rows):

            row = [str(eval(input_param['x_interval_ic'])[i][0]),
                   str(eval(input_param['x_interval_ic'])[i][1]),
                   str(eval(input_param['rho_ic'])[i]),
                   str(eval(input_param['vel_ic'])[i]),
                   str(eval(input_param['press_ic'])[i]),
                   str(eval(input_param['temp_ic'])[i]),
                   str(eval(input_param['mass_frac_ic'])[i])
                   ]
            
            solver_param['ic_data'].append(row)

        solver_param['num_species']   = len(eval(solver_param['ic_data'][0][6]))

    ### physics ###
    solver_param['gas_model']     =  input_param['gas_model']
    solver_param['gamma']         =  1.4 
    solver_param['flux_scheme']   =  input_param['flux_scheme']
    solver_param['limiter']       =  eval(input_param['limiter'])
    solver_param['limiter_method']=  input_param['limiter_method']
    solver_param['viscous_flag']  =  eval(input_param['viscous'])

    ### visualization ###
    solver_param['visual']              = eval(input_param['visual'])
    solver_param['variable1']           = input_param['variable1']
    solver_param['variable2']           = input_param['variable2']
    solver_param['variable3']           = input_param['variable3']
    solver_param['variable4']           = input_param['variable4']
    solver_param['vis_update_interval'] = int(input_param['update_interval'])
    solver_param['plot_fom_flag']       = eval(input_param['plot_fom'])
    
    ### Input Directory ###
    solver_param['working_dir']         = args.working_directory


    ### Saving Data ###
    solver_param['save_interval']       = int(input_param['save_interval'])

    ### Saving Data ###
    solver_param['profiling']           = eval(input_param['profiling'])

    
    ### Some Basic Calculations ###

    solver_param['dx']            = (solver_param['x_final'] - solver_param['x_initial']) / solver_param['cell_number']
    solver_param['vol']           = solver_param['dx']
    solver_param['x']             = np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number']) )

    if solver_param['gas_model']  == 'Non-Reacting Air':

        solver_param['num_species']   = 0
        solver_param['num_prim_var']  = 4
        solver_param['num_state_var'] = 3 

    else:

        
        solver_param['num_prim_var']  = 4 + solver_param['num_species']
        solver_param['num_state_var'] = solver_param['num_prim_var'] - 1 # no temp
    
    return solver_param

def initialize_state(solver_param):

    # create state variable
    state = {}

    num_state_var = solver_param['num_state_var']
    num_prim_var  = solver_param['num_prim_var']
    num_species   = solver_param['num_species']

    state['cons_results_save']      = np.zeros(( num_state_var , int(solver_param['cell_number']) ))
    state['prim_results_save']      = np.zeros(( num_prim_var  , int(solver_param['cell_number']) ))

    state['Q_cons']                 = np.zeros((num_state_var*int(solver_param['cell_number'])+(num_state_var*4)))
    state['Q_prim']                 = np.zeros((num_prim_var *int(solver_param['cell_number'])+(num_prim_var*4)))

    if solver_param['gas_model'] != 'Non-Reacting Air':

        mech_file_dir = solver_param['working_dir']+'/chem.yaml'

        # state['gas']       = ct.Solution(mech_file_dir)
        state['gas']       = ct.Solution('h2o2.yaml')
        state['gas'].basis = 'mass'
        state['gas_array'] = ct.SolutionArray(state['gas'],(1,int(solver_param['cell_number'])+4))

        if solver_param['flux_scheme'] == '2nd Order Roe':
            
            num_cell_with_ghosts = solver_param['cell_number']+4
            num_faces_with_ghosts = num_cell_with_ghosts + 1

            num_subfaces = 2*num_faces_with_ghosts

            state['gas_array_2nd_order'] = ct.SolutionArray(  state['gas'],(1,int(num_subfaces))   )

        # heat release is also plotted part of prim when there is a combustion case
        state['prim_results_save']      = np.zeros(( num_prim_var+1  , int(solver_param['cell_number'])))


    return state

def ic_generator(solver_param,state):

    # already ic profile given

    if np.size(solver_param['ic_data']) == solver_param['num_prim_var'] * int(solver_param['cell_number']):

        full_ic_profile = solver_add_ghost(solver_param['cell_number'],solver_param['num_prim_var'],solver_param['ic_data'].ravel())

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

        if solver_param['gas_model'] != 'Non-Reacting Air':

            MF = np.zeros( (solver_param['num_species'],num_cell+4) )

            for region in range(0,num_region):

                indx = np.where(  (x >= float(solver_param['ic_data'][region][0]))  &  (x <= float(solver_param['ic_data'][region][1]))  )

                MF_list = eval(solver_param['ic_data'][region][6])
                MF_np   = np.array(MF_list)

                for cell in indx[0]:

                    MF[:,cell] = MF_np
                
            state['Q_prim'] = np.vstack((rho,vx,P,T,MF)).ravel()

    return state

def update_ghost_cell(solver_param,state):

    num_cell = solver_param['cell_number']

    Q_prim   = state['Q_prim']
    Q_cons   = state['Q_cons']
    Q_prim_user = results_solver2user_converter(solver_param['num_prim_var'],num_cell,Q_prim)

    if solver_param['gas_model'] != 'Non-Reacting Air':

        num_eqn = 5

    else:

        num_eqn = 4

    for eqn in range(num_eqn):

        inlet_bc  = solver_param['bc_data'][eqn,0]
        outlet_bc = solver_param['bc_data'][eqn,1]

        ################################################## inlet ################################################## 
        
        ##### periodic ##### 

        if inlet_bc == 'periodic':

            Q_prim_user[eqn,0:2] = Q_prim_user[eqn,-2:].reshape(-1,1)

        ##### wall ##### 

        elif inlet_bc == 'wall':
            
            # velocity
            if eqn == 1: 

                Q_prim_user[eqn,0:2] = - Q_prim_user[eqn,2].reshape(-1,1)

            else:

                Q_prim_user[eqn,0:2] = Q_prim_user[eqn,2].reshape(-1,1)

        ##### value/extrap ##### 

        else:
            
            ##### extrap #####

            if inlet_bc == 'extrapolate':

                Q_prim_user[eqn,0:2] = Q_prim_user[eqn,2].reshape(-1,1)

            ##### value ##### 

            else: 
                
                if eqn == 4:

                    if '/' in inlet_bc:
                        
                        parts = inlet_bc.split('/')

                        value = float(parts[0])  # value
                        A     = float(parts[1])  # amplitude
                        f     = float(parts[2])  # frequency

                        t = state['time']

                        Q_prim_user[4:,0:2] = value + A*np.sin(f*t)

                    else:
                    
                        Q_prim_user[4:,0:2] = np.array(eval(inlet_bc)).reshape(-1,1)

                else:

                    if '/' in inlet_bc:
                        
                        parts = inlet_bc.split('/')

                        value = float(parts[0])  # value
                        A     = float(parts[1])  # amplitude
                        f     = float(parts[2])  # frequency

                        t = state['time']

                        Q_prim_user[eqn,0:2] = value + A*np.sin(f*t)


                    else:

                        Q_prim_user[eqn,0:2] = float(inlet_bc)


        ################################################## outlet ################################################## 
        
        ##### periodic ##### 

        if outlet_bc == 'periodic':

            Q_prim_user[eqn,-2:] = Q_prim_user[eqn,0:2].reshape(-1,1)

        ##### wall ##### 

        elif outlet_bc == 'wall':

            # velocity
            if eqn == 1: 

                Q_prim_user[eqn,-2:] = - Q_prim_user[eqn,-3].reshape(-1,1)
                
            else:

                Q_prim_user[eqn,-2:] = Q_prim_user[eqn,-3].reshape(-1,1)

        ##### value/extrap ##### 

        else:
            
            ##### extrap #####

            if outlet_bc == 'extrapolate':

                Q_prim_user[eqn,-2:] = Q_prim_user[eqn,-3].reshape(-1,1)

            ##### value ##### 

            else:

                if eqn == 4:

                    if '/' in outlet_bc:
                        
                        parts = outlet_bc.split('/')

                        value = float(parts[0])  # value
                        A     = float(parts[1])  # amplitude
                        f     = float(parts[2])  # frequency

                        t = state['time']

                        Q_prim_user[4:,-2:] = value + A*np.sin(f*t)

                    else:
                    
                        Q_prim_user[4:,-2:] = np.array(eval(outlet_bc)).reshape(-1,1)

                else:

                    if '/' in outlet_bc:
                        
                        parts = outlet_bc.split('/')

                        value = float(parts[0])  # value
                        A     = float(parts[1])  # amplitude
                        f     = float(parts[2])  # frequency

                        t = state['time']

                        Q_prim_user[eqn,-2:] = value + A*np.sin(f*t)


                    else:
                    
                        Q_prim_user[eqn,-2:] = float(outlet_bc)


        # if outlet_bc == 'periodic':

        #     Q_prim_user[eqn,-2:] = Q_prim_user[eqn,0:2].reshape(-1,1)

        # ##### wall ##### 

        # if inlet_bc == 'wall':
            
        #     # velocity
        #     if eqn == 1: 

        #         Q_prim_user[eqn,0:2] = - Q_prim_user[eqn,2].reshape(-1,1)

        #     else:

        #         Q_prim_user[eqn,0:2] = Q_prim_user[eqn,2].reshape(-1,1)

        # if outlet_bc == 'wall':

        #     # velocity
        #     if eqn == 1: 

        #         Q_prim_user[eqn,-2:] = - Q_prim_user[eqn,-3].reshape(-1,1)
                
        #     else:

        #         Q_prim_user[eqn,-2:] = Q_prim_user[eqn,-3].reshape(-1,1)

        # ##### inlet extrapolate ##### 

        # if inlet_bc == 'extrapolate':

        #     Q_prim_user[eqn,0:2] = Q_prim_user[eqn,2].reshape(-1,1)

        # else:
            
        #     if eqn == 4:

        #         if '/' in inlet_bc:
                    
        #             parts = inlet_bc.split('/')

        #             value = float(parts[0])  # value
        #             A     = float(parts[1])  # amplitude
        #             f     = float(parts[2])  # frequency

        #             t = state['time']

        #             Q_prim_user[4:,0:2] = value + A*np.sin(f*t)

        #         else:
                
        #             Q_prim_user[4:,0:2] = np.array(eval(inlet_bc)).reshape(-1,1)

        #     else:

        #         if '/' in inlet_bc:
                    
        #             parts = inlet_bc.split('/')

        #             value = float(parts[0])  # value
        #             A     = float(parts[1])  # amplitude
        #             f     = float(parts[2])  # frequency

        #             t = state['time']

        #             Q_prim_user[eqn,0:2] = value + A*np.sin(f*t)


        #         else:

        #             Q_prim_user[eqn,0:2] = float(inlet_bc)

        # ##### outlet extrapolate ##### 

        # if outlet_bc == 'extrapolate':

        #     Q_prim_user[eqn,-2:] = Q_prim_user[eqn,-3].reshape(-1,1)

        # else:

        #     if eqn == 4:

        #         if '/' in outlet_bc:
                    
        #             parts = outlet_bc.split('/')

        #             value = float(parts[0])  # value
        #             A     = float(parts[1])  # amplitude
        #             f     = float(parts[2])  # frequency

        #             t = state['time']

        #             Q_prim_user[4:,-2:] = value + A*np.sin(f*t)

        #         else:
                
        #             Q_prim_user[4:,-2:] = np.array(eval(outlet_bc)).reshape(-1,1)

        #     else:

        #         if '/' in outlet_bc:
                    
        #             parts = outlet_bc.split('/')

        #             value = float(parts[0])  # value
        #             A     = float(parts[1])  # amplitude
        #             f     = float(parts[2])  # frequency

        #             t = state['time']

        #             Q_prim_user[eqn,-2:] = value + A*np.sin(f*t)


        #         else:
                 
        #             Q_prim_user[eqn,-2:] = float(outlet_bc)

    
    Q_prim_solver = results_user2solver_converter(Q_prim_user)
    
    state['Q_prim'] = Q_prim_solver
    state = prim2cons_converter(solver_param,state)

    return state
    
def prim2cons_converter(solver_param, state):

    if solver_param['gas_model'] == 'Non-Reacting Air':

        num_cell = solver_param['cell_number']

        Q_prim   = state['Q_prim']

        Q_prim_user = results_solver2user_converter(solver_param['num_prim_var'],num_cell,Q_prim)

        vol = solver_param['vol']

        gamma = solver_param['gamma']

        rho = Q_prim_user[0,:]
        vx  = Q_prim_user[1,:] 
        P   = Q_prim_user[2,:]

        mass   = rho * vol
        momx   = rho * vx * vol
        energy = (P/(gamma-1) + 0.5 * rho * (vx**2)) * vol

        state['Q_cons'] = np.vstack((mass,momx,energy)).ravel()

    else:

        num_cell = solver_param['cell_number']

        Q_prim   = state['Q_prim']

        Q_prim_user = results_solver2user_converter(solver_param['num_prim_var'],num_cell,Q_prim)

        vol = solver_param['vol']

        rho = Q_prim_user[0,:]
        vx  = Q_prim_user[1,:] 
        P   = Q_prim_user[2,:]
        T   = Q_prim_user[3,:]
        Y   = Q_prim_user[4:,:]

        Y_cantera = find_mass_fraction_full_cantera(Y)
        
        state['gas_array'].TPY = T,P,Y_cantera

        internal_energy = np.squeeze(state['gas_array'].int_energy_mass)
    
        internal_energy_tot = internal_energy + (0.5 * (vx **2))

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(internal_energy)
        # plt.plot(internal_energy_tot,linestyle='--')

        mass   = rho * vol
        momx   = rho * vx * vol
        energy = (rho * internal_energy_tot) * vol
        mass_species = rho * Y * vol

        state['Q_cons'] = np.vstack((mass,momx,energy,mass_species)).ravel()     

    return state

def cons2prim_converter(solver_param, state):

    if solver_param['gas_model'] == 'Non-Reacting Air':

        vol         = solver_param['vol']
        gamma       = solver_param['gamma']

        Q_cons      = state['Q_cons']
        Q_cons_user = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

        mass        = Q_cons_user[0,:]
        momx        = Q_cons_user[1,:]
        energy      = Q_cons_user[2,:]
        
        rho = mass / vol
        vx  = momx / rho / vol
        P   = (energy/vol - (0.5 * rho * (vx**2))) * (gamma-1)
        T   = P / rho / (ct.gas_constant / state['MW_gas']) 

        state['Q_prim'] = np.vstack((rho,vx,P,T)).ravel()

        # state = update_ghost_cell(solver_param,state)

    else:

        vol         = solver_param['vol']
        gamma       = solver_param['gamma']

        Q_cons      = state['Q_cons']
        Q_cons_user = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

        mass        = Q_cons_user[0,:]
        momx        = Q_cons_user[1,:]
        energy      = Q_cons_user[2,:]
        mass_species= Q_cons_user[3:,:]
        
        rho = mass / vol
        vx  = momx / rho / vol
        MF  = mass_species / rho / vol

        MF_ct = find_mass_fraction_full_cantera(MF)

        sp_vol = 1/rho

        internal_energy = (energy/vol/rho)-(0.5*vx**2)

        state['gas_array'].UVY = internal_energy,sp_vol,MF_ct

        T = np.squeeze(state['gas_array'].T)
        P = np.squeeze(state['gas_array'].P)

        state['Q_prim'] = np.vstack((rho,vx,P,T,MF)).ravel()

        # import matplotlib.pyplot as plt
        # fig, ax1 = plt.subplots()

        # color = 'tab:red'
        # ax1.plot(internal_energy, color=color)

        # ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

        # color = 'tab:blue'
        # ax2.plot(internal_energy, color=color)


        # state = update_ghost_cell(solver_param,state)

    return  state

def slope_limit(df_dx,Q_user,dx,solver_param):

    if solver_param['limiter_method'] == 'minmod':
        
       df_dx = minmod_limiter(df_dx,Q_user,dx)

    elif solver_param['limiter_method'] == 'barth':
        
        df_dx = barth_jespersen(df_dx,Q_user,dx,solver_param)

    return df_dx

def minmod_limiter(df_dx,Q_user,dx):


    df_dx = np.maximum(0., np.minimum(1., ( (Q_user-np.roll(Q_user,1,axis=0))/dx)/(df_dx + 1.0e-8*(df_dx==0)))) * df_dx
    df_dx = np.maximum(0., np.minimum(1., (-(Q_user-np.roll(Q_user,-1,axis=0))/dx)/(df_dx + 1.0e-8*(df_dx==0)))) * df_dx


    return df_dx

def barth_jespersen(df_dx,Q_user,dx,solver_param):

    Q_shape  = np.shape(Q_user)

    num_vars = Q_shape[0]
    num_cell = Q_shape[1]

    phi = np.zeros_like(Q_user)

    face_prim_user = np.zeros((num_vars,2*(num_cell)+2))
    
    for indx in range(1,num_cell):

        face_prim_user[:,2*indx]   = Q_user[:,indx-1] + df_dx[:,indx-1] * (dx/2)     # left face 
        face_prim_user[:,2*indx+1] = Q_user[:,indx]   - df_dx[:,indx] * (dx/2)       # right face 

    phi_left1  = 1
    phi_left2  = 1
    phi_left3  = 1
    phi_right1 = 1
    phi_right2 = 1
    phi_right3 = 1
    
    for var in range(num_vars):

        for indx in range(1,num_cell-1):

            max_neighbor = np.max((Q_user[var,indx-1],Q_user[var,indx],Q_user[var,indx+1]))
            min_neighbor = np.min((Q_user[var,indx-1],Q_user[var,indx],Q_user[var,indx+1]))

            left_face_estimate  = face_prim_user[var,2*indx+1]  
            right_face_estimate = face_prim_user[var,2*(indx+1)] 
            
            # left face check 
            if  left_face_estimate - Q_user[var,indx] > 0:

                phi_left1 = np.min(( 1 , (max_neighbor - Q_user[var,indx]) / (left_face_estimate -  Q_user[var,indx]) ))

            elif left_face_estimate - Q_user[var,indx] < 0:

                phi_left2 = np.min(( 1 , (min_neighbor - Q_user[var,indx]) / (left_face_estimate -  Q_user[var,indx]) ))

            else:
                
                phi_left3 = 1


            # right face check 
            if  right_face_estimate - Q_user[var,indx] > 0:

                phi_right1 = np.min(( 1 , (max_neighbor - Q_user[var,indx]) / (right_face_estimate -  Q_user[var,indx]) ))

            elif right_face_estimate - Q_user[var,indx] < 0:

                phi_right2 = np.min(( 1 , (min_neighbor - Q_user[var,indx]) / (right_face_estimate -  Q_user[var,indx]) ))

            else:
                
                phi_right3 = 1

            phi[var,indx]=np.min((phi_left1,phi_left2,phi_left3,phi_right1,phi_right2,phi_right3))


    df_dx = df_dx * phi

    return df_dx

def results_recorder(solver_param, rom_param, state):

    # Prepare the name for the files to be saved
    dir_results = os.path.join(solver_param['working_dir'], f"{solver_param['solver_mode']}_results")
    iter = solver_param['iter']
    save_title = str(iter)+'iteration'

    if solver_param['solver_mode'] == 'FOM':
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, f"{save_title}_cons.npy"), state['cons_results_save'])
        np.save(os.path.join(dir_results, f"{save_title}_prim.npy"), state['prim_results_save'])

    elif solver_param['solver_mode'] == 'ROM' and iter == 0:
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, 'cons_prim'     , f"{save_title}_cons.npy")           , state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim'     , f"{save_title}_prim.npy")           , state['prim_results_save'])
        np.save(os.path.join(dir_results, 'q_r'           , f"{save_title}_q_r.npy")            , rom_param['q_red0']       )
        np.save(os.path.join(dir_results, 'basis'         , f"{save_title}_basis.npy")          , rom_param['basis']        )
        np.save(os.path.join(dir_results, 'q_ref'         , f"{save_title}_q_ref.npy")          , rom_param['q_ref']        )
        np.save(os.path.join(dir_results, 'norm'          , f"{save_title}_norm.npy")           , rom_param['normalizor']   )
        np.save(os.path.join(dir_results, 'denorm'        , f"{save_title}_denorm.npy")         , rom_param['denormalizor'] )
        np.save(os.path.join(dir_results, 'samples_user'  , f"{save_title}_samples_user.npy")   , rom_param['S_indx_user']  )
        np.save(os.path.join(dir_results, 'samples_solver', f"{save_title}_samples_solver.npy") , rom_param['S_indx_solver'])

    elif solver_param['solver_mode'] == 'ROM':
        np.save(os.path.join(dir_results, 'cons_prim', f"{save_title}_cons.npy"), state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim', f"{save_title}_prim.npy"), state['prim_results_save'])
        np.save(os.path.join(dir_results, 'q_r'      , f"{save_title}_q_r.npy") , rom_param['q_red0'])

    elif (solver_param['solver_mode'] == 'Adaptive ROM' and iter < int(solver_param['init_training_win'])):
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, 'cons_prim', f"{save_title}_cons.npy"), state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim', f"{save_title}_prim.npy"), state['prim_results_save'])

    elif (solver_param['solver_mode'] == 'Adaptive ROM' and iter == int(solver_param['init_training_win'])):
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, 'cons_prim'      , f"{save_title}_cons.npy")           , state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim'      , f"{save_title}_prim.npy")           , state['prim_results_save'])
        np.save(os.path.join(dir_results, 'basis'          , f"{save_title}_basis.npy")          , rom_param['basis']        )
        np.save(os.path.join(dir_results, 'q_ref'          , f"{save_title}_q_ref.npy")          , rom_param['q_ref']        )
        np.save(os.path.join(dir_results, 'norm'           , f"{save_title}_norm.npy")           , rom_param['normalizor']   )
        np.save(os.path.join(dir_results, 'denorm'         , f"{save_title}_denorm.npy")         , rom_param['denormalizor'] )
        np.save(os.path.join(dir_results, 'samples_user'   , f"{save_title}_samples_user.npy")   , rom_param['S_indx_user']  )
        np.save(os.path.join(dir_results, 'samples_solver' , f"{save_title}_samples_solver.npy") , rom_param['S_indx_solver'])

        # q_r needed for init training windows
        for indx in range(int(solver_param['init_training_win']) + 1):
            save_title = 'iteration' + str(indx)
            np.save(os.path.join(dir_results, 'q_r', f"{save_title}_q_r.npy"), rom_param['q_red0'])

    elif (solver_param['solver_mode'] == 'Adaptive ROM' and iter > int(solver_param['init_training_win'])):
        # Save the results and end the simulation
        np.save(os.path.join(dir_results, 'cons_prim'     , f"{save_title}_cons.npy")          , state['cons_results_save'])
        np.save(os.path.join(dir_results, 'cons_prim'     , f"{save_title}_prim.npy")          , state['prim_results_save'])
        np.save(os.path.join(dir_results, 'q_r'           , f"{save_title}_q_r.npy")           , rom_param['q_red0']       )
        np.save(os.path.join(dir_results, 'basis'         , f"{save_title}_basis.npy")         , rom_param['basis']        )
        np.save(os.path.join(dir_results, 'samples_user'  , f"{save_title}_samples_user.npy")  , rom_param['S_indx_user']  )
        np.save(os.path.join(dir_results, 'samples_solver', f"{save_title}_samples_solver.npy"), rom_param['S_indx_solver'])  


def rusanov_flux_calculator(solver_param,state):

    Q_cons           = state['Q_cons']
    # state            = cons2prim_converter(solver_param,state)
    Q_prim           = state['Q_prim']
    Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]

    Y      = Q_prim_user[4:,:]
    Y_ct   = find_mass_fraction_full_cantera(Y)

    vol           = solver_param['vol']
    dx            = vol

    rho_face_right = rho[1:]
    rho_face_left  = rho[0:-1]

    p_face_right   = press[1:]
    p_face_left    = press[0:-1]

    vx_face_right  = vx[1:]
    vx_face_left   = vx[0:-1]

    Y_face_right   = Y[:,1:]
    Y_face_left    = Y[:,0:-1]

    # rho_grad    = gradient_calculator(rho,dx)
    # P_grad      = gradient_calculator(press,dx)
    # vx_grad     = gradient_calculator(vx,dx)
    # Y_grad      = gradient_calculator(Y,dx)

    # rho_grad    = slope_limit(rho,dx,rho_grad)
    # P_grad      = slope_limit(press,dx,P_grad)
    # vx_grad     = slope_limit(vx,dx,vx_grad)
    # Y_grad      = slope_limit(Y,dx,Y_grad)

    # rho_face_left,rho_face_right    = extrapolate_center2face(rho,rho_grad,dx)
    # p_face_left,p_face_right        = extrapolate_center2face(press,P_grad,dx)
    # vx_face_left,vx_face_right      = extrapolate_center2face(vx,vx_grad,dx)

    # Y_face_left  = np.zeros((len(Y[:,0]),len(Y[0,:])+1))
    # Y_face_right = np.zeros((len(Y[:,0]),len(Y[0,:])+1))

    # for indx in range(1,len(Y[0,:])):

    #     Y_face_left[:,indx]  = Y[:,indx-1] + Y_grad[:,indx-1] * (dx/2)
    #     Y_face_right[:,indx] = Y[:,indx]   - Y_grad[:,indx] * (dx/2)
    


    state['gas_array'].TPY = temp,press,Y_ct

    # int_en_ct         = np.squeeze(state['gas_array'].int_energy_mass)

    # int_en_right   = int_en[1:]   
    # int_en_left    = int_en[0:-1] 

    tot_energy     = Q_cons_user[2,:] / vol / rho

    int_en          = tot_energy - 0.5*(vx**2)

    int_en_right   = int_en[1:]   
    int_en_left    = int_en[0:-1] 

    # e_grad    = gradient_calculator(int_en,dx)

    # e_grad    = slope_limit(int_en,dx,e_grad)

    # int_en_left,int_en_right    = extrapolate_center2face(int_en,e_grad,dx)

    en_L           = rho_face_left  * (int_en_left + 0.5 * (vx_face_left**2))
    en_R           = rho_face_right * (int_en_right + 0.5 * (vx_face_right**2))

    C              = np.squeeze(state['gas_array'].sound_speed)

    C_R            = C[1:]
    C_L            = C[0:-1]


    diffusion_L = np.max( np.abs(vx_face_left+C_L) )
    diffusion_R = np.max( np.abs(vx_face_right+C_R) )

    diffusion = 0.5 * np.max([diffusion_L,diffusion_R])

    flux_mass   = 0.5*(rho_face_left*vx_face_left + rho_face_right*vx_face_right)                               - diffusion*(rho_face_right-rho_face_left)
    flux_momx   = 0.5*(rho_face_left*vx_face_left**2+p_face_left + rho_face_right*vx_face_right**2+p_face_right)- diffusion*(rho_face_right*vx_face_right-rho_face_left*vx_face_left)
    flux_energy = 0.5*(vx_face_left*(en_L+p_face_left) + vx_face_right*(en_R+p_face_right))                     - diffusion*(en_R-en_L)

    flux_spec   = 0.5*(rho_face_left*vx_face_left*Y_face_left + rho_face_right*vx_face_right*Y_face_right)       - diffusion*(rho_face_right*Y_face_right - rho_face_left * Y_face_left)

    flux = np.vstack((flux_mass,flux_momx,flux_energy,flux_spec))

    # Create arrays of zeros
    left_zeros = np.zeros((flux.shape[0], 1))
    right_zeros = np.zeros((flux.shape[0], 1))

    flux_final = np.hstack((left_zeros, flux, right_zeros))

    state['flux_cons'] = flux_final

    return state

def backup_first_order_roe_inviscid_flux_calculator(solver_param,rom_param,state):

    if solver_param['gas_model'] == 'Non-Reacting Air':
    
        Q_cons           = state['Q_cons']
        # state            = cons2prim_converter(solver_param,state)
        Q_prim           = state['Q_prim']
        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]

        
        gamma  = solver_param['gamma']
        vol    = solver_param['vol']
        num_state_var = solver_param['num_state_var']

        # breakpoint()

        # c_max = np.max(np.abs(np.sqrt(gamma*press/rho)))

        # cfl = c_max*solver_param['dt']/vol

        # dt = 0.08*vol/c_max

        S_indx_user = rom_param['S_indx_user']

        en               = press  / (gamma-1) + 0.5 * rho  * (vx**2)
        # number of cell
        cell_num = len(rho)

        # total enthalpy
        htot = gamma/(gamma-1)*press/rho+0.5*vx**2

        flux = np.zeros((num_state_var,cell_num+1))
        diffusion = np.zeros((num_state_var,cell_num+1))

        if solver_param['hyper'] == True:

            range_flux = S_indx_user + 2 
            range_flux_neighbor_left  = range_flux - 1
            range_flux = np.concatenate((range_flux,range_flux_neighbor_left))
            range_flux = np.sort(np.unique(range_flux))


        else : 
            
            range_flux = range(0,cell_num-1)

        for j in range_flux:
        
            # Compute Roe averages
            R=np.sqrt(rho[j+1]/rho[j])                      # R_{j+1/2}
            rmoy=R*rho[j]                                   # {hat rho}_{j+1/2}
            umoy=(R*vx[j+1]+vx[j])/(R+1)                    # {hat U}_{j+1/2}
            hmoy=(R*htot[j+1]+htot[j])/(R+1);               # {hat H}_{j+1/2}
            amoy=np.sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy))  # {hat a}_{j+1/2}
            
            # Auxiliary variables used to compute P_{j+1/2}^{-1}
            alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy)
            alph2=(gamma-1)/(amoy*amoy)

            # Compute matrix P^{-1}_{j+1/2}
            Pinv = np.array([[0.5*(alph1+umoy/amoy), -0.5*(alph2*umoy+1/amoy),  alph2/2],
                            [1-alph1,                alph2*umoy,                -alph2 ],
                            [0.5*(alph1-umoy/amoy),  -0.5*(alph2*umoy-1/amoy),  alph2/2]])
                    
            # Compute matrix P_{j+1/2}
            P    = np.array([[ 1,              1,              1              ],
                            [umoy-amoy,        umoy,           umoy+amoy      ],
                            [hmoy-amoy*umoy,   0.5*umoy*umoy,  hmoy+amoy*umoy ]])
            
            # Compute matrix Lambda_{j+1/2}
            lamb = np.array([[ abs(umoy-amoy),  0,              0                 ],
                            [0,                 abs(umoy),      0                 ],
                            [0,                 0,              abs(umoy+amoy)    ]])
                        
            # Compute Roe matrix |A_{j+1/2}|
            A = P @ lamb @ Pinv

            diffusion[:,j+1] = 0.5 * A @ (Q_cons_user[:,j+1]-Q_cons_user[:,j]) / vol

            flux[0,j+1] = 0.5*(rho[j]*vx[j]               + rho[j+1]*vx[j+1])                         - diffusion[0,j+1] 
            flux[1,j+1] = 0.5*(rho[j]*vx[j]**2 + press[j] + rho[j+1]*vx[j+1]**2+press[j+1])           - diffusion[1,j+1] 
            flux[2,j+1] = 0.5*(vx[j]*(en[j]+press[j])     + vx[j+1]*(en[j+1]+press[j+1]))             - diffusion[2,j+1] 

    ### If It is a Multi-Species Case ###

    else :
            
        Q_cons           = state['Q_cons']
        # state            = cons2prim_converter(solver_param,state)
        Q_prim           = state['Q_prim']
        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]
        temp   = Q_prim_user[3,:]


        Y      = Q_prim_user[4:,:]
        Y_ct   = find_mass_fraction_full_cantera(Y)


        vol    = solver_param['vol']
        num_state_var = solver_param['num_state_var']
        num_species   = solver_param['num_species']

        # breakpoint()

        # c_max = np.max(c)

        # cfl = c_max*1e-12/vol

        # dt = 0.08*vol/c_max

        S_indx_user = rom_param['S_indx_user']

        # total energy (rho * e_t)
        en               = Q_cons_user[2,:] / vol

        state['gas_array'].TPY = temp,press,Y_ct

        c       = np.squeeze(state['gas_array'].sound_speed)
        int_en  = np.squeeze(state['gas_array'].int_energy_mass)
        h       = np.squeeze(state['gas_array'].enthalpy_mass)

        # total enthalpy
        htot = h + (0.5*vx**2)

        # number of cell
        cell_num = len(rho)

        flux = np.zeros((num_state_var,cell_num+1))
        diffusion = np.zeros((3,cell_num+1))

        if solver_param['hyper'] == True:

            range_flux = S_indx_user + 2 
            range_flux_neighbor_left  = range_flux - 1
            range_flux = np.concatenate((range_flux,range_flux_neighbor_left))
            range_flux = np.sort(np.unique(range_flux))

        else : 
            
            range_flux = range(0,cell_num-1)

        for j in range_flux:
        
            # Compute Roe averages
            R=np.sqrt(rho[j+1]/rho[j])                      # R_{j+1/2}
            rmoy=R*rho[j]                                   # {hat rho}_{j+1/2}
            umoy=(R*vx[j+1]+vx[j])/(R+1)                    # {hat U}_{j+1/2}
            Hmoy=(R*htot[j+1]+htot[j])/(R+1);               # {hat H}_{j+1/2} 
            cmoy = (R*c[j+1]+c[j])/(R+1);                   # {hat c}_{j+1/2} 

            del_p  = press[j+1]- press[j]
            del_vx = vx[j+1]   - vx[j]
            del_rho= rho[j+1]  - rho[j]

            diffusion[0,j+1] = rmoy*del_vx + del_rho*umoy
            diffusion[1,j+1] = del_p + (2*rmoy*umoy*del_vx) + (umoy**2*del_rho)

            l1 = umoy+cmoy
            l2 = umoy-cmoy
            l3 = umoy

            e1 = np.array([1,umoy+cmoy,Hmoy+umoy*cmoy])
            e2 = np.array([1,umoy-cmoy,Hmoy-umoy*cmoy])

            a1 = (del_p+rmoy*cmoy*del_vx)/(2*cmoy*cmoy)
            a2 = (del_p-rmoy*cmoy*del_vx)/(2*cmoy*cmoy)

            dF3 = (vx[j+1]*(en[j+1]+press[j+1])) - (vx[j]*(en[j]+press[j]))

            X = dF3 - e1[2]*l1*a1 - e2[2]*l2*a2

            X = X * np.sign(X)

            diffusion[2,j+1] = e1[2]*np.abs(l1)*a1 + e2[2]*np.abs(l2)*a2 + X
                            
            flux[0,j+1] = 0.5*(rho[j]*vx[j]               + rho[j+1]*vx[j+1])                         - 0.5*np.abs(diffusion[0,j+1])
            flux[1,j+1] = 0.5*(rho[j]*vx[j]**2 + press[j] + rho[j+1]*vx[j+1]**2+press[j+1])           - 0.5*np.abs(diffusion[1,j+1])
            flux[2,j+1] = 0.5*(vx[j]*(en[j]+press[j])     + vx[j+1]*(en[j+1]+press[j+1]))             - 0.5*diffusion[2,j+1]
            
            # calculating mass species fluxes in a semi-decoupled approach

            for sp in range(num_species):

                if flux[0,j+1] > 0:

                    flux[sp+3,j+1] =  flux[0,j+1] * Y[sp,j]

                else:

                    flux[sp+3,j+1] = flux[0,j+1] * Y[sp,j+1]


    # breakpoint()
    state['flux_cons'] = flux

    return state

def first_order_roe_inviscid_flux_calculator(solver_param,rom_param,state):

    if solver_param['gas_model'] == 'Non-Reacting Air':

        state            = cons2prim_converter(solver_param,state)
        state            = update_ghost_cell(solver_param,state)
    
        Q_cons           = state['Q_cons']
        Q_prim           = state['Q_prim']

        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]

        
        gamma  = solver_param['gamma']
        vol    = solver_param['vol']
        num_state_var = solver_param['num_state_var']

        # breakpoint()

        # c_max = np.max(np.abs(np.sqrt(gamma*press/rho)))

        # cfl = c_max*solver_param['dt']/vol

        # dt = 0.08*vol/c_max

        S_indx_user = rom_param['S_indx_user']

        en               = press  / (gamma-1) + 0.5 * rho  * (vx**2)
        # number of cell
        cell_num = len(rho)

        # total enthalpy
        htot = gamma/(gamma-1)*press/rho+0.5*vx**2

        flux = np.zeros((num_state_var,cell_num+1))
        diffusion = np.zeros((num_state_var,cell_num+1))

        if solver_param['hyper'] == True:

            range_flux = S_indx_user + 2 
            range_flux_neighbor_left  = range_flux - 1
            range_flux = np.concatenate((range_flux,range_flux_neighbor_left))
            range_flux = np.sort(np.unique(range_flux))


        else : 
            
            range_flux = range(0,cell_num-1)

        for j in range_flux:
        
            # Compute Roe averages
            R=np.sqrt(rho[j+1]/rho[j])                      # R_{j+1/2}
            rmoy=R*rho[j]                                   # {hat rho}_{j+1/2}
            umoy=(R*vx[j+1]+vx[j])/(R+1)                    # {hat U}_{j+1/2}
            hmoy=(R*htot[j+1]+htot[j])/(R+1);               # {hat H}_{j+1/2}
            amoy=np.sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy))  # {hat a}_{j+1/2}
            
            # Auxiliary variables used to compute P_{j+1/2}^{-1}
            alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy)
            alph2=(gamma-1)/(amoy*amoy)

            # Compute matrix P^{-1}_{j+1/2}
            Pinv = np.array([[0.5*(alph1+umoy/amoy), -0.5*(alph2*umoy+1/amoy),  alph2/2],
                            [1-alph1,                alph2*umoy,                -alph2 ],
                            [0.5*(alph1-umoy/amoy),  -0.5*(alph2*umoy-1/amoy),  alph2/2]])
                    
            # Compute matrix P_{j+1/2}
            P    = np.array([[ 1,              1,              1              ],
                            [umoy-amoy,        umoy,           umoy+amoy      ],
                            [hmoy-amoy*umoy,   0.5*umoy*umoy,  hmoy+amoy*umoy ]])
            
            # Compute matrix Lambda_{j+1/2}
            lamb = np.array([[ abs(umoy-amoy),  0,              0                 ],
                            [0,                 abs(umoy),      0                 ],
                            [0,                 0,              abs(umoy+amoy)    ]])
                        
            # Compute Roe matrix |A_{j+1/2}|
            A = P @ lamb @ Pinv

            diffusion[:,j+1] = 0.5 * A @ (Q_cons_user[:,j+1]-Q_cons_user[:,j]) / vol

            flux[0,j+1] = 0.5*(rho[j]*vx[j]               + rho[j+1]*vx[j+1])                         - diffusion[0,j+1] 
            flux[1,j+1] = 0.5*(rho[j]*vx[j]**2 + press[j] + rho[j+1]*vx[j+1]**2+press[j+1])           - diffusion[1,j+1] 
            flux[2,j+1] = 0.5*(vx[j]*(en[j]+press[j])     + vx[j+1]*(en[j+1]+press[j+1]))             - diffusion[2,j+1] 

    ### If It is a Multi-Species Case ###

    else :
        
        state            = cons2prim_converter(solver_param,state)
        state            = update_ghost_cell(solver_param,state)

        Q_cons           = state['Q_cons']
        Q_prim           = state['Q_prim']

        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]
        temp   = Q_prim_user[3,:]


        Y      = Q_prim_user[4:,:]
        Y_ct   = find_mass_fraction_full_cantera(Y)


        vol    = solver_param['vol']
        num_state_var = solver_param['num_state_var']
        num_species   = solver_param['num_species']



        S_indx_user = rom_param['S_indx_user']

        # total energy (rho * e_t)
        en               = Q_cons_user[2,:] / vol

        state['gas_array'].TPY = temp,press,Y_ct

        c       = np.squeeze(state['gas_array'].sound_speed)
        int_en  = np.squeeze(state['gas_array'].int_energy_mass)
        h       = np.squeeze(state['gas_array'].enthalpy_mass)

        # breakpoint()

        # c_max = np.max(c)

        # cfl = c_max*1e-8/vol

        # dt = 0.08*vol/c_max

        # total enthalpy
        htot = h + (0.5*(vx**2))

        # number of cell
        cell_num = len(rho)

        flux = np.zeros((num_state_var,cell_num+1))
        diss_matrix = np.zeros((num_state_var,num_state_var))

        if solver_param['hyper'] == True:

            range_flux = S_indx_user + 2 
            range_flux_neighbor_left  = range_flux - 1
            range_flux = np.concatenate((range_flux,range_flux_neighbor_left))
            range_flux = np.sort(np.unique(range_flux))

        else : 
            
            range_flux = range(0,cell_num-1)

        for j in range_flux:
        
            # Compute Roe averages
            R=np.sqrt(rho[j+1]/rho[j])                      # R_{j+1/2}
            rmoy=R*rho[j]                                   # {hat rho}_{j+1/2}
            umoy=(R*vx[j+1]+vx[j])/(R+1)                    # {hat U}_{j+1/2}
            Hmoy=(R*htot[j+1]+htot[j])/(R+1)                # {hat H}_{j+1/2}
            emoy=(R*int_en[j+1]+int_en[j])/(R+1)              
            hmoy=Hmoy - (0.5*umoy*umoy)
            Ymoy=(R*Y[:,j+1]+Y[:,j])/(R+1)                  

            Ymoy_ct = find_mass_fraction_full_cantera(Ymoy.reshape(-1,1))

            state['gas'].UVY = emoy,1/rmoy,Ymoy_ct

            Pmoy         = state['gas'].P
            Tmoy         = state['gas'].T
            cpmoy        = state['gas'].cp
            cmoy         = state['gas'].sound_speed
            meanMWmoy    = state['gas'].mean_molecular_weight
            MWmoy        = state['gas'].molecular_weights
            partial_hmoy = state['gas'].partial_molar_enthalpies / MWmoy
            
            d_rho_d_press = rmoy / Pmoy
            d_rho_d_temp = -rmoy/Tmoy

            d_rho_d_mass_frac = np.zeros_like(Ymoy)

            for sp in range(len(Ymoy)):

                d_rho_d_mass_frac[sp] = rmoy*meanMWmoy*(1/MWmoy[-1] - 1/MWmoy[sp])

            d_enth_d_press = 0
            d_enth_d_temp = cpmoy

            d_enth_d_mass_frac = np.zeros_like(Ymoy)

            for sp in range(len(Ymoy)):

                d_enth_d_mass_frac[sp] = partial_hmoy[sp] - partial_hmoy[-1]

            # Gamma terms for energy equation
            g_press     = rmoy * d_enth_d_press + d_rho_d_press * Hmoy - 1.0
            g_temp      = rmoy * d_enth_d_temp + d_rho_d_temp * Hmoy
            g_mass_frac = rmoy * d_enth_d_mass_frac + Hmoy * d_rho_d_mass_frac

            # Characteristic speeds
            lambda1 = umoy + cmoy
            lambda2 = umoy - cmoy
            lambda1_abs = np.absolute(lambda1)
            lambda2_abs = np.absolute(lambda2)

            r_roe = (lambda2_abs - lambda1_abs) / (lambda2 - lambda1)
            alpha = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)
            beta  = np.power(cmoy, 2.0) * (lambda1_abs - lambda2_abs) / (lambda1 - lambda2)
            phi   = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)

            eta = (1.0 - rmoy * d_enth_d_press) / d_enth_d_temp
            psi = eta * d_rho_d_temp + rmoy * d_rho_d_press

            vel_abs = np.absolute(umoy)

            beta_star = beta * psi
            beta_e = beta * (rmoy * g_press + g_temp * eta)
            phi_star = d_rho_d_press * phi + d_rho_d_temp * eta * (phi - vel_abs) / rmoy
            phi_e = g_press * phi + g_temp * eta * (phi - vel_abs) / rmoy
            m = rmoy * alpha
            e = rmoy * umoy * alpha

            delta_p = press[j]-press[j+1]
            delta_u = vx[j]-vx[j+1]
            delta_T = temp[j]-temp[j+1]
            delta_Y = (Y[:,j]-Y[:,j+1]).reshape(-1,1)


            del_q_prim = np.vstack((delta_p,delta_u,delta_T,delta_Y))

            diss_matrix[ 0 , 0 ] = phi_star
            diss_matrix[ 0 , 1 ] = beta_star
            diss_matrix[ 0 , 2 ] = vel_abs * d_rho_d_temp
            diss_matrix[ 0 , 3:] = vel_abs * d_rho_d_mass_frac

            diss_matrix[ 1 , 0 ] = umoy * phi_star + r_roe
            diss_matrix[ 1 , 1 ] = umoy * beta_star + m
            diss_matrix[ 1 , 2 ] = umoy * vel_abs * d_rho_d_temp
            diss_matrix[ 1 , 3:] = (umoy * vel_abs) * d_rho_d_mass_frac

            diss_matrix[ 2 , 0 ] = phi_e + r_roe * umoy
            diss_matrix[ 2 , 1 ] = beta_e + e
            diss_matrix[ 2 , 2 ] = g_temp * vel_abs
            diss_matrix[ 2 , 3:] = g_mass_frac * vel_abs

            diss_matrix[3:, 0] = Ymoy * phi_star
            diss_matrix[3:, 1] = Ymoy * beta_star
            diss_matrix[3:, 2] = Ymoy * (vel_abs * d_rho_d_temp)

            for mf_idx_out in range(3, num_state_var):

                for mf_idx_in in range(3, num_state_var):

                    if mf_idx_out == mf_idx_in:
                        diss_matrix[mf_idx_out, mf_idx_in] = vel_abs * (
                            rmoy + Ymoy[mf_idx_out - 3] * d_rho_d_mass_frac[mf_idx_in - 3]
                        )
                    else:
                        diss_matrix[mf_idx_out, mf_idx_in] = (
                            vel_abs * Ymoy[mf_idx_out - 3] * d_rho_d_mass_frac[mf_idx_in - 3]
                        )

            dissipation = diss_matrix @ del_q_prim
        
            flux[0,j+1] = 0.5*(rho[j]*vx[j]               + rho[j+1]*vx[j+1])                         + 0.5 * dissipation[0 ,0]
            flux[1,j+1] = 0.5*(rho[j]*vx[j]**2 + press[j] + rho[j+1]*vx[j+1]**2+press[j+1])           + 0.5 * dissipation[1 ,0]
            flux[2,j+1] = 0.5*(vx[j]*(en[j]+press[j])     + vx[j+1]*(en[j+1]+press[j+1]))             + 0.5 * dissipation[2 ,0]
            flux[3:,j+1]= 0.5*(rho[j]*vx[j]*Y[:,j]        + rho[j+1]*vx[j+1]*Y[:,j+1] )               + 0.5 * dissipation[3:,0]
            

    state['flux_cons'] = flux

    return state

def first_order_rusanov_viscous_flux_calculator(solver_param,rom_param,state):

    if solver_param['gas_model'] != 'Non-Reacting Air':
    
        # load the basic information
        Q_cons           = state['Q_cons']
        # state            = cons2prim_converter(solver_param,state)
        Q_prim           = state['Q_prim']
        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]
        temp   = Q_prim_user[3,:]
        Y      = Q_prim_user[4:,:]

        Y_full = find_mass_fraction_full(Y)
        Y_ct   = find_mass_fraction_full_cantera(Y)

        vol           = solver_param['vol']
        num_state_var = solver_param['num_state_var']
        dx            = vol

        state['gas_array'].TPY      = temp,press,Y_ct

        dyn_vsc_mix                 = np.squeeze(state['gas_array'].viscosity)
        mass_diff_mix               = np.transpose(state['gas_array'].mix_diff_coeffs_mass[0,:,:])
        therm_cond_mix              = np.squeeze(state['gas_array'].thermal_conductivity)

        MW                          = state['gas_array'].molecular_weights
        MW_mix                      = np.squeeze(state['gas_array'].mean_molecular_weight)
        enthalpies                  = state['gas_array'].partial_molar_enthalpies / MW

        h                           = np.transpose(enthalpies[0,:,:])

        X                           = np.transpose(state['gas_array'].X[0,:,:])

        X[X==0] = 1e-100

        du_dx = (np.roll(vx,-1)            - np.roll(vx,1)               )/ 2 / dx
        dT_dx = (np.roll(temp,-1)          - np.roll(temp,1)             )/ 2 / dx
        dY_dx = (np.roll(Y_full,-1,axis=1) - np.roll(Y_full,1,axis=1)    )/ 2 / dx
        dX_dx = (np.roll(X,-1,axis=1)      - np.roll(X,1,axis=1)    )     / 2 / dx

        du_dx = slope_limit(vx  , dx , du_dx)
        dT_dx = slope_limit(temp, dx , dT_dx)
        dY_dx = slope_limit(Y_full , dx , dY_dx)
        dX_dx = slope_limit(X , dx , dX_dx)

        Wk_W = np.zeros_like(X)

        for i in range(Wk_W.shape[0]):

            Wk_W[i,:] = MW[i] / MW_mix

        corr_vel                    = np.sum(mass_diff_mix*Wk_W*dX_dx)

        diff_vel                    = -mass_diff_mix*dX_dx/X + corr_vel

        tau = 4/3 * dyn_vsc_mix * du_dx

        q   = -therm_cond_mix * dT_dx + rho*np.sum(h*diff_vel*Y_full,axis=0)


        tau_right = tau[1:]
        tau_left  = tau[0:-1]

        vx_right  = vx[1:]
        vx_left   = vx[0:-1]
        
        q_right   = q[1:]
        q_left    = q[0:-1]

        rho_right = rho[1:]
        rho_left  = rho[0:-1]

        Y_right = Y[:,1:]
        Y_left  = Y[:,0:-1]


        D_right = mass_diff_mix[:-1,1:]
        D_left  = mass_diff_mix[:-1,0:-1]

        dY_dx_right = dY_dx[:-1,1:]
        dY_dx_left  = dY_dx[:-1,0:-1]

        dX_dx_right = dX_dx[:-1,1:]
        dX_dx_left  = dX_dx[:-1,0:-1]

        Wk_W_right = Wk_W[:-1,1:]
        Wk_W_left  = Wk_W[:-1,0:-1]
        
        
        flux_momx   = 0.5*(tau_left+tau_right)
        flux_energy = 0.5*((vx_left*tau_left-q_left)+(vx_right*tau_right-q_right))

        flux_spec   = 0.5*( (rho_left*D_left*Wk_W_left*dX_dx_left-rho_left*corr_vel*Y_left) + (rho_right*D_right*Wk_W_right*dX_dx_right-rho_right*corr_vel*Y_right) )

        flux_mass   = np.zeros_like(flux_momx)

        flux = np.vstack((flux_mass,flux_momx,flux_energy,flux_spec))

        right_zeros = np.zeros((flux.shape[0], 1))
        left_zeros = np.zeros((flux.shape[0], 1))

        flux_final = np.hstack((left_zeros,flux, right_zeros))

        state['flux_visc_cons'] = flux_final

    return state

def first_order_roe_viscous_flux_calculator(solver_param,rom_param,state):

    state            = cons2prim_converter(solver_param,state)
    state            = update_ghost_cell(solver_param,state)

    Q_cons           = state['Q_cons']
    Q_prim           = state['Q_prim']

    Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]

    Y      = Q_prim_user[4:,:]
    Y_full = find_mass_fraction_full(Y)
    Y_ct   = find_mass_fraction_full_cantera(Y)


    vol           = solver_param['vol']
    dx            = vol
    num_state_var = solver_param['num_state_var']
    num_species   = solver_param['num_species']

    S_indx_user = rom_param['S_indx_user']

    # total energy (rho * e_t)
    en               = Q_cons_user[2,:] / vol

    state['gas_array'].TPY = temp,press,Y_ct

    c       = np.squeeze(state['gas_array'].sound_speed)
    int_en  = np.squeeze(state['gas_array'].int_energy_mass)
    h       = np.squeeze(state['gas_array'].enthalpy_mass)

    # total enthalpy
    htot = h + (0.5*(vx**2))

    # number of cell
    cell_num = len(rho)

    flux = np.zeros((num_state_var,cell_num+1))

    if solver_param['hyper'] == True:

        range_flux = S_indx_user + 2 
        range_flux_neighbor_left  = range_flux - 1
        range_flux = np.concatenate((range_flux,range_flux_neighbor_left))
        range_flux = np.sort(np.unique(range_flux))


    else : 
        
        range_flux = range(0,cell_num-1)

    for j in range_flux:
    
        # Compute Roe averages
        R=np.sqrt(rho[j+1]/rho[j])                      # R_{j+1/2}
        rmoy=R*rho[j]                                   # {hat rho}_{j+1/2}
        umoy=(R*vx[j+1]+vx[j])/(R+1)                    # {hat U}_{j+1/2}
        Hmoy=(R*htot[j+1]+htot[j])/(R+1);               # {hat H}_{j+1/2}
        hmoy=(R*h[j+1]+h[j])/(R+1);                     # {hat h}_{j+1/2}
        emoy=(R*int_en[j+1]+int_en[j])/(R+1);           # {hat e}_{j+1/2}
        Ymoy=(R*Y[:,j+1]+Y[:,j])/(R+1);                 # {hat Y}_{j+1/2}          
        Pmoy = (hmoy - emoy)*rmoy                       # {hat P}_{j+1/2}

        Ymoy_full = np.squeeze( find_mass_fraction_full(Ymoy.reshape(-1,1)) )

        state['gas'].UVY            = emoy,1/rmoy,Ymoy_full

        dyn_vsc_mix                 = state['gas'].viscosity
        mass_diff_mix               = state['gas'].mix_diff_coeffs_mass
        therm_cond_mix              = state['gas'].thermal_conductivity

        MW                          = state['gas'].molecular_weights
        MW_mix                      = state['gas'].mean_molecular_weight
        enthalpies                  = state['gas'].partial_molar_enthalpies / MW


        du_dx = (vx[j+1] - vx[j]) /dx
        dT_dx = (temp[j+1] - temp[j]) /dx
        dY_dx = (Y_full[:,j+1] - Y_full[:,j]) / dx

        corr_vel  = np.sum(mass_diff_mix*dY_dx)

        diff_vel  = -mass_diff_mix*dY_dx/Ymoy_full + corr_vel

        tau       = 4/3 * dyn_vsc_mix * du_dx

        q         = -therm_cond_mix * dT_dx + rmoy*np.sum(enthalpies*diff_vel*Ymoy_full)

        flux[0,j+1] = 0 
        flux[1,j+1] = tau
        flux[2,j+1] = (umoy * tau) - q
        flux[3:,j+1]= (rmoy*mass_diff_mix[:-1]*dY_dx[:-1])-(rmoy*corr_vel*Ymoy)

    state['flux_visc_cons'] = flux

    return state

def second_order_roe_flux_calculator(solver_param,rom_param,state):

    if solver_param['gas_model'] == 'Non-Reacting Air':

        state            = cons2prim_converter(solver_param,state)
        state            = update_ghost_cell(solver_param,state)
    
        Q_cons           = state['Q_cons']
        Q_prim           = state['Q_prim']
        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        rho    = Q_prim_user[0,:]
        vx     = Q_prim_user[1,:]
        press  = Q_prim_user[2,:]

        gamma  = solver_param['gamma']
        vol    = solver_param['vol']
        dx     = vol

        S_indx_user = rom_param['S_indx_user']

        # number of cell
        cell_num = len(rho)

        # breakpoint()
        # calculate gradient
        df_dx = (np.roll(Q_prim_user,-1) - np.roll(Q_prim_user,1)) / (2*dx)

        # limiter

        if solver_param['limiter']:

            df_dx = slope_limit(df_dx,Q_prim_user,dx,solver_param)

        # face_reconstruction
        face_prim_user = np.zeros((solver_param['num_prim_var'],2*(cell_num)+2))

        for indx in range(1,cell_num):

            face_prim_user[:,2*indx]   = Q_prim_user[:,indx-1] + df_dx[:,indx-1] * (dx/2)     # left face 
            face_prim_user[:,2*indx+1] = Q_prim_user[:,indx]   - df_dx[:,indx] * (dx/2)       # right face 

        rho   = face_prim_user[0,:]
        vx    = face_prim_user[1,:]
        press = face_prim_user[2,:]

        # total enthalpy
        # htot = gamma/(gamma-1)*press/rho+0.5*vx**2
        # breakpoint()
        flux = np.zeros((3,cell_num+1))
        diffusion = np.zeros((3,cell_num+1))

        if solver_param['hyper'] == True:

            range_flux = S_indx_user + 2 

            range_flux_neighbor_left   = range_flux - 1
            range_flux_neighbor_right  = range_flux + 1 
            range_flux = np.concatenate((range_flux,range_flux_neighbor_left,range_flux_neighbor_right))
            range_flux = np.sort(np.unique(range_flux))


        else : 
            
            range_flux = range(1,cell_num)

        for j in range_flux:
            en_L               = press[2*j]    / (gamma-1) + 0.5 * rho[2*j]    * (vx[2*j]**2)
            en_R               = press[2*j+1]  / (gamma-1) + 0.5 * rho[2*j+1]  * (vx[2*j+1]**2)

            # total enthalpy
            htot_L = gamma/(gamma-1)*press[2*j]/rho[2*j]+0.5*vx[2*j]**2
            htot_R = gamma/(gamma-1)*press[2*j+1]/rho[2*j+1]+0.5*vx[2*j+1]**2
        
            # Compute Roe averages
            R=np.sqrt(rho[2*j+1]/rho[2*j])                      # R_{j+1/2}
            rmoy=R*rho[j]                                       # {hat rho}_{j+1/2}
            umoy=(R*vx[2*j+1]+vx[2*j])/(R+1)                    # {hat U}_{j+1/2}
            hmoy=(R*htot_R+htot_L)/(R+1);               # {hat H}_{j+1/2}
            amoy=np.sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy))  # {hat a}_{j+1/2}
            
            # Auxiliary variables used to compute P_{j+1/2}^{-1}
            alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy)
            alph2=(gamma-1)/(amoy*amoy)

            # Compute matrix P^{-1}_{j+1/2}
            Pinv = np.array([[0.5*(alph1+umoy/amoy), -0.5*(alph2*umoy+1/amoy),  alph2/2],
                            [1-alph1,                alph2*umoy,                -alph2 ],
                            [0.5*(alph1-umoy/amoy),  -0.5*(alph2*umoy-1/amoy),  alph2/2]])
                    
            # Compute matrix P_{j+1/2}
            P    = np.array([[ 1,              1,              1              ],
                            [umoy-amoy,        umoy,           umoy+amoy      ],
                            [hmoy-amoy*umoy,   0.5*umoy*umoy,  hmoy+amoy*umoy ]])
            
            # Compute matrix Lambda_{j+1/2}
            lamb = np.array([[ abs(umoy-amoy),  0,              0                 ],
                            [0,                 abs(umoy),      0                 ],
                            [0,                 0,              abs(umoy+amoy)    ]])
                        
            # Compute Roe matrix |A_{j+1/2}|
            A = P @ lamb @ Pinv

            diffusion[:,j] = 0.5 * A @ (Q_cons_user[:,j]-Q_cons_user[:,j-1]) / vol

            flux[0,j] = 0.5*(rho[2*j]*vx[2*j]                 + rho[2*j+1]*vx[2*j+1])                         - diffusion[0,j] 
            flux[1,j] = 0.5*(rho[2*j]*vx[2*j]**2 + press[2*j] + rho[2*j+1]*vx[2*j+1]**2+press[2*j+1])         - diffusion[1,j] 
            flux[2,j] = 0.5*(vx[2*j]*(en_L+press[2*j])        + vx[2*j+1]*(en_R+press[2*j+1]))                - diffusion[2,j] 


    ### If It is a Multi-Species Case ###

    else :
        
        state            = cons2prim_converter(solver_param,state)
        state            = update_ghost_cell(solver_param,state)

        Q_cons           = state['Q_cons']
        Q_prim           = state['Q_prim']

        Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
        Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

        vol    = solver_param['vol']
        dx     = vol

        # number of cell
        cell_num = len(Q_cons_user[0,:])

        # calculate gradient
        df_dx_prim = (np.roll(Q_prim_user,-1) - np.roll(Q_prim_user,1)) / (2*dx)
        df_dx_cons = (np.roll(Q_cons_user,-1) - np.roll(Q_cons_user,1)) / (2*dx)

        # limiter
        slope_limiter = solver_param['limiter']

        if slope_limiter:

            df_dx_prim = np.maximum(0., np.minimum(1., ( (Q_prim_user-np.roll(Q_prim_user,1,axis=0))/dx)/(df_dx_prim + 1.0e-8*(df_dx_prim==0)))) * df_dx_prim
            df_dx_prim = np.maximum(0., np.minimum(1., (-(Q_prim_user-np.roll(Q_prim_user,-1,axis=0))/dx)/(df_dx_prim + 1.0e-8*(df_dx_prim==0)))) * df_dx_prim

            df_dx_cons = np.maximum(0., np.minimum(1., ( (Q_cons_user-np.roll(Q_cons_user,1,axis=0))/dx)/(df_dx_cons + 1.0e-8*(df_dx_cons==0)))) * df_dx_cons
            df_dx_cons = np.maximum(0., np.minimum(1., (-(Q_cons_user-np.roll(Q_cons_user,-1,axis=0))/dx)/(df_dx_cons + 1.0e-8*(df_dx_cons==0)))) * df_dx_cons

        # face_reconstruction
        face_prim_user = np.zeros((solver_param['num_prim_var'],2*(cell_num)+2))
        face_cons_user = np.zeros((solver_param['num_state_var'],2*(cell_num)+2))

        for indx in range(1,cell_num):

            face_prim_user[:,2*indx]   = Q_prim_user[:,indx-1] + df_dx_prim[:,indx-1] * (dx/2)     # left face 
            face_prim_user[:,2*indx+1] = Q_prim_user[:,indx]   - df_dx_prim[:,indx] * (dx/2)       # right face   

            face_cons_user[:,2*indx]   = Q_cons_user[:,indx-1] + df_dx_cons[:,indx-1] * (dx/2)     # left face 
            face_cons_user[:,2*indx+1] = Q_cons_user[:,indx]   - df_dx_cons[:,indx] * (dx/2)       # right face         

        # # fill the ghost cells
        face_prim_user[:,0:2] = face_prim_user[:,2].reshape(-1,1)
        face_prim_user[:,-2:] = face_prim_user[:,-3].reshape(-1,1)

        face_cons_user[:,0:2] = face_cons_user[:,2].reshape(-1,1)
        face_cons_user[:,-2:] = face_cons_user[:,-3].reshape(-1,1)


        rho    = face_prim_user[0,:]
        vx     = face_prim_user[1,:]
        press  = face_prim_user[2,:]
        temp   = face_prim_user[3,:]


        Y      = face_prim_user[4:,:]
        Y_ct   = find_mass_fraction_full_cantera(Y)


        num_state_var = solver_param['num_state_var']
        num_species   = solver_param['num_species']

        S_indx_user = rom_param['S_indx_user']

        en               = face_cons_user[2,:] / vol

        state['gas_array_2nd_order'].TPY = temp,press,Y_ct

        c       = np.squeeze(state['gas_array_2nd_order'].sound_speed)
        int_en  = np.squeeze(state['gas_array_2nd_order'].int_energy_mass)
        h       = np.squeeze(state['gas_array_2nd_order'].enthalpy_mass)

        # total enthalpy
        htot = h + (0.5*(vx**2))

        flux = np.zeros((num_state_var,cell_num+1))
        diss_matrix = np.zeros((num_state_var,num_state_var))

        if solver_param['hyper'] == True:

            range_flux = S_indx_user + 2 
            range_flux_neighbor_left  = range_flux - 1
            range_flux = np.concatenate((range_flux,range_flux_neighbor_left))
            range_flux = np.sort(np.unique(range_flux))

        else : 
            
            range_flux = range(1,cell_num)

        for j in range_flux:
        
            # Compute Roe averages
            R=np.sqrt(rho[2*j+1]/rho[2*j])                      # R_{j+1/2}
            rmoy=R*rho[2*j]                                   # {hat rho}_{j+1/2}
            umoy=(R*vx[2*j+1]+vx[2*j])/(R+1)                    # {hat U}_{j+1/2}
            Hmoy=(R*htot[2*j+1]+htot[2*j])/(R+1)                # {hat H}_{j+1/2}
            emoy=(R*int_en[2*j+1]+int_en[2*j])/(R+1)              
            hmoy=Hmoy - (0.5*umoy*umoy)
            Ymoy=(R*Y[:,2*j+1]+Y[:,2*j])/(R+1)                  

            Ymoy_ct = find_mass_fraction_full_cantera(Ymoy.reshape(-1,1))

            state['gas'].UVY = emoy,1/rmoy,Ymoy_ct

            Pmoy         = state['gas'].P
            Tmoy         = state['gas'].T
            cpmoy        = state['gas'].cp
            cmoy         = state['gas'].sound_speed
            meanMWmoy    = state['gas'].mean_molecular_weight
            MWmoy        = state['gas'].molecular_weights
            partial_hmoy = state['gas'].partial_molar_enthalpies / MWmoy
            
            d_rho_d_press = rmoy / Pmoy
            d_rho_d_temp = -rmoy/Tmoy

            d_rho_d_mass_frac = np.zeros_like(Ymoy)

            for sp in range(len(Ymoy)):

                d_rho_d_mass_frac[sp] = rmoy*meanMWmoy*(1/MWmoy[-1] - 1/MWmoy[sp])

            d_enth_d_press = 0
            d_enth_d_temp = cpmoy

            d_enth_d_mass_frac = np.zeros_like(Ymoy)

            for sp in range(len(Ymoy)):

                d_enth_d_mass_frac[sp] = partial_hmoy[sp] - partial_hmoy[-1]

            # Gamma terms for energy equation
            g_press     = rmoy * d_enth_d_press + d_rho_d_press * Hmoy - 1.0
            g_temp      = rmoy * d_enth_d_temp + d_rho_d_temp * Hmoy
            g_mass_frac = rmoy * d_enth_d_mass_frac + Hmoy * d_rho_d_mass_frac

            # Characteristic speeds
            lambda1 = umoy + cmoy
            lambda2 = umoy - cmoy
            lambda1_abs = np.absolute(lambda1)
            lambda2_abs = np.absolute(lambda2)

            r_roe = (lambda2_abs - lambda1_abs) / (lambda2 - lambda1)
            alpha = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)
            beta  = np.power(cmoy, 2.0) * (lambda1_abs - lambda2_abs) / (lambda1 - lambda2)
            phi   = cmoy * (lambda1_abs + lambda2_abs) / (lambda1 - lambda2)

            eta = (1.0 - rmoy * d_enth_d_press) / d_enth_d_temp
            psi = eta * d_rho_d_temp + rmoy * d_rho_d_press

            vel_abs = np.absolute(umoy)

            beta_star = beta * psi
            beta_e = beta * (rmoy * g_press + g_temp * eta)
            phi_star = d_rho_d_press * phi + d_rho_d_temp * eta * (phi - vel_abs) / rmoy
            phi_e = g_press * phi + g_temp * eta * (phi - vel_abs) / rmoy
            m = rmoy * alpha
            e = rmoy * umoy * alpha

            delta_p = press[2*j]-press[2*j+1]
            delta_u = vx[2*j]-vx[2*j+1]
            delta_T = temp[2*j]-temp[2*j+1]
            delta_Y = (Y[:,2*j]-Y[:,2*j+1]).reshape(-1,1)


            del_q_prim = np.vstack((delta_p,delta_u,delta_T,delta_Y))

            diss_matrix[ 0 , 0 ] = phi_star
            diss_matrix[ 0 , 1 ] = beta_star
            diss_matrix[ 0 , 2 ] = vel_abs * d_rho_d_temp
            diss_matrix[ 0 , 3:] = vel_abs * d_rho_d_mass_frac

            diss_matrix[ 1 , 0 ] = umoy * phi_star + r_roe
            diss_matrix[ 1 , 1 ] = umoy * beta_star + m
            diss_matrix[ 1 , 2 ] = umoy * vel_abs * d_rho_d_temp
            diss_matrix[ 1 , 3:] = (umoy * vel_abs) * d_rho_d_mass_frac

            diss_matrix[ 2 , 0 ] = phi_e + r_roe * umoy
            diss_matrix[ 2 , 1 ] = beta_e + e
            diss_matrix[ 2 , 2 ] = g_temp * vel_abs
            diss_matrix[ 2 , 3:] = g_mass_frac * vel_abs

            diss_matrix[3:, 0] = Ymoy * phi_star
            diss_matrix[3:, 1] = Ymoy * beta_star
            diss_matrix[3:, 2] = Ymoy * (vel_abs * d_rho_d_temp)

            for mf_idx_out in range(3, num_state_var):

                for mf_idx_in in range(3, num_state_var):

                    if mf_idx_out == mf_idx_in:
                        diss_matrix[mf_idx_out, mf_idx_in] = vel_abs * (
                            rmoy + Ymoy[mf_idx_out - 3] * d_rho_d_mass_frac[mf_idx_in - 3]
                        )
                    else:
                        diss_matrix[mf_idx_out, mf_idx_in] = (
                            vel_abs * Ymoy[mf_idx_out - 3] * d_rho_d_mass_frac[mf_idx_in - 3]
                        )

            dissipation = diss_matrix @ del_q_prim
        
            flux[0,j] = 0.5*(rho[2*j]*vx[2*j]                 + rho[2*j+1]*vx[2*j+1])                           + 0.5 * dissipation[0 ,0]
            flux[1,j] = 0.5*(rho[2*j]*vx[2*j]**2 + press[2*j] + rho[2*j+1]*vx[2*j+1]**2+press[2*j+1])           + 0.5 * dissipation[1 ,0]
            flux[2,j] = 0.5*(vx[2*j]*(en[2*j]+press[2*j])     + vx[2*j+1]*(en[2*j+1]+press[2*j+1]))             + 0.5 * dissipation[2 ,0]
            flux[3:,j]= 0.5*(rho[2*j]*vx[2*j]*Y[:,2*j]        + rho[2*j+1]*vx[2*j+1]*Y[:,2*j+1] )               + 0.5 * dissipation[3:,0]
    # breakpoint()

    # flux = np.hstack((flux[:,0].reshape(-1,1),flux[:,0].reshape(-1,1),flux,flux[:,-1].reshape(-1,1),flux[:,-1].reshape(-1,1))) 

    state['flux_cons'] = flux

    return state

def d_flux_dx_calculator(solver_param,rom_param,state):

    S_indx_user = rom_param['S_indx_user']

    if solver_param['viscous_flag']:
        
        flux = state['flux_cons']-state['flux_visc_cons']

    else:

        flux = state['flux_cons']

    if solver_param['hyper'] == True:

        d_flux_dx = np.zeros((solver_param['num_state_var'] , len(S_indx_user)))

        S_indx_user = S_indx_user + 2

        counter = 0

        for indx in S_indx_user:

            d_flux_dx[:,counter] = -(flux[:,indx+1] - flux[:,indx])
            counter = counter + 1

    else:

        d_flux_dx = np.zeros((solver_param['num_state_var'] , solver_param['cell_number'] + 4))

        for indx in range(0,len(d_flux_dx[0,:])):

            d_flux_dx[:,indx] = -(flux[:,indx+1] - flux[:,indx])
    
    state['d_flux_dx'] = d_flux_dx.ravel()
    
    return state

def viscous_d_flux_dx_calculator(solver_param,rom_param,state):

    S_indx_user = rom_param['S_indx_user']

    flux = state['flux_visc_cons']

    if solver_param['hyper'] == True:

        d_flux_dx = np.zeros((solver_param['num_state_var'] , len(S_indx_user)))

        S_indx_user = S_indx_user + 2

        counter = 0

        for indx in S_indx_user:

            d_flux_dx[:,counter] = (flux[:,indx+1] - flux[:,indx])
            counter = counter + 1

    else:

        d_flux_dx = np.zeros((solver_param['num_state_var'] , solver_param['cell_number'] + 4))

        for indx in range(0,len(d_flux_dx[0,:])):

            d_flux_dx[:,indx] = (flux[:,indx+1] - flux[:,indx])


    visc_flux     = d_flux_dx.ravel()
    inviscid_flux = state['d_flux_dx']

    # add viscous flux into inviscid flux
    state['d_flux_dx'] = inviscid_flux - visc_flux
    
    return state

def results_solver2user_converter(num_state_var,num_cell,Q):

    Q_reshaped = np.reshape(Q,(num_state_var,num_cell+4))

    return Q_reshaped

def results_user2solver_converter(Q):

    Q_reshaped = np.ravel(Q)

    return Q_reshaped

def solver_eliminate_ghost(solver_param,Q_full):

    Q_full_user = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_full)
    Q_int_user = Q_full_user[:,2:-2]
    Q_int_solver = results_user2solver_converter(Q_int_user)

    return Q_int_solver

def solver_add_ghost(cell_number,num_var,Q_int):

    Q_int_user  = results_solver2user_converter(num_var,cell_number-4,Q_int)
    Q_int_full = np.column_stack((Q_int_user[:,0], Q_int_user[:,0], Q_int_user , Q_int_user[:,-1] , Q_int_user[:,-1]))
    Q_full_solver= results_user2solver_converter(Q_int_full)
    

    return Q_full_solver

def find_mass_fraction_full(MF):

    # this function is just finding the last species mass fraction and adds its value to last row

    MF_shape = np.shape(MF)

    # add the 1 last species
    MF_last_row = np.zeros(MF_shape[1])

    for indx in range(0,MF_shape[1]):

        MF_last_row[indx] = 1.0 - np.sum(MF[:,indx])

    MF = np.vstack((MF,MF_last_row))

    MF[MF==0] = 1e-30

    return MF

def find_mass_fraction_full_cantera(MF):

    # this function first finds the full mass fraction matrix and 
    # then converts it to the shape that is suitable for cantera

    MF = find_mass_fraction_full(MF)

    MF_shape = np.shape(MF)

    # reshape the MF array suitable for cantera

    MF_reshaped = np.zeros( (1,MF_shape[1],MF_shape[0]) )

    for indx in range(0,MF_shape[0]):

        MF_reshaped[:,:,indx] = MF[indx,:]

    return MF_reshaped

def source_calculator(solver_param,rom_param,state):

    # load the basic information
    Q_cons           = state['Q_cons']
    # state            = cons2prim_converter(solver_param,state)
    Q_prim           = state['Q_prim']
    Q_cons_user      = results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]
    Y      = Q_prim_user[4:,:]

    Y_ct = find_mass_fraction_full_cantera(Y)

    vol           = solver_param['vol']
    num_state_var = solver_param['num_state_var']

    state['gas_array'].TPY = temp,press,Y_ct

    net_production_rate_ct = state['gas_array'].net_production_rates
    MW_species             = state['gas_array'].molecular_weights
    state['heat_release']  = np.squeeze(state['gas_array'].heat_release_rate)
    state['int_energy']    = np.squeeze(state['gas_array'].int_energy_mass)

    # import matplotlib.pyplot as plt

    # plt.figure()
    # plt.plot(state['gas_array'].u[0,:])
    # plt.plot(e,linestyle='--')

    n_species = solver_param['num_species'] 

    source_terms   = np.zeros((solver_param['num_state_var'] , solver_param['cell_number'] + 4))

    for indx in range(n_species):

        source_terms[indx+3,:] = net_production_rate_ct[0,:,indx] * MW_species[indx] * vol


    state['source_terms'] = source_terms.ravel()

    if solver_param['hyper'] == True:

        source_terms_int = solver_eliminate_ghost(solver_param,state['source_terms'])

        state['source_terms'] = source_terms_int[rom_param['S_indx_solver']]

    state['d_flux_dx'] = state['d_flux_dx'] + state['source_terms']

    return state

def residual_calculator(solver_param,rom_param,state):
    
    # compute flux
    if solver_param['flux_scheme'] == 'Rusanov':

        state = rusanov_flux_calculator(solver_param,state)

        if solver_param['viscous_flag']:

            state = first_order_rusanov_viscous_flux_calculator(solver_param,rom_param,state)

    elif solver_param['flux_scheme'] == '1st Order Roe':

        state = first_order_roe_inviscid_flux_calculator(solver_param,rom_param,state)

        if solver_param['viscous_flag']:

            state = first_order_roe_viscous_flux_calculator(solver_param,rom_param,state)

    elif solver_param['flux_scheme'] == '2nd Order Roe':

        state = second_order_roe_flux_calculator(solver_param,rom_param,state)

        if solver_param['viscous_flag']:

            state = first_order_roe_viscous_flux_calculator(solver_param,rom_param,state)

    # inviscid flux vector
    state   = d_flux_dx_calculator(solver_param,rom_param,state)

    if solver_param['gas_model'] != 'Non-Reacting Air':

        state = source_calculator(solver_param,rom_param,state)

    return state









    

