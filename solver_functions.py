import numpy as np

def solver_parameters_collector(self):

    solver_param = {}

    ### solver_mode ###
    if self.fom_mode_checkbox_check_var.get() == True:

        solver_param['solver_mode'] = 'FOM'

    elif self.rom_mode_checkbox_check_var.get() == True:

        solver_param['solver_mode'] = 'ROM'
        solver_param['pod_energy'] = float(self.energy_capture_entry_var.get())

    elif self.adaptive_rom_mode_checkbox_check_var.get() == True:

        solver_param['solver_mode'] = 'Adaptive ROM'
        solver_param['adaptive_rom_method'] = self.adaptive_rom_method_entry_var.get()
        solver_param['pod_energy'] = float(self.energy_capture_entry_var.get())
        solver_param['init_training_win'] = self.training_window_entry_var.get()
        solver_param['unsampled_update_freq'] = int(self.unsampled_update_freq_entry_var.get())
        
    solver_param['rom_method'] = self.rom_method_entry_var.get()
    solver_param['hyper']      = self.hyper_method_checkbox_check_var.get()
    solver_param['sampling_method'] = self.hyper_method_entry_var.get()


    ### time discretization ###
    solver_param['dt'] = float(self.dt_entry_var.get())
    solver_param['num_step'] = int(self.num_step_entry_var.get())
    solver_param['time_scheme'] = self.time_scheme_entry_var.get()
    solver_param['dual_time'] = self.dt_entry_var.get()
    solver_param['res_tol'] = float(self.dt_entry_var.get())

    ### space discretization ###
    solver_param['x_initial'] = float(self.mesh_x_init_entry_var.get())
    solver_param['x_final'] = float(self.mesh_x_final_entry_var.get())
    solver_param['cell_number'] = int(self.cell_number_entry_var.get())

    ### inlet BC ###
    # solver_param['press_inlet'] = float(self.inlet_press_entry_var.get())
    # solver_param['temp_inlet'] = float(self.inlet_temp_entry_var.get())
    # solver_param['vel_inlet'] = float(self.inlet_vel_entry_var.get())
    # solver_param['rho_inlet'] = float(self.inlet_rho_entry_var.get())
    # solver_param['mass_frac_inlet'] = self.inlet_mass_frac_entry_var.get()

    # ### outlet BC ###
    # solver_param['press_outlet'] = float(self.outlet_press_entry_var.get())
    # solver_param['temp_outlet'] = float(self.outlet_temp_entry_var.get())
    # solver_param['vel_outlet'] = float(self.outlet_vel_entry_var.get())
    # solver_param['rho_outlet'] = float(self.outlet_rho_entry_var.get())
    # solver_param['mass_frac_outlet'] = self.outlet_mass_frac_entry_var.get()

    solver_param['non_reflective_bc'] = True

    ### physics ###
    solver_param['gas_model']  = self.gas_model_entry_var.get()
    solver_param['gamma']      = 1.4 
    solver_param['flux_scheme']= self.flux_scheme_entry_var.get()
    solver_param['limiter']    = self.limiter_checkbox_check_var.get()

    ### visualization ###
    solver_param['variable1']           = self.visual_1_option_entry_var.get()
    solver_param['variable2']           = self.visual_2_option_entry_var.get()
    solver_param['variable3']           = self.visual_3_option_entry_var.get()
    solver_param['variable4']           = self.visual_4_option_entry_var.get()
    solver_param['vis_update_interval'] = int(self.visual_update_interval_entry_var.get())
    solver_param['plot_fom_flag']       = self.fom_plot_check_var.get()

    ### ic data ###
    solver_param['ic_data']   = self.ic_data

    ### Input Directory ###
    solver_param['working_dir'] = self.working_dir_entry_var.get()
    solver_param['FOM_result_dir'] = self.FOM_file_entry_var.get()

    ### Some Basic Calculations ###

    solver_param['dx']      = (solver_param['x_final'] - solver_param['x_initial']) / solver_param['cell_number']
    solver_param['vol']     = solver_param['dx']
    solver_param['x']       = np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number']) )

    return solver_param

def initialize_state(solver_param):

    # create state variable
    state = {}

    state['cons_results_save']      = np.zeros(( 3 , int(solver_param['cell_number'])    , int(int(solver_param['num_step'])) )  )
    state['prim_results_save']      = np.zeros(( 3 , int(solver_param['cell_number'])    , int(int(solver_param['num_step'])))  )

    state['Q_cons']                 = np.zeros((3*int(solver_param['cell_number'])+(3*4)))
    state['Q_prim']                 = np.zeros((3*int(solver_param['cell_number'])+(3*4)))

    return state

def ic_generator(solver_param,state):

    num_region = len(solver_param['ic_data'])
    num_cell   = solver_param['cell_number']
    x          =  np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number'])+4 )

    rho = np.zeros(num_cell+4)
    vx  = np.zeros(num_cell+4)
    P   = np.zeros(num_cell+4)

    for region in range(0,num_region):
        
        indx = np.where(  (x >= float(solver_param['ic_data'][region][0]))  &  (x <= float(solver_param['ic_data'][region][1]))  )
        
        rho[indx] = eval(solver_param['ic_data'][region][5])
        vx [indx] = eval(solver_param['ic_data'][region][4])
        P  [indx] = eval(solver_param['ic_data'][region][2])
    
    state['Q_prim'] = np.vstack((rho,vx,P)).ravel()

    return state

def update_ghost_cell(solver_param,state):

    num_cell = solver_param['cell_number']

    Q_prim   = state['Q_prim']
    Q_cons   = state['Q_cons']

    Q_prim_user = results_solver2user_converter(num_cell,Q_prim)
    Q_cons_user = results_solver2user_converter(num_cell,Q_cons)
    
    if solver_param['non_reflective_bc']:

        Q_prim_user[:,0:2]   = Q_prim_user[:,3].reshape(-1, 1)

        Q_prim_user[:,-2:]   = Q_prim_user[:,-3].reshape(-1, 1)

        Q_cons_user[:,0:2]   = Q_cons_user[:,3].reshape(-1, 1)

        Q_cons_user[:,-2:]   = Q_cons_user[:,-3].reshape(-1, 1)

    elif solver_param['periodic_bc']:

        Q_prim_user[:,0:2]   = Q_prim_user[:,-4:-2]
        Q_prim_user[:,-2:]   = Q_prim_user[:,2:4]
        Q_prim_user[:,1]     = Q_prim_user[:,-3]
        Q_prim_user[:,-2]    = Q_prim_user[:,2]

        Q_cons_user[:,0:2]   = Q_cons_user[:,-4:-2]
        Q_cons_user[:,-2:]   = Q_cons_user[:,2:4]
        Q_cons_user[:,1]     = Q_cons_user[:,-3]
        Q_cons_user[:,-2]    = Q_cons_user[:,2]


    Q_prim_solver = results_user2solver_converter(Q_prim_user)
    Q_cons_solver = results_user2solver_converter(Q_cons_user)
    
    state['Q_prim'] = Q_prim_solver
    state['Q_cons'] = Q_cons_solver

    return state
    
def prim2cons_converter(solver_param, state):

    num_cell = solver_param['cell_number']

    Q_prim   = state['Q_prim']

    Q_prim_user = results_solver2user_converter(num_cell,Q_prim)

    vol = solver_param['vol']

    gamma = solver_param['gamma']

    rho = Q_prim_user[0,:]
    vx  = Q_prim_user[1,:] 
    P   = Q_prim_user[2,:]

    mass   = rho * vol
    momx   = rho * vx * vol
    energy = (P/(gamma-1) + 0.5 * rho * (vx**2)) * vol

    state['Q_cons'] = np.vstack((mass,momx,energy)).ravel()

    return state

def cons2prim_converter(solver_param, state):

    vol         = solver_param['vol']
    gamma       = solver_param['gamma']
    Q_cons      = state['Q_cons']
    Q_cons_user = results_solver2user_converter(solver_param['cell_number'],Q_cons)
    mass        = Q_cons_user[0,:]
    momx        = Q_cons_user[1,:]
    energy      = Q_cons_user[2,:]

    rho = mass / vol
    vx  = momx / rho / vol
    P   = (energy/vol - (0.5 * rho * (vx**2))) * (gamma-1)

    state['Q_prim'] = np.vstack((rho,vx,P)).ravel()

    state = update_ghost_cell(solver_param,state)

    return  state

def gradient_calculator(var,dx):

    df_dx = (np.roll(var,-1) - np.roll(var,1)) / (2*dx)

    return df_dx

def slope_limit(f, dx, df_dx):

	df_dx = np.maximum(0., np.minimum(1., ( (f-np.roll(f,1,axis=0))/dx)/(df_dx + 1.0e-8*(df_dx==0)))) * df_dx
	df_dx = np.maximum(0., np.minimum(1., (-(f-np.roll(f,-1,axis=0))/dx)/(df_dx + 1.0e-8*(df_dx==0)))) * df_dx

	return df_dx

def extrapolate_center2face( var , d_var , dx):

    # var_left_face  = var - (d_var * dx/2)
    # var_left_face  = np.roll(var_left_face,-1)
    # var_right_face = var + (d_var * dx/2)

    var_left_face  = np.zeros(len(var)+1)
    var_right_face = np.zeros(len(var)+1)

    for indx in range(1,len(var)):

        var_left_face[indx]  = var[indx-1] + d_var[indx-1] * (dx/2)
        var_right_face[indx] = var[indx] - d_var[indx] * (dx/2)

    return var_left_face , var_right_face

def rusanov_flux_calculator(mass , momx , energy ,gamma , vol , dx , slope_limiter , sol_time):

    # convert cons to prim
    rho , vx , p = cons2prim_converter(mass , momx , energy , gamma , vol,sol_time)

    # calculate gradients
    d_rho_dx = gradient_calculator(rho , dx)
    d_vx_dx  = gradient_calculator(vx , dx)
    d_p_dx   = gradient_calculator(p , dx)

    if slope_limiter:

        d_rho_dx = slope_limit(rho , dx , d_rho_dx)
        d_vx_dx  = slope_limit(vx  , dx , d_vx_dx )
        d_p_dx   = slope_limit(p   , dx , d_p_dx  )

    # extrapolate from center to face
    rho_face_left , rho_face_right = extrapolate_center2face(rho,d_rho_dx,dx)
    vx_face_left  , vx_face_right  = extrapolate_center2face(vx,d_vx_dx,dx)
    p_face_left   , p_face_right   = extrapolate_center2face(p,d_p_dx,dx)

    # start flux calculation
    en_L = p_face_left  / (gamma-1) + 0.5 * rho_face_left  * (vx_face_left**2)
    en_R = p_face_right / (gamma-1) + 0.5 * rho_face_right * (vx_face_right**2)

    C_L = np.sqrt(gamma*p_face_left/rho_face_left)
    C_R = np.sqrt(gamma*p_face_right/rho_face_right)   

    C_mean = 0.5*(C_L+C_R)
    vx_mean = 0.5*(vx_face_left+vx_face_right)

    diffusion =  0.5 * (C_mean + vx_mean)

    flux_mass   = 0.5*(rho_face_left*vx_face_left + rho_face_right*vx_face_right)                               - diffusion*(rho_face_right-rho_face_left)
    flux_momx   = 0.5*(rho_face_left*vx_face_left**2+p_face_left + rho_face_right*vx_face_right**2+p_face_right)- diffusion*(rho_face_right*vx_face_right-rho_face_left*vx_face_left)
    flux_energy = 0.5*(vx_face_left*(en_L+p_face_left) + vx_face_right*(en_R+p_face_right))                     - diffusion*(en_R-en_L)

    return flux_mass, flux_momx, flux_energy

def roe_flux_calculator(solver_param,rom_param,state):
    
    Q_cons           = state['Q_cons']
    state            = cons2prim_converter(solver_param,state)
    Q_prim           = state['Q_prim']
    Q_cons_user      = results_solver2user_converter(solver_param['cell_number'],Q_cons)
    Q_prim_user      = results_solver2user_converter(solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]

    gamma  = solver_param['gamma']
    vol    = solver_param['vol']

    S_indx_user = rom_param['S_indx_user']

    en               = press  / (gamma-1) + 0.5 * rho  * (vx**2)
    # number of cell
    cell_num = len(rho)

    # total enthalpy
    htot = gamma/(gamma-1)*press/rho+0.5*vx**2

    flux = np.zeros((3,cell_num+1))
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
        flux[1,j+1] = 0.5*(rho[j]*vx[j]**2 + press[j] + rho[j+1]*vx[j+1]+press[j+1])              - diffusion[1,j+1] 
        flux[2,j+1] = 0.5*(vx[j]*(en[j]+press[j])     + vx[j+1]*(en[j+1]+press[j+1]))             - diffusion[2,j+1] 

    state['flux_cons'] = flux

    return state

def inviscid_d_flux_dx_calculator(solver_param,rom_param,state):

    S_indx_user = rom_param['S_indx_user']

    flux = state['flux_cons']

    if solver_param['hyper'] == True:

        d_flux_dx = np.zeros((3 , len(S_indx_user)))

        S_indx_user = S_indx_user + 2

        counter = 0

        for indx in S_indx_user:

            d_flux_dx[:,counter] = -(flux[:,indx+1] - flux[:,indx])
            counter = counter + 1

    else:

        d_flux_dx = np.zeros((3 , solver_param['cell_number'] + 4))

        for indx in range(0,len(d_flux_dx[0,:])):

            d_flux_dx[:,indx] = -(flux[:,indx+1] - flux[:,indx])
    
    state['d_flux_dx'] = d_flux_dx.ravel()
    
    return state

def results_solver2user_converter(num_cell,Q):

    Q_reshaped = np.reshape(Q,(3,num_cell+4))

    return Q_reshaped

def results_user2solver_converter(Q):

    Q_reshaped = np.ravel(Q)

    return Q_reshaped

def solver_eliminate_ghost(solver_param,Q_full):

    Q_full_user = results_solver2user_converter(solver_param['cell_number'],Q_full)
    Q_int_user = Q_full_user[:,2:-2]
    Q_int_solver = results_user2solver_converter(Q_int_user)

    return Q_int_solver

def solver_add_ghost(solver_param,Q_int):

    Q_int_user  = results_solver2user_converter(solver_param['cell_number']-4,Q_int)
    Q_int_full = np.column_stack((Q_int_user[:,0], Q_int_user[:,0], Q_int_user , Q_int_user[:,-1] , Q_int_user[:,-1]))
    Q_full_solver= results_user2solver_converter(Q_int_full)

    return Q_full_solver

def residual_calculator(solver_param,rom_param,state):
    
    # compute flux
    if solver_param['flux_scheme'] == 'Rusanov':
        state = rusanov_flux_calculator(solver_param,state)

    elif solver_param['flux_scheme'] == 'Roe':
        state = roe_flux_calculator(solver_param,rom_param,state)

    # inviscid flux vector terms
    state   = inviscid_d_flux_dx_calculator(solver_param,rom_param,state)

    return state









    

