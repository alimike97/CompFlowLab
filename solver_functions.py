import numpy as np

def solver_parameters_collector(self):

    solver_param = {}

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
    solver_param['press_inlet'] = float(self.inlet_press_entry_var.get())
    solver_param['temp_inlet'] = float(self.inlet_temp_entry_var.get())
    solver_param['vel_inlet'] = float(self.inlet_vel_entry_var.get())
    solver_param['rho_inlet'] = float(self.inlet_rho_entry_var.get())
    solver_param['mass_frac_inlet'] = self.inlet_mass_frac_entry_var.get()

    ### outlet BC ###
    solver_param['press_outlet'] = float(self.outlet_press_entry_var.get())
    solver_param['temp_outlet'] = float(self.outlet_temp_entry_var.get())
    solver_param['vel_outlet'] = float(self.outlet_vel_entry_var.get())
    solver_param['rho_outlet'] = float(self.outlet_rho_entry_var.get())
    solver_param['mass_frac_outlet'] = self.outlet_mass_frac_entry_var.get()

    ### gas model ###
    solver_param['gas_model'] = self.gas_model_entry_var.get()

    ### ROM ###
    solver_param['calc_rom'] = self.rom_method_checkbox_check_var.get()
    solver_param['rom_method'] = self.rom_method_entry_var.get()
    solver_param['pod_energy'] = float(self.energy_capture_entry_var.get())

    ### Hyper-Reduction ###
    solver_param['hyper'] = self.hyper_method_checkbox_check_var.get()
    solver_param['hyper_method'] = self.hyper_method_entry_var.get()

    ### visualization ###
    solver_param['variable1'] = self.visual_1_option_entry_var.get()
    solver_param['variable2'] = self.visual_2_option_entry_var.get()
    solver_param['variable3'] = self.visual_3_option_entry_var.get()
    solver_param['variable4'] = self.visual_4_option_entry_var.get()

    ### ic data ###
    solver_param['ic_data']   = self.ic_data

    ### Input Directory ###
    solver_param['working_dir'] = self.working_dir_entry_var.get()
    # solver_param['FOM_Result'] = self.FOM_re

    return solver_param

def ic_generator(solver_param):

    num_region = len(solver_param['ic_data'])
    num_cell   = solver_param['cell_number']
    x       =  np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number'])+4 )

    rho = np.zeros(num_cell+4)
    vx  = np.zeros(num_cell+4)
    P   = np.zeros(num_cell+4)

    for region in range(0,num_region):
        
        indx = np.where(  (x >= float(solver_param['ic_data'][region][0]))  &  (x <= float(solver_param['ic_data'][region][1]))  )
        
        rho[indx] = eval(solver_param['ic_data'][region][5])
        vx [indx] = eval(solver_param['ic_data'][region][4])
        P  [indx] = eval(solver_param['ic_data'][region][2])
         
    return rho,vx,P


def add_ghost_cell(rho , vx , P ,sol_time):

    rho[0:2]   = rho[3]
    vx[0:2]    = vx[3]
    P[0:2]     = P[3]

    rho[-2:] = rho[-3]
    vx[-2:]  = vx[-3]
    P[-2:]   = P[-3]

    
    # rho[0:2]   = rho[-4:-2]
    # vx[0:2]    = vx[-4:-2]
    # P[0:2]     = P[-4:-2]

    # rho[-2:] = rho[2:4]
    # vx[-2:]  = vx[2:4]
    # P[-2:]   = P[2:4]

    # rho[1]   = rho[-3]
    # vx[1]    = vx[-3]
    # P[1]     = P[-3]

    # rho[-2] = rho[2]
    # vx[-2]  = vx[2]
    # P[-2]   = P[2]

    # if periodic_inlet:
         
    #     pass
         
    # if periodic_outlet:
         
    # pert =  0.0001* np.sin(2*np.pi*1e6*sol_time)
    # P[-3] = P[-3] * (1 + pert)
 
    return rho,vx,P
    

def prim2cons_converter(rho , vx , P , gamma , vol):

    mass   = rho 
    momx   = rho * vx 
    energy = (P/(gamma-1) + 0.5 * rho * (vx**2)) 

    return mass , momx , energy

def cons2prim_converter(mass , momx , energy , gamma , vol,sol_time):

    rho = mass 
    vx  = momx / rho 
    P   = (energy - (0.5 * rho * (vx**2))) * (gamma-1)

    rho , vx , P = add_ghost_cell(rho,vx,P,sol_time)

    return  rho , vx , P

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
    for indx in range(1,len(var)-1):

        var_left_face[indx]  = var[indx-1] + d_var[indx-1] * (dx/2)
        var_right_face[indx] = var[indx] - d_var[indx] * (dx/2)

    return var_left_face , var_right_face

def rusanov_flux_calculator(rho_face_left , rho_face_right , vx_face_left , vx_face_right , p_face_left , p_face_right , gamma):

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

def roe_flux_calculator(mass , momx , energy , gamma , vol):

    
    Q_cons       = np.vstack((mass , momx , energy))
    rho , vx , press = cons2prim_converter(mass,momx,energy,gamma,vol,0)

    # number of cell
    cell_num = len(rho)

    # total enthalpy
    htot = gamma/(gamma-1)*press/rho+0.5*vx**2

    flux = np.zeros((3,cell_num+1))
    diffusion = np.zeros((3,cell_num+1))

    for j in range (0,cell_num-1):
    
        # Compute Roe averages
        R=np.sqrt(rho[j+1]/rho[j])                   # R_{j+1/2}
        rmoy=R*rho[j]                                # {hat rho}_{j+1/2}
        umoy=(R*vx[j+1]+vx[j])/(R+1)                # {hat U}_{j+1/2}
        hmoy=(R*htot[j+1]+htot[j])/(R+1);            # {hat H}_{j+1/2}
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

        diffusion[:,j+1] = 0.5 * A @ (Q_cons[:,j+1]-Q_cons[:,j])

        flux[0,j+1] = 0.5*(rho[j]*vx[j]               + rho[j+1]*vx[j+1])                         - diffusion[0,j+1]
        flux[1,j+1] = 0.5*(rho[j]*vx[j]**2 + press[j] + rho[j+1]*vx[j+1]+press[j+1])              - diffusion[1,j+1]
        flux[2,j+1] = 0.5*(vx[j]*(energy[j]+press[j]) + vx[j+1]*(energy[j+1]+press[j+1]))         - diffusion[2,j+1]


    flux_mass   = flux[0,:]
    flux_momx   = flux[1,:]
    flux_energy = flux[2,:]

    return flux_mass, flux_momx, flux_energy

def inviscid_d_flux_dx_calculator(flux , dx):
    
    d_flux_dx = np.zeros(len(flux)-1)

    for indx in range(0,len(d_flux_dx)-1):

        d_flux_dx[indx] = -(flux[indx+1] - flux[indx])/dx

    return d_flux_dx
