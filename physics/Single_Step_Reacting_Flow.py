import numpy as np
import cantera as ct
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.interpolate import RegularGridInterpolator


from utils import reshape_func
from boundary_condition import bc_func
 
def prim2cons_converter(solver_param, state):

    num_cell = solver_param['cell_number']

    Q_prim   = state['Q_prim']

    Q_prim_user = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],num_cell,Q_prim)

    vol = solver_param['vol']

    rho = Q_prim_user[0,:]
    vx  = Q_prim_user[1,:] 
    P   = Q_prim_user[2,:]
    T   = Q_prim_user[3,:]
    Y   = Q_prim_user[4:,:]

    Y_full=reshape_func.find_mass_fraction_full(Y)

    Cp_mix    = state['gas_lookup_table']['cp']
    MW_mix    = state['gas_lookup_table']['MW']
    R_mix     = state['gas_lookup_table']['R_univ'] / MW_mix
    Cv_mix    = Cp_mix - R_mix
    h_ref     = np.array(state['gas_lookup_table']['href'])[:,np.newaxis]
    h_f_mix   = np.sum(h_ref * Y_full, axis=0)


    mass   = rho * vol
    momx   = rho * vx * vol
    energy = rho * ((T*Cv_mix+h_f_mix) + 0.5 *(vx**2))*vol
    mass_species = rho * Y * vol

    state['Q_cons'] = np.vstack((mass,momx,energy,mass_species)).ravel()

    return state

def cons2prim_converter(solver_param, state):

    vol         = solver_param['vol']

    Q_cons      = state['Q_cons']
    Q_cons_user = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

    mass        = Q_cons_user[0,:]
    momx        = Q_cons_user[1,:]
    energy      = Q_cons_user[2,:]
    mass_species= Q_cons_user[3:,:]

    rho = mass / vol
    vx  = momx / rho / vol
    MF  = mass_species / rho / vol

    Y_full=reshape_func.find_mass_fraction_full(MF)

    Cp_mix    = state['gas_lookup_table']['cp']
    MW_mix    = state['gas_lookup_table']['MW']
    R_mix     = state['gas_lookup_table']['R_univ'] / MW_mix
    Cv_mix    = Cp_mix - R_mix
    h_ref     = np.array(state['gas_lookup_table']['href'])[:,np.newaxis]
    h_f_mix   = np.sum(h_ref * Y_full, axis=0)

    T  = (energy/rho/vol - (0.5 *(vx**2)) - h_f_mix) / Cv_mix

    P = rho * R_mix * T
 
    state['Q_prim'] = np.vstack((rho,vx,P,T,MF)).ravel()

    return state

def cache_cantera(solver_param,state):


    Q_prim           = state['Q_prim']
    Q_prim_user      = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    T     = Q_prim_user[3,:]
    Y     = Q_prim_user[4:,:]
    Y_full=reshape_func.find_mass_fraction_full(Y)

    Cp_mix    = state['gas_lookup_table']['cp']
    MW_mix    = state['gas_lookup_table']['MW']
    R_mix     = state['gas_lookup_table']['R_univ'] / MW_mix
    Cv_mix    = Cp_mix - R_mix
    gamma_mix = Cp_mix/Cv_mix
    h_ref     = np.array(state['gas_lookup_table']['href'])[:,np.newaxis]
    h_f_mix   = np.sum(h_ref * Y_full, axis=0)

    state['gas_cached_props'] = {}

    state['gas_cached_props']['sound_speed']        = np.sqrt(gamma_mix*R_mix*T)
    state['gas_cached_props']['int_energy_mass']    = Cv_mix * T + h_f_mix
    state['gas_cached_props']['enthalpy']           = Cp_mix * T + h_ref
    state['gas_cached_props']['enthalpy_mix']       = Cp_mix * T + h_f_mix
    state['gas_cached_props']['MW_mix']             = MW_mix
    state['gas_cached_props']['cp']                 = Cp_mix
    state['gas_cached_props']['cv']                 = Cv_mix

    return state

def cache_roe_cantera(state,emoy,rmoy,Ymoy):

    Ymoy_full = reshape_func.find_mass_fraction_full(Ymoy)

    Cp_moy    = state['gas_lookup_table']['cp']
    MW_moy    = state['gas_lookup_table']['MW']
    R_moy     = state['gas_lookup_table']['R_univ'] / MW_moy
    Cv_moy    = Cp_moy - R_moy
    gamma_moy = Cp_moy/Cv_moy
    h_ref     = np.array(state['gas_lookup_table']['href'])[:,np.newaxis]
    h_f_moy   = np.sum(h_ref * Ymoy_full, axis=0)


    Tmoy      = (emoy-h_f_moy) / Cv_moy
    Pmoy      = rmoy * R_moy * Tmoy
    cmoy      = np.sqrt(gamma_moy*R_moy*Tmoy)

    enthalpies= Cp_moy*Tmoy + h_ref

    dyn_vis_mix_moy = np.zeros_like(Tmoy) + state['gas_lookup_table']['mu']
    
    Pr   = state['gas_lookup_table']['Pr']

    k_mix_moy = dyn_vis_mix_moy * Cp_moy / Pr

    D_mix_moy =  k_mix_moy / rmoy / Cp_moy

    state['gas_roe_cached_props'] = {
        'Pmoy': Pmoy,
        'Tmoy': Tmoy,
        'cpmoy': Cp_moy,
        'cmoy': cmoy,
        'MWmoy': MW_moy,
        'dyn_vsc_mix': dyn_vis_mix_moy,
        'therm_cond_mix': k_mix_moy,
        'diff_mix': D_mix_moy,
        'enthalpies': enthalpies,
    }

    return state

def second_order_roe_inviscid_flux_calculator(solver_param,rom_param,state):
    
    Q_cons           = state['Q_cons']
    Q_prim           = state['Q_prim']

    Q_cons_user      = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]


    Y      = Q_prim_user[4:,:]


    vol           = solver_param['vol']
    num_state_var = solver_param['num_state_var']


    en               = Q_cons_user[2,:] / vol

    c       = state['gas_cached_props']['sound_speed']
    int_en  = state['gas_cached_props']['int_energy_mass']
    h       = state['gas_cached_props']['enthalpy_mix']

    # total enthalpy
    htot = h + (0.5*(vx**2))

    # number of cell
    cell_num = len(rho)

    flux = np.zeros((num_state_var,cell_num+1))
    
    if solver_param['hyper'] == True:

        S_indx_user = rom_param['S_indx_user']

        S_indx_user      = S_indx_user + 2
        range_flux_right = np.sort(np.concatenate((S_indx_user,S_indx_user+1)))
        range_flux_left  = range_flux_right - 1

        rho_left  = rho[range_flux_left]
        rho_right = rho[range_flux_right]

        vx_left  = vx[range_flux_left]
        vx_right = vx[range_flux_right]

        htot_left  = htot[range_flux_left]
        htot_right = htot[range_flux_right]

        press_left  = press[range_flux_left]
        press_right = press[range_flux_right]

        temp_left  = temp[range_flux_left]
        temp_right = temp[range_flux_right]

        en_left    = en[range_flux_left]
        en_right   = en[range_flux_right]

        int_en_left    = int_en[range_flux_left]
        int_en_right   = int_en[range_flux_right]

        Y_left         = Y[:,range_flux_left]
        Y_right        = Y[:,range_flux_right]

        del_q_prim = np.zeros((2*len(S_indx_user),num_state_var,1))

        diss_matrix = np.zeros((2*len(S_indx_user),num_state_var,num_state_var))

        # gas_array = ct.SolutionArray(state['gas'],2*len(S_indx_user))

    else : 
        
        rho_left  = rho[:-1]
        rho_right = rho[1:]

        vx_left  = vx[:-1]
        vx_right = vx[1:]

        htot_left  = htot[:-1]
        htot_right = htot[1:]

        press_left  = press[:-1]
        press_right = press[1:]

        temp_left  = temp[:-1]
        temp_right = temp[1:]

        en_left    = en[:-1]
        en_right   = en[1:]

        int_en_left  = int_en[:-1]
        int_en_right = int_en[1:]

        Y_left         = Y[:,:-1]
        Y_right        = Y[:,1:]

        del_q_prim = np.zeros((cell_num-1,num_state_var,1))

        diss_matrix = np.zeros((cell_num-1,num_state_var,num_state_var))

        # gas_array = ct.SolutionArray(state['gas'],cell_num-1)
    
    R   = np.sqrt(rho_right/rho_left)

    rmoy=R*rho_left                                   # {hat rho}_{j+1/2}
    umoy=(R*vx_right+vx_left)/(R+1)                   # {hat U}_{j+1/2}
    Hmoy=(R*htot_right+htot_left)/(R+1)               # {hat H}_{j+1/2}
    emoy=(R*int_en_right+int_en_left)/(R+1)              
    hmoy=Hmoy - (0.5*umoy*umoy)
    Ymoy=(R*Y_right+Y_left)/(R+1)                  

    # Ymoy_ct = reshape_func.find_mass_fraction_full_cantera(Ymoy)


    state = cache_roe_cantera(state,emoy,rmoy,Ymoy)

    Pmoy         = state['gas_roe_cached_props']['Pmoy']
    Tmoy         = state['gas_roe_cached_props']['Tmoy']
    cpmoy        = state['gas_roe_cached_props']['cpmoy']
    cmoy         = state['gas_roe_cached_props']['cmoy']

    # MWmoy        = state['gas_cached_props']['molecular_weights']
    meanMWmoy    = state['gas_roe_cached_props']['MWmoy']
    partial_hmoy = state['gas_roe_cached_props']['enthalpies'].T

        
    d_rho_d_press = rmoy / Pmoy
    d_rho_d_temp = -rmoy/Tmoy

    # d_rho_d_mass_frac = np.zeros_like(Ymoy)

    #     for sp in range(len(Ymoy)):

    #         d_rho_d_mass_frac[sp] = rmoy*meanMWmoy*(1/MWmoy[-1] - 1/MWmoy[sp])
    d_rho_d_mass_frac_const_term =  0
    d_rho_d_mass_frac = rmoy * meanMWmoy * d_rho_d_mass_frac_const_term

    d_enth_d_press = 0
    d_enth_d_temp  = cpmoy

    # d_enth_d_mass_frac = np.zeros_like(Ymoy)

    # for sp in range(len(Ymoy)):
    
    d_enth_d_mass_frac = partial_hmoy[:,:-1] - partial_hmoy[:,-1,np.newaxis]

    #     # Gamma terms for energy equation
    g_press     = rmoy * d_enth_d_press + d_rho_d_press * Hmoy - 1.0
    g_temp      = rmoy * d_enth_d_temp + d_rho_d_temp * Hmoy
    g_mass_frac = rmoy[:,np.newaxis] * d_enth_d_mass_frac + Hmoy[:,np.newaxis] * d_rho_d_mass_frac[:,None]

    #     # Characteristic speeds
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

    delta_p = press_left-press_right
    delta_u = vx_left-vx_right
    delta_T = temp_left-temp_right
    delta_Y = (Y_left-Y_right)

    # del_q_prim = np.zeros((cell_num-1,num_state_var,1))

    del_q_prim[:,0,:]  = delta_p.reshape(-1,1)
    del_q_prim[:,1,:]  = delta_u.reshape(-1,1)
    del_q_prim[:,2,:]  = delta_T.reshape(-1,1)
    del_q_prim[:,3:,:] = delta_Y.T[:,:,np.newaxis]


    diss_matrix[: , 0 , 0 ] = phi_star
    diss_matrix[: , 0 , 1 ] = beta_star
    diss_matrix[: , 0 , 2 ] = vel_abs * d_rho_d_temp
    diss_matrix[: , 0 , 3:] = (vel_abs* d_rho_d_mass_frac)[:,None]

    diss_matrix[: , 1 , 0 ] = umoy * phi_star + r_roe
    diss_matrix[: , 1 , 1 ] = umoy * beta_star + m
    diss_matrix[: , 1 , 2 ] = umoy * vel_abs * d_rho_d_temp
    diss_matrix[: , 1 , 3:] = ((umoy * vel_abs) * d_rho_d_mass_frac)[:,None]

    diss_matrix[: , 2 , 0 ] = phi_e + r_roe * umoy
    diss_matrix[: , 2 , 1 ] = beta_e + e
    diss_matrix[: , 2 , 2 ] = g_temp * vel_abs
    diss_matrix[: , 2 , 3:] = g_mass_frac * vel_abs[:,np.newaxis]

    diss_matrix[: , 3:, 0] = Ymoy.T * phi_star[:,np.newaxis]
    diss_matrix[: , 3:, 1] = Ymoy.T * beta_star[:,np.newaxis]
    diss_matrix[: , 3:, 2] = Ymoy.T * (vel_abs * d_rho_d_temp)[:,np.newaxis]


    idx_out, idx_in = np.meshgrid(np.arange(num_state_var-3), 
                                np.arange(num_state_var-3), indexing='ij')
    
    diag_mask = idx_out == idx_in

    off_diag = vel_abs[:,None] * Ymoy.T * d_rho_d_mass_frac[:,None]
    off_diag = np.repeat(off_diag[:,:,np.newaxis],np.shape(off_diag)[1],axis=2)

    diag = vel_abs[:,np.newaxis] * (rmoy[:,np.newaxis] + Ymoy.T * d_rho_d_mass_frac[:,None])

    diss_matrix[:,3:,3:] = off_diag
    diss_matrix[:,3:,3:][:,diag_mask] = diag

    dissipation = np.matmul(diss_matrix , del_q_prim)

    if solver_param['hyper'] == True:

        flux[0 ,range_flux_right] = 0.5*(rho_left*vx_left                 + rho_right*vx_right)                 + 0.5 * dissipation[:,0 ,0]
        flux[1 ,range_flux_right] = 0.5*(rho_left*vx_left**2 + press_left + rho_right*vx_right**2+press_right)  + 0.5 * dissipation[:,1 ,0]
        flux[2 ,range_flux_right] = 0.5*(vx_left*(en_left+press_left)     + vx_right*(en_right+press_right))    + 0.5 * dissipation[:,2 ,0]
        flux[3:,range_flux_right] = 0.5*(rho_left*vx_left*Y_left          + rho_right*vx_right*Y_right )        + 0.5 * dissipation[:,3:,0].T


    else:

        flux[0,1:-1] = 0.5*(rho_left*vx_left                 + rho_right*vx_right)                 + 0.5 * dissipation[:,0 ,0]
        flux[1,1:-1] = 0.5*(rho_left*vx_left**2 + press_left + rho_right*vx_right**2+press_right)  + 0.5 * dissipation[:,1 ,0]
        flux[2,1:-1] = 0.5*(vx_left*(en_left+press_left)     + vx_right*(en_right+press_right))    + 0.5 * dissipation[:,2 ,0]
        flux[3:,1:-1]= 0.5*(rho_left*vx_left*Y_left          + rho_right*vx_right*Y_right )        + 0.5 * dissipation[:,3:,0].T
            
    state['flux_cons'] = flux

    return state

def second_order_roe_inviscid_flux_calculator_for(solver_param,rom_param,state):
    
    Q_cons           = state['Q_cons']
    Q_prim           = state['Q_prim']

    Q_cons_user      = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]


    Y      = Q_prim_user[4:,:]
    Y_ct   = reshape_func.find_mass_fraction_full_cantera(Y)


    vol    = solver_param['vol']
    num_state_var = solver_param['num_state_var']
    num_species   = solver_param['num_species']


    # total energy (rho * e_t)
    en               = Q_cons_user[2,:] / vol

    # state['gas_array'].TPY = temp,press,Y_ct

    c       = state['gas_cached_props']['sound_speed']    
    int_en  = state['gas_cached_props']['int_energy_mass']
    h       = state['gas_cached_props']['enthalpy_mix']       

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

        S_indx_user = rom_param['S_indx_user']

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

        Ymoy_full= reshape_func.find_mass_fraction_full(Ymoy)

        Cp_moy       = state['gas_lookup_table']['cp']
        MW_moy       = state['gas_lookup_table']['MW']
        R_moy        = state['gas_lookup_table']['R_univ'] / MW_moy
        Cv_moy       = Cp_moy - R_moy
        gamma_moy    = Cp_moy/Cv_moy
        h_ref        = np.array(state['gas_lookup_table']['href'])[:,np.newaxis]
        h_f_moy      = np.sum(h_ref * Ymoy_full, axis=0)


        Tmoy         = (emoy-h_f_moy) / Cv_moy
        Pmoy         = rmoy * R_moy * Tmoy
        cpmoy        = Cp_moy
        cmoy         = np.sqrt(gamma_moy*R_moy*Tmoy)
        meanMWmoy    = state['gas_lookup_table']['MW']
        MWmoy        = state['gas_lookup_table']['MW']
        partial_hmoy = Cp_moy*Tmoy + h_ref
        
        d_rho_d_press = rmoy / Pmoy
        d_rho_d_temp = -rmoy/Tmoy

        d_rho_d_mass_frac = np.zeros_like(Ymoy)

        for sp in range(len(Ymoy)):

            d_rho_d_mass_frac[sp] = 0

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

def second_order_roe_viscous_flux_calculator(solver_param,rom_param,state):

    Q_cons           = state['Q_cons']
    Q_prim           = state['Q_prim']

    Q_cons_user      = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]

    Y      = Q_prim_user[4:,:]
    Y_full = reshape_func.find_mass_fraction_full(Y)
    Y_ct   = reshape_func.find_mass_fraction_full_cantera(Y)


    vol           = solver_param['vol']
    dx            = vol
    num_state_var = solver_param['num_state_var']
    num_species   = solver_param['num_species']

    # total energy (rho * e_t)
    en               = Q_cons_user[2,:] / vol

    # state['gas_array'].TPY = temp,press,Y_ct

    c       = state['gas_cached_props']['sound_speed']
    int_en  = state['gas_cached_props']['int_energy_mass']
    h       = state['gas_cached_props']['enthalpy_mix']

    # total enthalpy
    htot = h + (0.5*(vx**2))

    # number of cell
    cell_num = len(rho)

    flux = np.zeros((num_state_var,cell_num+1))

    if solver_param['hyper'] == True:

        S_indx_user = rom_param['S_indx_user']

        S_indx_user      = S_indx_user + 2
        range_flux_right = np.sort(np.concatenate((S_indx_user,S_indx_user+1)))
        range_flux_left  = range_flux_right - 1

        rho_left  = rho[range_flux_left]
        rho_right = rho[range_flux_right]

        vx_left  = vx[range_flux_left]
        vx_right = vx[range_flux_right]

        htot_left  = htot[range_flux_left]
        htot_right = htot[range_flux_right]

        press_left  = press[range_flux_left]
        press_right = press[range_flux_right]

        temp_left  = temp[range_flux_left]
        temp_right = temp[range_flux_right]

        en_left    = en[range_flux_left]
        en_right   = en[range_flux_right]

        int_en_left    = int_en[range_flux_left]
        int_en_right   = int_en[range_flux_right]

        Y_left         = Y[:,range_flux_left]
        Y_right        = Y[:,range_flux_right]

        Y_full_left         = Y_full[:,range_flux_left]
        Y_full_right        = Y_full[:,range_flux_right]

        left_state = Q_cons_user[:,range_flux_left]
        right_state= Q_cons_user[:,range_flux_right]  
   
        # gas_array = ct.SolutionArray(state['gas'],2*len(S_indx_user))   

    else : 
        
        rho_left  = rho[:-1]
        rho_right = rho[1:]

        vx_left  = vx[:-1]
        vx_right = vx[1:]

        htot_left  = htot[:-1]
        htot_right = htot[1:]

        press_left  = press[:-1]
        press_right = press[1:]

        temp_left  = temp[:-1]
        temp_right = temp[1:]

        en_left    = en[:-1]
        en_right   = en[1:]

        int_en_left  = int_en[:-1]
        int_en_right = int_en[1:]

        Y_left         = Y[:,:-1]
        Y_right        = Y[:,1:]

        Y_full_left         = Y_full[:,:-1]
        Y_full_right        = Y_full[:,1:]

        left_state = Q_cons_user[:,:-1]
        right_state= Q_cons_user[:,1:]

        # gas_array = ct.SolutionArray(state['gas'],cell_num-1)


    R = np.sqrt(rho_right/rho_left)

    rmoy=R*rho_left                                   # {hat rho}_{j+1/2}
    umoy=(R*vx_right+vx_left)/(R+1)                   # {hat U}_{j+1/2}
    Hmoy=(R*htot_right+htot_left)/(R+1)               # {hat H}_{j+1/2}
    emoy=(R*int_en_right+int_en_left)/(R+1)              
    hmoy=Hmoy - (0.5*umoy*umoy)
    Ymoy=(R*Y_right+Y_left)/(R+1)                  

    Ymoy_ct = reshape_func.find_mass_fraction_full_cantera(Ymoy)

    # gas_array.UVY = emoy,1/rmoy,Ymoy_ct

    # gas_array                   = state['gas_array_roe']

    dyn_vsc_mix                 = state['gas_roe_cached_props']['dyn_vsc_mix']
    therm_cond_mix              = state['gas_roe_cached_props']['therm_cond_mix']
    enthalpies                  = state['gas_roe_cached_props']['enthalpies']
    mass_diff_mix               = state['gas_roe_cached_props']['diff_mix'][:,None]


    du_dx = (vx_right - vx_left) /dx
    dT_dx = (temp_right - temp_left) /dx
    dY_dx = (Y_full_right - Y_full_left) / dx

    corr_vel  = np.sum(mass_diff_mix*dY_dx.T,axis=1)

    diff_vel  = -mass_diff_mix*dY_dx.T/Ymoy_ct[0,:,:] + corr_vel[:,np.newaxis]

    tau       = 4/3 * dyn_vsc_mix * du_dx

    q         = -therm_cond_mix * dT_dx + rmoy*np.sum(enthalpies.T*diff_vel*Ymoy_ct[0,:,:],axis=1)

    if solver_param['hyper'] == True:

        flux[0 ,range_flux_right] = 0 
        flux[1 ,range_flux_right] = tau
        flux[2 ,range_flux_right] = (umoy * tau) - q
        flux[3:,range_flux_right] = (rmoy[:,np.newaxis]*mass_diff_mix*dY_dx[:-1].T).T-(rmoy*corr_vel*Ymoy)

    else: 

        flux[0 ,1:-1] = 0 
        flux[1 ,1:-1] = tau
        flux[2 ,1:-1] = (umoy * tau) - q
        flux[3:,1:-1] = (rmoy[:,np.newaxis]*mass_diff_mix*dY_dx[:-1].T).T-(rmoy*corr_vel*Ymoy)

        
    state['flux_visc_cons'] = flux

    return state

def second_order_roe_viscous_flux_calculator_for(solver_param,rom_param,state):

    # state            = cons2prim_converter(solver_param,state)
    # state            = update_ghost_cell(solver_param,state)

    Q_cons           = state['Q_cons']
    Q_prim           = state['Q_prim']

    Q_cons_user      = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho    = Q_prim_user[0,:]
    vx     = Q_prim_user[1,:]
    press  = Q_prim_user[2,:]
    temp   = Q_prim_user[3,:]

    Y      = Q_prim_user[4:,:]
    Y_full = reshape_func.find_mass_fraction_full(Y)
    Y_ct   = reshape_func.find_mass_fraction_full_cantera(Y)


    vol           = solver_param['vol']
    dx            = vol
    num_state_var = solver_param['num_state_var']
    num_species   = solver_param['num_species']



    # total energy (rho * e_t)
    en               = Q_cons_user[2,:] / vol

    # state['gas_array'].TPY = temp,press,Y_ct

    c       = state['gas_cached_props']['sound_speed']
    int_en  = state['gas_cached_props']['int_energy_mass']
    h       = state['gas_cached_props']['enthalpy_mix']

    # total enthalpy
    htot = h + (0.5*(vx**2))

    # number of cell
    cell_num = len(rho)

    flux = np.zeros((num_state_var,cell_num+1))

    if solver_param['hyper'] == True:

        S_indx_user = rom_param['S_indx_user']

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

        Ymoy_full = reshape_func.find_mass_fraction_full(Ymoy)

        # state['gas'].UVY            = emoy,1/rmoy,Ymoy_full

        Cp_moy              = state['gas_lookup_table']['cp']
        MW_moy              = state['gas_lookup_table']['MW']
        R_moy               = state['gas_lookup_table']['R_univ'] / MW_moy
        Cv_moy              = Cp_moy - R_moy
        gamma_moy           = Cp_moy/Cv_moy
        h_ref               = np.array(state['gas_lookup_table']['href'])[:,np.newaxis]
        h_f_moy             = np.sum(h_ref * Ymoy_full, axis=0)
            
        Tmoy                = (emoy-h_f_moy) / Cv_moy
        Pr                  = state['gas_lookup_table']['Pr']
        dyn_vsc_mix         = state['gas_lookup_table']['mu']
        therm_cond_mix      = dyn_vsc_mix * Cp_moy / Pr
        mass_diff_mix       = therm_cond_mix / rmoy / Cp_moy
        # MW_mix              = state['gas'].mean_molecular_weight
        enthalpies          = Cp_moy*Tmoy + h_ref


        du_dx = (vx[j+1] - vx[j]) /dx
        dT_dx = (temp[j+1] - temp[j]) /dx
        dY_dx = (Y_full[:,j+1] - Y_full[:,j]) / dx

        corr_vel  = np.sum(mass_diff_mix*dY_dx)

        diff_vel  = -mass_diff_mix*dY_dx[:,np.newaxis]/Ymoy_full + corr_vel

        tau       = 4/3 * dyn_vsc_mix * du_dx

        q         = -therm_cond_mix * dT_dx + rmoy*np.sum(enthalpies*diff_vel*Ymoy_full)

        flux[0,j+1] = 0 
        flux[1,j+1] = tau
        flux[2,j+1] = (umoy * tau) - q
        flux[3:,j+1]= (rmoy*mass_diff_mix*dY_dx[:-1])-(rmoy*corr_vel*Ymoy)

    state['flux_visc_cons'] = flux

    return state

def d_flux_dx_calculator(solver_param,rom_param,state):

    # apply viscous flux if exsted
    if solver_param['viscous_flag']:
        
        flux = state['flux_cons']-state['flux_visc_cons']

    else:

        flux = state['flux_cons']
    
    # apply the flux 
    if solver_param['hyper'] == True:

        S_indx_user      = rom_param['S_indx_user']
        S_indx_user      = S_indx_user + 2
        range_flux_right = S_indx_user + 1
        range_flux_left  = S_indx_user

        d_flux_dx = -(flux[:, range_flux_right] - flux[:, range_flux_left])

    else:

        d_flux_dx = -(flux[:, 1:] - flux[:, :-1])
    
    state['d_flux_dx'] = d_flux_dx.ravel()
    
    return state

def injection_correction(solver_param,state):

    # read current states
    Q_prim      = state['Q_prim']
    Q_prim_user = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    Q_cons      = state['Q_cons']
    Q_cons_user = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)

    rho         = Q_prim_user[0,:]
    u           = Q_prim_user[1,:]    
    P           = Q_prim_user[2,:]
    T           = Q_prim_user[3,:]
    Y           = Q_prim_user[4:,:]
    Y_full      = reshape_func.find_mass_fraction_full(Y)

    vol         = solver_param['vol']

    # read injection regions index
    add_indx    = state['injection_add_final']
    sub_indx    = state['injection_sub_init'] 

    # HR          = state['heat_release']

    indx_init   = np.argmax(np.abs(u)) - sub_indx
    indx_final  = indx_init + add_indx
    detonation  = np.arange(indx_init,indx_final)%(solver_param['cell_number']+4)
    inj_indx    = np.arange(0,solver_param['cell_number']+4,1)
    inj_indx    = inj_indx[~np.isin(inj_indx,detonation)]

    # compute injected fluid properties 
    # ref: properties before entering to injector
    # inj: properties after entering to injector

    rho_in          = solver_param['injcetion_prim_state'][0]
    v_in            = solver_param['injcetion_prim_state'][1]
    P_in            = solver_param['injcetion_prim_state'][2]
    T_in            = solver_param['injcetion_prim_state'][3]
    # Y_in            = solver_param['injcetion_prim_state'][4:] 
    Y_in            = 1
    Y_in_full       = Y_in + np.zeros_like(Y_full)
    Y_in_full[1,:]  = 0

    # compute injection rate 
    area_in     = solver_param['injector_face_area']

    mass_current= rho * vol
    mass_in     = rho_in * v_in * area_in * solver_param['dt']

    total_mass  = mass_current+mass_in

    # species mass conservation
    partial_mass_current = mass_current    * Y
    partial_mass_inject  = mass_in         * Y_in
    Y_mix                = (partial_mass_current + partial_mass_inject) / total_mass
    Y_mix_full           = reshape_func.find_mass_fraction_full(Y_mix)

    # energy conservation
    h_ref     = np.array(state['gas_lookup_table']['href'])[:,np.newaxis]
    h_f_in    = np.sum(h_ref * Y_in_full , axis=0)
    h_f_mix   = np.sum(h_ref * Y_mix_full, axis=0)
    cv        = state['gas_cached_props']['cv']

    int_energy_current    = state['gas_cached_props']['int_energy_mass'][inj_indx]
    int_energy_in         = np.zeros_like(T[inj_indx]) + cv * T_in + h_f_in[inj_indx]

    int_energy_mix        = ((mass_current[inj_indx] * int_energy_current) + (mass_in * int_energy_in))/total_mass[inj_indx]
    vol_inj               = mass_in / rho_in
    total_volume          = vol + vol_inj
    sp_volume_mix         = total_volume / total_mass[inj_indx]

    # mass conservation 
    rho_mix = 1 / sp_volume_mix

    # Pressure and Temperature of mix
    MW_mix= state['gas_lookup_table']['MW']
    R_mix = state['gas_lookup_table']['R_univ'] / MW_mix
    T_mix = (int_energy_mix - h_f_mix[inj_indx])/cv
    # T_mix[T_mix<T_in] = T_in
    P_mix = rho_mix * R_mix * T_mix

    # momentum conservation using isentropic
    gamma_current                    = state['gas_cached_props']['cp']/state['gas_cached_props']['cv']
    speed_sound_current              = state['gas_cached_props']['sound_speed'][inj_indx]
    mach_current                     = u[inj_indx]/speed_sound_current
    P_tot_current                    = P[inj_indx] * (1+(gamma_current-1)/2*mach_current**2)**((gamma_current-1)/gamma_current)
    P_tot_mix                        = P_tot_current

    gamma_mix                        = gamma_current
    speed_sound_mix                  = np.sqrt(gamma_mix * R_mix * T_mix)
    pressure_ratio                   = P_tot_mix/P_mix
    pressure_ratio[pressure_ratio<1] = 1
    mach_mix                         = np.sqrt(2/(gamma_mix-1)*((pressure_ratio)**((gamma_mix-1)/gamma_mix)-1))
    u_mix                            = mach_mix * speed_sound_mix

    # replace the influenced cells states
    mass         = rho_mix * vol
    momx         = rho_mix * u_mix * vol
    energy       = rho_mix * ((T_mix*cv+h_f_mix[inj_indx]) + 0.5 *(u_mix**2))*vol
    mass_species = rho_mix * Y_mix[0,inj_indx] * vol

    Q_cons_inject= np.vstack((mass,momx,energy,mass_species))

    Q_cons_user[:,inj_indx] = Q_cons_inject

    smoothing_start_indx = (detonation[0]-10)%(solver_param['cell_number']+4) 
    smoothing_end_indx   = (detonation[0]+10)%(solver_param['cell_number']+4) 

    q_left  = Q_cons_user[:,smoothing_start_indx] 
    q_right = Q_cons_user[:,smoothing_end_indx]

    x_knwon = np.array([0,20]) 
    y_known = np.array([q_left,q_right]).T 
    f_interp = interp1d(x_knwon,y_known,kind='linear') 
    x = np.arange(0,20) 
    q_ramp = f_interp(x) 

    if smoothing_start_indx < smoothing_end_indx: 
        Q_cons_user[:, smoothing_start_indx:smoothing_end_indx] = q_ramp 
        
    else: 
        wrap_len = (solver_param['cell_number']+4) - smoothing_start_indx 
        Q_cons_user[:, smoothing_start_indx:] = q_ramp[:, :wrap_len] 
        Q_cons_user[:, :smoothing_end_indx]   = q_ramp[:, wrap_len:]

    # --- Reapply periodic BCs to maintain consistency ---
    Q_cons_user[:, 0:2] = Q_cons_user[:, -4:-2]   # left ghosts
    Q_cons_user[:, -2:] = Q_cons_user[:, 2:4]  # right ghosts

    ncell_total = solver_param['cell_number'] + 4  # Total cells including ghosts
    ncell_interior = solver_param['cell_number']   # Interior cells only
    
    # Define smoothing region (10 cells on each side of detonation)
    smoothing_width = 10
    smoothing_start = (detonation[0] - smoothing_width) % ncell_interior
    smoothing_end = (detonation[0] + smoothing_width) % ncell_interior
    
    # Get the actual smoothing length (accounts for periodic wrapping)
    if smoothing_start <= smoothing_end:
        smoothing_length = smoothing_end - smoothing_start
    else:
        smoothing_length = (ncell_interior - smoothing_start) + smoothing_end
    
    # Adjust for ghost cells - work only on interior cells
    # Ghost cells are typically at indices: [0:2] and [-2:]
    interior_start = 2  # First interior cell index
    interior_end = interior_start + ncell_interior
    
    # Convert to absolute indices including ghost cells
    abs_start = interior_start + smoothing_start
    abs_end = interior_start + smoothing_end
    
    # Get the boundary values for interpolation
    # Make sure we're using interior cells, not ghost cells
    q_left_idx = interior_start + ((detonation[0] - smoothing_width) % ncell_interior)
    q_right_idx = interior_start + ((detonation[0] + smoothing_width) % ncell_interior)
    
    q_left = Q_cons_user[:, q_left_idx]
    q_right = Q_cons_user[:, q_right_idx]
    
    # Create interpolation
    x_known = np.array([0, smoothing_length])
    y_known = np.array([q_left, q_right]).T
    f_interp = interp1d(x_known, y_known, kind='linear', axis=1)
    
    # Generate smoothed values
    x_interp = np.arange(0, smoothing_length)
    q_ramp = f_interp(x_interp)
    
    # Apply smoothing with proper periodic handling
    if smoothing_start < smoothing_end:
        # No wrap-around case
        start_idx = interior_start + smoothing_start
        end_idx = interior_start + smoothing_end
        Q_cons_user[:, start_idx:end_idx] = q_ramp
    else:
        # Wrap-around case
        # First segment: from start to end of domain
        wrap_len1 = ncell_interior - smoothing_start
        start_idx1 = interior_start + smoothing_start
        Q_cons_user[:, start_idx1:start_idx1 + wrap_len1] = q_ramp[:, :wrap_len1]
        
        # Second segment: from start of domain to end point
        wrap_len2 = smoothing_end
        start_idx2 = interior_start
        Q_cons_user[:, start_idx2:start_idx2 + wrap_len2] = q_ramp[:, wrap_len1:wrap_len1 + wrap_len2]
    
    # Reapply periodic BCs (important after modification)
    Q_cons_user[:, 0:2] = Q_cons_user[:, interior_end-2:interior_end]    # left ghosts from right interior
    Q_cons_user[:, -2:] = Q_cons_user[:, interior_start:interior_start+2]  # right ghosts from left interior
    
    state['Q_cons'] = reshape_func.results_user2solver_converter(Q_cons_user)

    return state

def source_calculator(solver_param,rom_param,state):

    # load the basic information
    Q_cons           = state['Q_cons']
    Q_prim           = state['Q_prim']
    Q_cons_user      = reshape_func.results_solver2user_converter(solver_param['num_state_var'],solver_param['cell_number'],Q_cons)
    Q_prim_user      = reshape_func.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],Q_prim)

    rho              = Q_prim_user[0,:]
    vx               = Q_prim_user[1,:]
    press            = Q_prim_user[2,:]
    temp             = Q_prim_user[3,:]
    Y                = Q_prim_user[4:,:]

    Y_full             = reshape_func.find_mass_fraction_full(Y)

    vol              = solver_param['vol']
    num_state_var    = solver_param['num_state_var']

    h_ref     = np.array(state['gas_lookup_table']['href'])[:,np.newaxis]


    A = state['gas_lookup_table']['A']
    Ea= state['gas_lookup_table']['Ea']
    n = state['gas_lookup_table']['n']

    MW = state['gas_lookup_table']['MW']
    Ru = state['gas_lookup_table']['R_univ'] / 1000

    reactant_concent = (rho * Y) / MW

    k = A * (temp**n) * np.exp(-Ea / (Ru * temp))

    q = k * reactant_concent
    
    w_r = MW * -1 * q

    w_p = MW * +1 * q

    w   = np.vstack((w_r,w_p))

    ################################################################

    state['int_energy']    = state['gas_cached_props']['int_energy_mass']
    state['heat_release']  = - np.sum(w*h_ref,axis=0) 

    source_terms   = np.zeros((solver_param['num_state_var'] , solver_param['cell_number'] + 4))

    # source_terms[3:,:]   = (net_production_rate_R * MW * vol).T
    # source_terms[3:,:]   = reaction_source[0,:]/
    # source_terms[3:,:]   = reaction_source
    source_terms[3:,:]   = w_r * vol

    state['source_terms'] = source_terms.ravel()

    if solver_param['hyper'] == True:

        source_terms_int = reshape_func.solver_eliminate_ghost(solver_param['cell_number'],
                                                               solver_param['num_state_var'],
                                                               state['source_terms'])

        state['source_terms'] = source_terms_int[rom_param['S_indx_solver']]

    state['d_flux_dx'] = state['d_flux_dx'] + state['source_terms']

    return state

def residual_calculator(solver_param,rom_param,state):

    if solver_param['time_scheme'] != 'FDF':

        # update prim state
        state = cons2prim_converter(solver_param,state)

        # update the ghost cells
        state = bc_func.update_ghost_cell(solver_param,state)

        # update prim state
        state = prim2cons_converter(solver_param,state)

    # precompute reacting flow related variables
    state = cache_cantera(solver_param,state)
    
    if solver_param['flux_scheme'] == '2nd Order Roe':

        if solver_param['numpy_vector']:

            state = second_order_roe_inviscid_flux_calculator(solver_param,rom_param,state)

        else:

            state = second_order_roe_inviscid_flux_calculator_for(solver_param,rom_param,state)


        if solver_param['viscous_flag']:

            if solver_param['numpy_vector']:

                state = second_order_roe_viscous_flux_calculator(solver_param,rom_param,state)

            else: 

                state = second_order_roe_viscous_flux_calculator_for(solver_param,rom_param,state)

    # apply flux vector
    state   = d_flux_dx_calculator(solver_param,rom_param,state)

    # apply source term 
    state = source_calculator(solver_param,rom_param,state)

    return state
