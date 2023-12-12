import solver_functions
import numpy as np

def flux_calculator(mass,momx,energy,dx,gamma,vol,slope_limiter,sol_time,S_indx,hyper_flag,flux_scheme):


    if hyper_flag == True:
    
        mass  = mass   [S_indx+2]
        momx  = momx   [S_indx+2]
        energy= energy [S_indx+2]
    
    # convert cons to prim
    rho , vx , p = solver_functions.cons2prim_converter(mass , momx , energy , gamma , vol,sol_time)

    # calculate gradients
    d_rho_dx = solver_functions.gradient_calculator(rho , dx)
    d_vx_dx  = solver_functions.gradient_calculator(vx , dx)
    d_p_dx   = solver_functions.gradient_calculator(p , dx)

    if slope_limiter:

        d_rho_dx = solver_functions.slope_limit(rho , dx , d_rho_dx)
        d_vx_dx  = solver_functions.slope_limit(vx  , dx , d_vx_dx )
        d_p_dx   = solver_functions.slope_limit(p   , dx , d_p_dx  )

    # extrapolate from center to face
    rho_face_left , rho_face_right = solver_functions.extrapolate_center2face(rho,d_rho_dx,dx)
    vx_face_left  , vx_face_right  = solver_functions.extrapolate_center2face(vx,d_vx_dx,dx)
    p_face_left   , p_face_right   = solver_functions.extrapolate_center2face(p,d_p_dx,dx)

    # compute flux
    if flux_scheme == 'Rusanov':
        flux_mass, flux_momx, flux_energy = solver_functions.rusanov_flux_calculator(rho_face_left , rho_face_right , vx_face_left , vx_face_right , p_face_left , p_face_right , gamma)
    elif flux_scheme == 'Roe':
        flux_mass, flux_momx, flux_energy = solver_functions.roe_flux_calculator(mass , momx , energy , gamma , vol)
   
    # inviscid flux vector terms
    dflux_mass_dx   = solver_functions.inviscid_d_flux_dx_calculator( flux_mass   , dx )
    dflux_momx_dx   = solver_functions.inviscid_d_flux_dx_calculator( flux_momx   , dx )
    dflux_energy_dx = solver_functions.inviscid_d_flux_dx_calculator( flux_energy , dx )

    return dflux_mass_dx , dflux_momx_dx , dflux_energy_dx
