import solver_functions
import non_linear_terms
import time_integrator_functions
import visualization_functions
import rom_functions
import numpy as np
import matplotlib.pyplot as plt
import time

def driver(self):
    
    start_time = time.time()

    # collect all of variables from user interface
    solver_param = solver_functions.solver_parameters_collector(self)
    slope_limiter = False
    
    # some basic parameters
    dx      = (solver_param['x_final'] - solver_param['x_initial']) / solver_param['cell_number']
    vol     = dx
    dt      = solver_param['dt']
    res_tol = solver_param['res_tol']
    x       = np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number']) )
    sol_time= 0
    rom_flag= solver_param['calc_rom']
    hyper_flag = solver_param['hyper']
    interval= 100
    work_dir= solver_param['working_dir']
    flux_scheme='Rusanov'

    # get the initial condition
    rho , vx , p = solver_functions.ic_generator(solver_param)

    # add ghost cells
    rho , vx , p = solver_functions.add_ghost_cell(rho , vx , p ,sol_time)

    # get the gas properties
    gamma = 1.4

    # convert prim to cons
    mass , momx , energy = solver_functions.prim2cons_converter(rho , vx , p , gamma , vol)

    # stack cons and prim variables
    full_cons_results      = np.zeros(( 3 , int(solver_param['cell_number'])+4              )  )
    cons_results           = np.zeros(( 3 , int(solver_param['cell_number'])                )  )
    cons_results_save      = np.zeros(( 3 , int(solver_param['cell_number'])    , int(int(solver_param['num_step'])) )  )
    
    full_cons_results[0,:] = mass 
    full_cons_results[1,:] = momx 
    full_cons_results[2,:] = energy


    full_prim_results      = np.zeros(( 3 , int(solver_param['cell_number'])+4             )  )
    prim_results           = np.zeros(( 3 , int(solver_param['cell_number'])               )  )
    prim_results_save      = np.zeros(( 3 , int(solver_param['cell_number'])    , int(int(solver_param['num_step'])))  )
    full_prim_results[0,:] = rho 
    full_prim_results[1,:] = vx 
    full_prim_results[2,:] = p

    # initialize rom if necessary
    if rom_flag:

        basis , normalizor, denormalizor , q_ref = rom_functions.precomputer(solver_param)
        q_red  = basis.T @ np.hstack((full_cons_results[0,2:-2],full_cons_results[1,2:-2],full_cons_results[2,2:-2]))
        S_indx = np.array(range(0,512))

        if hyper_flag:
            pcc , S_indx       = rom_functions.hyper_precomputer(basis)

    else:

        S_indx                    = np.array(range(0,512))


    # create plot
    plt.close()
    fig , axs = plt.subplots(2,2)
    fig.set_size_inches(15,6)
    visual_var1,visual_var2,visual_var3,visual_var4 = visualization_functions.visual_var_collector(solver_param)
    plot1 , plot2 , plot3 , plot4                   = visualization_functions.initial_plot(axs)

    # begin simulation

    for iter in range(solver_param['num_step']):


        d_mass_dt , d_momx_dt , d_energy_dt = non_linear_terms.flux_calculator(mass,momx,energy,dx,gamma,vol,slope_limiter,sol_time,S_indx,hyper_flag,flux_scheme)

        if rom_flag:
             
            mass, momx, energy , q_red = rom_functions.order_reducer(solver_param,d_mass_dt,d_momx_dt,d_energy_dt,basis,normalizor,denormalizor,q_ref,q_red,pcc)

        else :

            # time integration
            if solver_param['time_scheme'] == 'Explicit - FD Euler':

                mass   = time_integrator_functions.explicit_fd_euler(mass   , d_mass_dt   , dt)
                momx   = time_integrator_functions.explicit_fd_euler(momx   , d_momx_dt   , dt)
                energy = time_integrator_functions.explicit_fd_euler(energy , d_energy_dt , dt)

            elif solver_param['time_scheme'] == 'Implicit - BD Euler':

                mass , momx , energy  = time_integrator_functions.implicit_bd_euler(mass,momx,energy,dx,dt,gamma,vol,res_tol,slope_limiter)

        mass[0:2]   = mass[3]
        momx[0:2]    = momx[3]
        energy[0:2]     = energy[3]

        mass[-2:] = mass[-3]
        momx[-2:]  = momx[-3]
        energy[-2:]   = energy[-3]

        # convert cons to prim
        rho , vx , p = solver_functions.cons2prim_converter(mass , momx , energy , gamma , vol,sol_time)

        # update results
        full_cons_results[0,:] = mass
        full_cons_results[1,:] = momx
        full_cons_results[2,:] = energy

        full_prim_results[0,:] = rho
        full_prim_results[1,:] = vx
        full_prim_results[2,:] = p

        # update results without ghost cells
        cons_results[0,:] = mass  [2:-2]
        cons_results[1,:] = momx  [2:-2]
        cons_results[2,:] = energy[2:-2]


        prim_results[0,:] = rho   [2:-2]
        prim_results[1,:] = vx    [2:-2]
        prim_results[2,:] = p     [2:-2]

        # visualization
        visualization_functions.in_progress_plot(fig,axs,x,prim_results,plot1,plot2,plot3,plot4,visual_var1,visual_var2,visual_var3,visual_var4,solver_param,iter)
        plt.show(block=False)
        
        print('Iteration: ' + str(iter))
        sol_time = sol_time + dt
        cons_results_save[:,:,iter] = cons_results
        prim_results_save[:,:,iter] = prim_results    

    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")

    if rom_flag:
        np.save( work_dir +"/ROM_cons.npy" ,cons_results_save)
        np.save( work_dir +"/ROM_prim.npy" ,prim_results_save)

    else: 
        np.save( work_dir +"/FOM_cons.npy" ,cons_results_save)
        np.save( work_dir +"/FOM_prim.npy" ,prim_results_save)


         

 
        






        







        


        
