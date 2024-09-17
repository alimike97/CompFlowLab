import numpy as np
import solver_functions

def visual_var_collector(solver_param):

    visual_options            = ['Density','Pressure','Temprature','Velocity','Heat Release']
    visual_options_indx_stack = [   0     ,     2    ,      3     ,     1    ,      1000    ]
    
    for item in visual_options:

        if solver_param['variable1'] == item:

            indx = visual_options.index(item)
            visual_var1 = visual_options_indx_stack[indx]

        elif solver_param['variable2'] == item:

            indx = visual_options.index(item)
            visual_var2 = visual_options_indx_stack[indx]

        elif solver_param['variable3'] == item:

            indx = visual_options.index(item)
            visual_var3 = visual_options_indx_stack[indx]

        elif solver_param['variable4'] == item:

            indx = visual_options.index(item)
            visual_var4 = visual_options_indx_stack[indx]

    if solver_param['variable1'] == 'None':
        
        visual_var1 = []

    if solver_param['variable2'] == 'None':
        
        visual_var2 = []

    if solver_param['variable3'] == 'None':
        
        visual_var3 = []       

    if solver_param['variable4'] == 'None':
        
        visual_var4 = []

    visual_param = {}

    visual_param['visual_var1'] = visual_var1
    visual_param['visual_var2'] = visual_var2
    visual_param['visual_var3'] = visual_var3
    visual_param['visual_var4'] = visual_var4

    return visual_param

def initial_plot(axs,solver_param,visual_param):

    # Plot in the first subplot (top-left)
    plot11 , = axs[0, 0].plot([], [] , color = 'C0' ,linestyle='-'  , label = 'ROM')
    plot12 , = axs[0, 0].plot([], [] , color = 'C0' ,linestyle='--' , label = 'FOM')

    # axs[0, 0].set_title('var1')

    # Plot in the second subplot (top-right)
    plot21 , = axs[0, 1].plot([], [], color='C1', linestyle='-'  ,  label = 'ROM')
    plot22 , = axs[0, 1].plot([], [], color='C1', linestyle='--' ,  label = 'FOM')
    
    # axs[0, 1].set_title('var2')

    # Plot in the third subplot (bottom-left)
    plot31 , = axs[1, 0].plot([], [], color='C2', linestyle='-' , label = 'ROM')
    plot32 , = axs[1, 0].plot([], [], color='C2', linestyle='--', label = 'FOM')
   
    # axs[1, 0].set_title('var3')

    # Plot in the fourth subplot (bottom-right)
    plot41 , = axs[1, 1].plot([], [], color='C3', linestyle='-' , label = 'ROM')
    plot42 , = axs[1, 1].plot([], [], color='C3', linestyle='--', label = 'FOM')

    # axs[1, 1].set_title('var4')

    visual_param['plot11'] = plot11 
    visual_param['plot21'] = plot21 
    visual_param['plot31'] = plot31 
    visual_param['plot41'] = plot41

    visual_param['plot12'] = plot12 
    visual_param['plot22'] = plot22 
    visual_param['plot32'] = plot32 
    visual_param['plot42'] = plot42 

    if solver_param['hyper'] == True or solver_param['solver_mode']=='Adaptive ROM':

        scatter1 = axs[0, 0].scatter([], [] , 7 ,color='black',marker='o',label='Sampling Points')
        scatter2 = axs[0, 1].scatter([], [] , 7 ,color='black',marker='o',label='Sampling Points') 
        scatter3 = axs[1, 0].scatter([], [] , 7 ,color='black',marker='o',label='Sampling Points')
        scatter4 = axs[1, 1].scatter([], [] , 7 ,color='black',marker='o',label='Sampling Points')

        visual_param['scatter1'] = scatter1
        visual_param['scatter2'] = scatter2
        visual_param['scatter3'] = scatter3
        visual_param['scatter4'] = scatter4

    
    return visual_param

def in_progress_plot(fig,axs,iter,solver_param,rom_param,state,visual_param):

    prim_results = solver_functions.results_solver2user_converter(solver_param['num_prim_var'],solver_param['cell_number'],state['Q_prim'])

    x = solver_param['x']

    visual_var1 = visual_param['visual_var1'] 
    visual_var2 = visual_param['visual_var2'] 
    visual_var3 = visual_param['visual_var3'] 
    visual_var4 = visual_param['visual_var4'] 

    y11= prim_results[visual_var1,2:-2]
    y21= prim_results[visual_var2,2:-2]
    y31= prim_results[visual_var3,2:-2]
    y41= prim_results[visual_var4,2:-2]

    # y41= state['heat_release'][2:-2]


    if y11.size == 0 : 

        y11 = 0 * prim_results[0,2:-2]

    if y21.size == 0 : 

        y21 = 0 * prim_results[0,2:-2]

    if y31.size == 0 : 

        y31 = 0 * prim_results[0,2:-2]

    if y41.size == 0 : 

        y41 = 0 * prim_results[0,2:-2]

    plot11 = visual_param['plot11']
    plot21 = visual_param['plot21']
    plot31 = visual_param['plot31']
    plot41 = visual_param['plot41']

    plot11.set_data(x,y11)
    plot21.set_data(x,y21)
    plot31.set_data(x,y31)
    plot41.set_data(x,y41)

    # axs[0,0].set_ylim(min(y11,default=0)*0.0099 , max(y11,default=0)*1.1)
    # axs[0,1].set_ylim(min(y21,default=0)*0.0099 , max(y21,default=0)*1.1)
    # axs[1,0].set_ylim(min(y31,default=0)*0.0099 , max(y31,default=0)*1.1)
    # axs[1,1].set_ylim(min(y41,default=0)*0.0099 , max(y41,default=0)*1.1)
    

    # breakpoint()

    
    # # axs[0,0].set_ylim(0 , 6)
    # axs[0,1].set_ylim(100e3 , 102e3)
    # # axs[1,0].set_ylim(0 , 8)
    # # axs[1,1].set_ylim(min(y4,default=0)*0.0099 , max(y4,default=0)*1.001)

    # axs[0,0].set_xlim( 0 , x[-2]*1.05)
    # axs[0,1].set_xlim( 0 , x[-2]*1.05)
    # axs[1,0].set_xlim( 0 , x[-2]*1.05)
    # axs[1,1].set_xlim( 0 , x[-2]*1.05)

    axs[0,0].set_ylabel(solver_param['variable1'])
    axs[0,1].set_ylabel(solver_param['variable2'])
    axs[1,0].set_ylabel(solver_param['variable3'])
    axs[1,1].set_ylabel(solver_param['variable4'])

    axs[0,0].relim()
    axs[0,0].autoscale()

    axs[0,1].relim()
    axs[0,1].autoscale()

    axs[1,0].relim()
    axs[1,0].autoscale()

    axs[1,1].relim()
    axs[1,1].autoscale()

    if solver_param['solver_mode'] == 'ROM' or (solver_param['plot_fom_flag'] == True and iter > int(solver_param['init_training_win'])):

        training_data_prim = rom_param['training_data_prim']

        y12= training_data_prim[visual_var1,:,iter]
        y22= training_data_prim[visual_var2,:,iter]
        y32= training_data_prim[visual_var3,:,iter]
        y42= training_data_prim[visual_var4,:,iter]

        if y12.size == 0 : 

            y12 = 0 * training_data_prim[0,:,0]

        if y22.size == 0 : 

            y22 = 0 * training_data_prim[0,:,0]

        if y32.size == 0 : 

            y32 = 0 * training_data_prim[0,:,0]

        if y42.size == 0 : 

            y42 = 0 * training_data_prim[0,:,0]

        plot12 = visual_param['plot12']
        plot22 = visual_param['plot22']
        plot32 = visual_param['plot32']
        plot42 = visual_param['plot42']

        plot12.set_data(x,y12)
        plot22.set_data(x,y22)
        plot32.set_data(x,y32)
        plot42.set_data(x,y42)

        axs[0,0].legend()
        axs[0,1].legend()
        axs[1,0].legend()
        axs[1,1].legend()

    if solver_param['hyper'] == True:

        S_indx_user = rom_param['S_indx_user']

        scatter1 = visual_param['scatter1']
        scatter2 = visual_param['scatter2']
        scatter3 = visual_param['scatter3']
        scatter4 = visual_param['scatter4']

        scatter1.set_offsets(np.column_stack((x[S_indx_user], y11[S_indx_user])))
        scatter2.set_offsets(np.column_stack((x[S_indx_user], y21[S_indx_user])))
        scatter3.set_offsets(np.column_stack((x[S_indx_user], y31[S_indx_user])))
        scatter4.set_offsets(np.column_stack((x[S_indx_user], y41[S_indx_user])))


    if iter % solver_param['vis_update_interval'] == 0:

        fig.canvas.draw()
        fig.canvas.flush_events()
        



    



