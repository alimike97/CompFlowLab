import numpy as np

def read_input_file(args):
    try:
        with open(args.input_file, 'r') as input_file:
            content = input_file.readlines()

        input_param = {}

        for line in content:
            line = line.strip()

            if not line or line.startswith('#'):
                continue
            keyword_value = line.split('=', 1)
            if len(keyword_value) == 2:
                keyword = keyword_value[0].strip()
                value = keyword_value[1].strip()
                input_param[keyword] = value

    except FileNotFoundError:
        print(f"Error: File not found.")

    except IOError:
        print(f"Error: Unable to read the file.")

    return input_param

def init_solver_param(args,input_param):

    solver_param = {}

    ### solver mode ###
    solver_param['solver_mode'] = str(input_param['solver_mode'])

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

        solver_param['ic_data']       = np.load(input_param['ic_path'])

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
    solver_param['flux_scheme']   =  input_param['flux_scheme']
    solver_param['limiter']       =  eval(input_param['limiter'])
    solver_param['limiter_method']=  input_param['limiter_method']
    solver_param['viscous_flag']  =  eval(input_param['viscous'])
    solver_param['injection']     =  eval(input_param['injection'])

    if solver_param['injection']:

        solver_param['injcetion_prim_state']       = np.load(input_param['injection_state_dir'])
        solver_param['injector_face_area']         = float(input_param['injector_face_area'])
        solver_param['non_inj_portion']            = float(input_param['non_injection_portion'])
        solver_param['non_inj_tail_portion']       = float(input_param['non_injection_tail_portion'])

    ### visualization ###
    solver_param['visual']              = eval(input_param['visual'])
    solver_param['save_visual']         = eval(input_param['save_visual'])
    solver_param['variable1']           = input_param['variable1']
    solver_param['variable2']           = input_param['variable2']
    solver_param['variable3']           = input_param['variable3']
    solver_param['variable4']           = input_param['variable4']
    solver_param['vis_update_interval'] = int(input_param['update_interval'])

    ### Input Directory ###
    solver_param['working_dir']         = args.working_directory

    ### Saving Data ###
    solver_param['save_interval']       = int(input_param['save_interval'])

    ### Profiling ###
    solver_param['profiling']           = eval(input_param['profiling'])

    ### Some Basic Calculations ###
    solver_param['dx']            = (solver_param['x_final'] - solver_param['x_initial']) / solver_param['cell_number']
    solver_param['vol']           = solver_param['dx']
    solver_param['x']             = np.linspace( solver_param['x_initial'] , solver_param['x_final'] , int(solver_param['cell_number']) )
    
    ### ROM Setup ###
    solver_param['rom_method']            = input_param['rom_method']
    solver_param['nl_rom_model']          = input_param['nl_rom_model']
    solver_param['adaptive_rom_method']   = input_param['arom_method']
    solver_param['pod_energy']            = float(input_param['pod_energy'])
    solver_param['hyper']                 = eval(input_param['hyper'])              
    solver_param['sampling_method']       = input_param['hyper_method']
    solver_param['sampling_rate']         = float(input_param['sampling_rate'])
    solver_param['unsampled_update_freq'] = int(input_param['unsampled_update_freq'])
    solver_param['init_training_win']     = float(input_param['init_training_win'])

    solver_param['training_data_dir']     = input_param['training_data_dir']
    solver_param['training_start_iter']   = int(input_param['training_start_iter'])
    solver_param['training_step_iter' ]   = int(input_param['training_step_iter' ])
    solver_param['training_end_iter'  ]   = int(input_param['training_end_iter'  ])

    solver_param['sarom_training_step'  ] = int(input_param['sarom_training_step'  ])
    solver_param['SAROM_SROM_solver'  ]   = input_param['sarom_srom_solver'  ]
    solver_param['sarom_cumsum_tol'  ]    = float(input_param['sarom_cumsum_tol'  ])
    solver_param['sarom_moving_avg_tol'  ]= float(input_param['sarom_moving_avg_tol'  ])

    solver_param['arom_restart']          = eval(input_param['arom_restart'])

    return solver_param