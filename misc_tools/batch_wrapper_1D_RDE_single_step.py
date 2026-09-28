import os
import numpy as np
import argparse
import sys
from pathlib import Path
import cantera as ct
import shutil
import subprocess
from  multiprocessing import Pool

def run_solver(inputs):

    case_dir        = inputs[0]  
    num_snapshots   = inputs[1]
    save_interval   = inputs[2]
    dx              = inputs[3]
    dt              = inputs[4]

    input_file = os.path.join(case_dir, 'input_file.inp')
    
    cmd = ["python", "main.py", case_dir, input_file]
    
    print(f"Starting simulation: {os.path.basename(case_dir)}\n")
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        sample = np.load(os.path.join(case_dir,'FOM_results','0iteration_prim.npy'))

        num_vars, num_cell = np.shape(sample)

        iter_list = np.arange(0,num_snapshots,save_interval)

        prim_gather = np.zeros((num_vars,num_cell,len(iter_list)))
        V_cj        = np.zeros((len(iter_list)))

        for i , iter  in enumerate(iter_list):

            prim_gather[:,:,i] = np.load(os.path.join(case_dir,'FOM_results',rf'{iter}iteration_prim.npy'))

            if iter != 0:

                past_HR_indx    = np.argmax(prim_gather[-1,:,i-1])
                current_HR_indx = np.argmax(prim_gather[-1,:,i])

                V_cj[i] = ((current_HR_indx - past_HR_indx)%num_cell) * dx / 1e4 / dt  # 1e4 is for unit consistency


        np.save((os.path.join(case_dir,'gather_prim.npy')),prim_gather)
        np.save((os.path.join(case_dir,'V_cj.npy')),V_cj)

    
        return f"SUCCESS: {case_dir}"
    except subprocess.CalledProcessError as e:
        return f"FAILED: {case_dir}\nError: {e.stderr}"
    

if __name__ == '__main__':

    root_dir = Path(__file__).resolve().parent.parent
    sys.path.append(str(root_dir))

    from compflowlab.utils import input_read_func

    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process an input file and specify a working directory.')
    parser.add_argument('working_directory', type=str, help='Path to the working directory.')
    parser.add_argument('input_file', type=str, help='Path to the input file.')

    args = parser.parse_args()

    # Check if the working directory exists
    if not os.path.isdir(args.working_directory):
        print(f"Error: The working directory '{args.working_directory}' does not exist.")
        exit(1)

    # Check if the input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: The input file '{args.input_file}' does not exist.")
        exit(1)

    if os.path.exists(os.path.join(args.working_directory,'batch_results')):

        shutil.rmtree(os.path.join(args.working_directory,'batch_results'))

    input_param  = input_read_func.read_input_file(args)

    if not input_param['injection_state_dir'].endswith('.inp'):
        print(f"The injection input is not in batch mode!")

    inj_info     = input_read_func.read_chem_file(input_param['injection_state_dir'])

    # phi_list = np.linspace(inj_info['phi_l'],inj_info['phi_u'],inj_info['phi_n'])
    T_list   = np.linspace(inj_info['T_l'],inj_info['T_u'],inj_info['T_n'])

    # PHI, PRESS = np.meshgrid(phi_list,P_list)

    # PHI   = np.ravel(PHI)
    # PRESS = np.ravel(PRESS)

    # gas = ct.Solution('h2o2.yaml')

    for temp in T_list:

        dir = os.path.join(args.working_directory,
                                'batch_results',
                                rf'inj_state_T0_{temp:.3f}')

        std_state_ratio = 300/temp

        rho_inj = np.array(2.5647098290444186)/std_state_ratio
        u_inj   = np.array([200])
        P_inj   = np.array(405300)
        T_inj   = np.array(temp)
        MF_inj  = np.array(1)

        q_inj  = np.vstack((rho_inj,u_inj,P_inj,T_inj,MF_inj.T))

        if not os.path.exists(dir):

            os.makedirs(dir)


        inj_dir = os.path.join(dir,f'inj_state_T0_{temp:.3f}.npy')

        np.save(inj_dir,q_inj)

        input_param['injection_state_dir']=inj_dir 

        dir_inp = os.path.join(dir,'input_file.inp')

        with open(dir_inp, 'w') as f:

            for key, value in input_param.items():
                # This writes: Key = Value
                f.write(f"{key} = {value}\n")


    all_case_dirs = [
        os.path.join(args.working_directory, 'batch_results', d) 
        for d in os.listdir(os.path.join(args.working_directory, 'batch_results'))
    ]

    if inj_info['n_processor'] == 0:

        num_cores = inj_info['T_n']

    else:

        num_cores = inj_info['n_processor']

    num_snapshots = int(input_param['num_steps'])
    save_interval = int(input_param['save_interval'])
    dx            = (float(input_param['x_final']) - float(input_param['x_initial']))/float(input_param['cell_number'])
    dt            = float(input_param['dt'])

    args_list = [(all_case_dirs[i],num_snapshots,save_interval,dx,dt) for i in range(len(all_case_dirs))]

    with Pool(processes=num_cores) as pool:
        results = pool.map(run_solver, args_list)





