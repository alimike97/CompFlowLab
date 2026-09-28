import os
import shutil
import argparse
import subprocess
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np

# Runs a batch of independent, full-order (solver_mode = FOM) simulations, one per ensemble member, each started from its own perturbed left/right initial condition

def run_member(inputs):

    case_dir, k, num_snapshots, save_interval, main_py = inputs

    input_file = os.path.join(case_dir, 'input_file.inp')

    cmd = [sys.executable, main_py, case_dir, input_file]

    print(f"Starting ensemble member {k}: {os.path.basename(case_dir)}\n")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        # gather this case's own saved snapshots into a single .npy per field, in its own
        # folder -- same as batch_wrapper_1D_RDE_single_step.py's gather step
        iter_list = np.arange(0, num_snapshots, save_interval)

        for suffix in ('cons', 'prim', 'res'):

            gathered = None

            for i, iter in enumerate(iter_list):

                snapshot = np.load(os.path.join(case_dir, 'FOM_results', f'{iter}iteration_{suffix}.npy'))

                if gathered is None:
                    gathered = np.zeros(snapshot.shape + (len(iter_list),))

                gathered[..., i] = snapshot

            np.save(os.path.join(case_dir, f'gather_{suffix}.npy'), gathered)

        return f"SUCCESS: member {k} ({case_dir})"

    except subprocess.CalledProcessError as e:
        return f"FAILED: member {k} ({case_dir})\nError: {e.stderr}"


if __name__ == '__main__':

    root_dir = Path(__file__).resolve().parent.parent
    main_py  = str(root_dir / 'main.py')
    sys.path.append(str(root_dir))

    from compflowlab.utils import input_read_func

    parser = argparse.ArgumentParser(description='Run an ensemble of independent FOM Sod shock tube simulations.')
    parser.add_argument('working_directory', type=str, help='Path to the working directory.')
    parser.add_argument('input_file', type=str, help='Path to the base input file (num_ensmbl / *_std / *_ic keys are read from it).')
    parser.add_argument('--num-processes', type=int, default=None, help='Number of parallel subprocesses (default: num_ensmbl).')

    args = parser.parse_args()

    if not os.path.isdir(args.working_directory):
        print(f"Error: The working directory '{args.working_directory}' does not exist.")
        exit(1)

    if not os.path.isfile(args.input_file):
        print(f"Error: The input file '{args.input_file}' does not exist.")
        exit(1)

    batch_dir = os.path.join(args.working_directory, 'ensemble_fom_batch')

    if os.path.exists(batch_dir):
        shutil.rmtree(batch_dir)

    input_param = input_read_func.read_input_file(args)

    num_ensmbl    = int(input_param['num_ensmbl'])
    num_snapshots = int(input_param['num_steps'])
    save_interval = int(input_param['save_interval'])

    rho_mean,   rho_std   = eval(input_param['rho_ic']),   eval(input_param['rho_std'])
    vel_mean,   vel_std   = eval(input_param['vel_ic']),   eval(input_param['vel_std'])
    press_mean, press_std = eval(input_param['press_ic']), eval(input_param['press_std'])
    temp_mean,  temp_std  = eval(input_param['temp_ic']),  eval(input_param['temp_std'])

    rng = np.random.default_rng(42)

    case_dirs = []

    for k in range(num_ensmbl):

        rho_L,   rho_R   = rng.normal(rho_mean,   rho_std)
        u_L,     u_R     = rng.normal(vel_mean,   vel_std)
        p_L,     p_R     = rng.normal(press_mean, press_std)
        T_L,     T_R     = rng.normal(temp_mean,  temp_std)

        case_dir = os.path.join(batch_dir, f'ensemble{k}')
        os.makedirs(case_dir)

        member_param = dict(input_param)
        member_param['solver_mode'] = 'FOM'
        member_param['rho_ic']      = str([float(rho_L), float(rho_R)])
        member_param['vel_ic']      = str([float(u_L),   float(u_R)])
        member_param['press_ic']    = str([float(p_L),   float(p_R)])
        member_param['temp_ic']     = str([float(T_L),   float(T_R)])

        with open(os.path.join(case_dir, 'input_file.inp'), 'w') as f:

            for key, value in member_param.items():
                f.write(f"{key} = {value}\n")

        case_dirs.append(case_dir)

    num_processes = args.num_processes or num_ensmbl

    args_list = [(case_dirs[k], k, num_snapshots, save_interval, main_py) for k in range(num_ensmbl)]

    with Pool(processes=num_processes) as pool:

        results = pool.map(run_member, args_list)

    for r in results:
        print(r)

    print(f"\nEach member's gathered snapshots (gather_cons.npy, gather_prim.npy, gather_res.npy) "
          f"were written into its own case folder under: {batch_dir}")
