import os
import argparse
import importlib
import cProfile
import pstats


import solver_functions
import non_linear_terms
import time_integrator_functions
import visualization_functions
import ui_solver_bridge
import rom_functions


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

# read the input file

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


# reload all of main modules to make sure all of updates are in actions
try:
    importlib.reload(ui_solver_bridge)
    print(f"Reloaded module: {ui_solver_bridge}")
except ImportError as e:
    print(f"Error reloading module: {ui_solver_bridge}")

try:
    importlib.reload(solver_functions)
    print(f"Reloaded module: {solver_functions}")
except ImportError as e:
    print(f"Error reloading module: {solver_functions}")

try:
    importlib.reload(non_linear_terms)
    print(f"Reloaded module: {non_linear_terms}")
except ImportError as e:
    print(f"Error reloading module: {non_linear_terms}")

try:
    importlib.reload(time_integrator_functions)
    print(f"Reloaded module: {time_integrator_functions}")
except ImportError as e:
    print(f"Error reloading module: {time_integrator_functions}")    

try:
    importlib.reload(visualization_functions)
    print(f"Reloaded module: {visualization_functions}")
except ImportError as e:
    print(f"Error reloading module: {visualization_functions}")    

try:
    importlib.reload(rom_functions)
    print(f"Reloaded module: {rom_functions}")
except ImportError as e:
    print(f"Error reloading module: {rom_functions}")


# collect all of variables from input file and prepare them for simulation
solver_param = solver_functions.solver_parameters_collector(args,input_param)

# start running the simulation

if solver_param['profiling'] == True:

    with cProfile.Profile() as pr:

        state = ui_solver_bridge.driver(args,solver_param)
    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename=args.working_directory +'/'+ 'profiling_results.prof')

else: 

    state = ui_solver_bridge.driver(args,solver_param)




