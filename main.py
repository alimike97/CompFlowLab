
import customtkinter
import ast
from tkinter import ttk
import importlib
import pandas as pd


import solver_functions
import non_linear_terms
import time_integrator_functions
import visualization_functions
import ui_solver_bridge
import rom_functions


class UI(customtkinter.CTk):

    def __init__(self):

        super().__init__()

        customtkinter.set_appearance_mode("Dark")      # Modes: system (default), light, dark
        customtkinter.set_default_color_theme("dark-blue")  # Themes: blue (default), dark-blue, green

        self.window_width = 1280
        self.window_length= 720

        self.screen_width  = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        self.x = (self.screen_width // 2) - (self.window_width // 2)
        self.y = (self.screen_height // 2) - (self.window_length // 2)

        self.geometry('{}x{}+{}+{}'.format(self.window_width, self.window_length, self.x, self.y))
        self.title('ROMify')
       
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1,weight=1)

        ######################   Navigation Frame   ######################

        self.navigation_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.navigation_frame.grid(row=0, column=0, sticky="nsew")
        self.navigation_frame.grid_rowconfigure(4, weight=1)

        self.navigation_frame_label = customtkinter.CTkLabel(self.navigation_frame, text="ROMify", compound="center" , font=customtkinter.CTkFont(size=15, weight="bold"))
        self.navigation_frame_label.grid(row=0, column=0, padx=0, pady=20)

        self.home_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Home",fg_color="transparent",  text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),anchor="w",command=self.home_button_callback)
        self.home_button.grid(row=1, column=0, sticky="ew")

        self.set_up_page_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Setup", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w",command=self.setup_button_callback)
        self.set_up_page_button.grid(row=2, column=0, sticky="ew")

        self.results_page_button = customtkinter.CTkButton(self.navigation_frame, corner_radius=0, height=40, border_spacing=10, text="Results", fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"), anchor="w",command=self.results_button_callback)
        self.results_page_button.grid(row=3, column=0, sticky="ew")

        self.home_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.home_frame.grid_columnconfigure(1, weight=1)

        self.set_up_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.set_up_frame.grid_rowconfigure(15,weight=1)
        self.set_up_frame.grid_columnconfigure(10,weight=1)

        self.results_page_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.results_page_frame.grid_columnconfigure(1, weight=1)


        ######################   Solver Mode   ######################

        self.solver_mode_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.solver_mode_frame.grid(row=0,column=0,columnspan = 3 , rowspan = 6 , pady=10,padx=10,sticky='new')

        self.solver_mode_group_label = customtkinter.CTkLabel(self.solver_mode_frame,text="Solver Mode" )
        self.solver_mode_group_label.grid(row=0 , column= 0 , columnspan = 3 )

        self.fom_mode_checkbox_check_var = customtkinter.BooleanVar(value=1)
        self.fom_mode_checkbox = customtkinter.CTkCheckBox(self.solver_mode_frame , text='FOM', variable=self.fom_mode_checkbox_check_var , 
                                                           command=lambda: self.solver_mode_checkbox_callback('FOM'))
        self.fom_mode_checkbox.grid(row=1,column=0,padx=5, pady=10)

        self.rom_mode_checkbox_check_var = customtkinter.BooleanVar()
        self.rom_mode_checkbox = customtkinter.CTkCheckBox(self.solver_mode_frame , text='ROM', variable=self.rom_mode_checkbox_check_var , 
                                                           command=lambda: self.solver_mode_checkbox_callback(option='ROM'))
        self.rom_mode_checkbox.grid(row=1,column=1,padx=5, pady=10)

        self.adaptive_rom_mode_checkbox_check_var = customtkinter.BooleanVar()
        self.adaptive_rom_mode_checkbox = customtkinter.CTkCheckBox(self.solver_mode_frame , text='Adaptive ROM', variable=self.adaptive_rom_mode_checkbox_check_var , 
                                                                    command=lambda: self.solver_mode_checkbox_callback(option='Adaptive ROM'))
        self.adaptive_rom_mode_checkbox.grid(row=1,column=2,padx=5, pady=10)

        self.rom_method_label = customtkinter.CTkLabel(self.solver_mode_frame , text='ROM Method')
        self.rom_method_label.grid(row=2,column=0,padx=5, pady=10)
        
        self.rom_method_entry_var = customtkinter.StringVar()
        self.rom_method_options = ['Galerkin','LSPG']
        self.rom_method = customtkinter.CTkOptionMenu(self.solver_mode_frame , values=self.rom_method_options , state='disabled' , variable=self.rom_method_entry_var)
        self.rom_method.grid(row=2,column=1,padx=5, pady=10)

        self.adaptive_rom_method_entry_var = customtkinter.StringVar()
        self.adaptive_rom_method_options = ['Single-Snapshot','Multi-Snapshot']
        self.adaptive_rom_method = customtkinter.CTkOptionMenu(self.solver_mode_frame , values=self.adaptive_rom_method_options , state='disabled' , variable=self.adaptive_rom_method_entry_var)
        self.adaptive_rom_method.grid(row=2,column=2,padx=2, pady=10)

        self.energy_capture_label = customtkinter.CTkLabel(self.solver_mode_frame , text='POD Energy Capture')
        self.energy_capture_label.grid(row=4,column=0,padx=5, pady=10)
        
        self.energy_capture_entry_var = customtkinter.StringVar()
        self.energy_capture = customtkinter.CTkEntry(self.solver_mode_frame , textvariable=self.energy_capture_entry_var , state='disabled')
        self.energy_capture.grid(row=4,column=1,padx=5, pady=10)

        self.hyper_method_checkbox_check_var = customtkinter.BooleanVar()
        self.hyper_method_checkbox = customtkinter.CTkCheckBox(self.solver_mode_frame , text='Hyper-Reduction', state='disabled' ,variable=self.hyper_method_checkbox_check_var)
        self.hyper_method_checkbox.grid(row=5,column=0,padx=5, pady=10)

        self.hyper_method_entry_var = customtkinter.StringVar()
        self.hyper_method_options = ['DEIM','QDEIM','Gappy POD','Gappy POD + E']
        self.hyper_method = customtkinter.CTkOptionMenu(self.solver_mode_frame , values=self.hyper_method_options ,state='disabled',variable=self.hyper_method_entry_var)
        self.hyper_method.grid(row=5,column=1,padx=5, pady=10)

        self.training_window_label = customtkinter.CTkLabel(self.solver_mode_frame , text='Initial Training Window')
        self.training_window_label.grid(row=6,column=0,padx=5, pady=10)
        
        self.training_window_entry_var = customtkinter.StringVar()
        self.training_window = customtkinter.CTkEntry(self.solver_mode_frame , textvariable=self.training_window_entry_var , state='disabled')
        self.training_window.grid(row=6,column=1,padx=5, pady=10)

        self.unsampled_update_freq_entry_var = customtkinter.StringVar()
        self.unsampled_update_freq = customtkinter.CTkEntry(self.solver_mode_frame , textvariable=self.unsampled_update_freq_entry_var , state='disabled')
        self.unsampled_update_freq.grid(row=6,column=2,padx=2, pady=10)

        ######################   Time Descrtization   ######################

        self.time_descritization_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.time_descritization_frame.grid(row=0,column=3,padx=10,pady=10,columnspan = 2 , rowspan = 6 , sticky='new')

        self.time_descritization_group_label = customtkinter.CTkLabel(self.time_descritization_frame,text="Time Discretization" )
        self.time_descritization_group_label.grid(row=0 , column= 0 , columnspan = 2 )

        self.dt_label = customtkinter.CTkLabel(self.time_descritization_frame,text='Time Step')
        self.dt_label.grid(row=1,column=0,padx=15,pady=10)

        self.dt_entry_var = customtkinter.StringVar()
        self.dt = customtkinter.CTkEntry(self.time_descritization_frame , textvariable=self.dt_entry_var )
        self.dt.grid(row=1,column=1,padx=15,pady=10)

        self.num_step_label = customtkinter.CTkLabel(self.time_descritization_frame,text='Number of Steps')
        self.num_step_label.grid(row=2,column=0,padx=15,pady=10)

        self.num_step_entry_var = customtkinter.StringVar()
        self.num_step = customtkinter.CTkEntry(self.time_descritization_frame,textvariable=self.num_step_entry_var)
        self.num_step.grid(row=2,column=1,padx=15,pady=10)

        self.time_scheme_label = customtkinter.CTkLabel(self.time_descritization_frame,text='Time Scheme')
        self.time_scheme_label.grid(row=3,column=0,padx=15,pady=10)

        self.time_scheme_entry_var = customtkinter.StringVar()
        self.time_scheme_options = ['Explicit - FD Euler' , 'Implicit - BD Euler']
        self.time_scheme = customtkinter.CTkOptionMenu(self.time_descritization_frame , values=self.time_scheme_options , 
                                                       variable=self.time_scheme_entry_var , command = self.time_scheme_callback)
        self.time_scheme.grid(row=3,column=1,padx=15,pady=10)

        self.dual_time_label = customtkinter.CTkLabel(self.time_descritization_frame,text='Dual Time')
        self.dual_time_label.grid(row=4,column=0,padx=15,pady=10)

        self.dual_time_entry_var = customtkinter.StringVar()
        self.dual_time_options = ['Yes' , 'No']
        self.dual_time = customtkinter.CTkOptionMenu(self.time_descritization_frame , values=self.dual_time_options ,variable=self.dual_time_entry_var)
        self.dual_time.grid(row=4,column=1,padx=15,pady=10)

        self.res_tol_label = customtkinter.CTkLabel(self.time_descritization_frame,text='Residual Tol.')
        self.res_tol_label.grid(row=5,column=0,padx=15,pady=10)

        self.res_tol_entry_var = customtkinter.StringVar()
        self.res_tol = customtkinter.CTkEntry(self.time_descritization_frame,textvariable=self.res_tol_entry_var)
        self.res_tol.grid(row=5,column=1,padx=15,pady=10)

        #####################   Space Descrtization   ######################

        self.space_descritization_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.space_descritization_frame.grid(row=6,column=3,padx=10,pady=10, columnspan=2 , rowspan=4 , sticky = 'new')

        self.space_descritization_group_label = customtkinter.CTkLabel(self.space_descritization_frame,text="Space Discretization")
        self.space_descritization_group_label.grid(row=0 , column= 0 , columnspan = 2) 

        
        self.mesh_x_init_label = customtkinter.CTkLabel(self.space_descritization_frame , text='x initial')
        self.mesh_x_init_label.grid(row=1,column=0,padx=25, pady=10)
        
        self.mesh_x_init_entry_var = customtkinter.StringVar()
        self.mesh_x_init = customtkinter.CTkEntry(self.space_descritization_frame , textvariable=self.mesh_x_init_entry_var)
        self.mesh_x_init.grid(row=1,column=1,padx=25, pady=10)

        self.mesh_x_final_label = customtkinter.CTkLabel(self.space_descritization_frame , text='x final')
        self.mesh_x_final_label.grid(row=2,column=0,padx=25, pady=10)

        self.mesh_x_final_entry_var = customtkinter.StringVar()
        self.mesh_x_final= customtkinter.CTkEntry(self.space_descritization_frame , textvariable=self.mesh_x_final_entry_var)
        self.mesh_x_final.grid(row=2,column=1,padx=25, pady=10)

        self.mesh_cell_number_label = customtkinter.CTkLabel(self.space_descritization_frame , text='cell number')
        self.mesh_cell_number_label.grid(row=3,column=0,padx=25, pady=10)

        self.cell_number_entry_var = customtkinter.StringVar()
        self.cell_number = customtkinter.CTkEntry(self.space_descritization_frame , textvariable=self.cell_number_entry_var)
        self.cell_number.grid(row=3,column=1,padx=25, pady=10)

        #####################   Inet BC   ######################

        # self.inlet_bc_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        # self.inlet_bc_frame.grid(row=1,column=1,padx=10,pady=10 , sticky = 'new')

        # self.inlet_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Inlet Boundary Condition')
        # self.inlet_label.grid(row=0,column=0,columnspan = 2)

        # self.inlet_press_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Pressure')
        # self.inlet_press_label.grid(row=1,column=0,padx=15,pady=10) 

        # self.inlet_press_entry_var = customtkinter.StringVar()
        # self.inlet_press = customtkinter.CTkEntry(self.inlet_bc_frame , textvariable=self.inlet_press_entry_var)
        # self.inlet_press.grid(row=1,column=1,padx=15,pady=10) 

        # self.inlet_temp_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Temperature')
        # self.inlet_temp_label.grid(row=2,column=0,padx=15,pady=10)  

        # self.inlet_temp_entry_var = customtkinter.StringVar()
        # self.inlet_temp = customtkinter.CTkEntry(self.inlet_bc_frame,textvariable=self.inlet_temp_entry_var)
        # self.inlet_temp.grid(row=2,column=1,padx=15,pady=10) 

        # self.inlet_vel_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Velocity')
        # self.inlet_vel_label.grid(row=3,column=0,padx=15,pady=10) 

        # self.inlet_vel_entry_var = customtkinter.StringVar()
        # self.inlet_vel = customtkinter.CTkEntry(self.inlet_bc_frame , textvariable=self.inlet_vel_entry_var)
        # self.inlet_vel.grid(row=3,column=1,padx=15,pady=10) 

        # self.inlet_rho_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Density')
        # self.inlet_rho_label.grid(row=4,column=0,padx=15,pady=10) 

        # self.inlet_rho_entry_var = customtkinter.StringVar()
        # self.inlet_rho = customtkinter.CTkEntry(self.inlet_bc_frame,textvariable=self.inlet_rho_entry_var)
        # self.inlet_rho.grid(row=4,column=1,padx=15,pady=10) 

        # self.inlet_mass_frac_label = customtkinter.CTkLabel(self.inlet_bc_frame,text='Mass Fraction')
        # self.inlet_mass_frac_label.grid(row=5,column=0,padx=15,pady=10) 

        # self.inlet_mass_frac_entry_var = customtkinter.StringVar()
        # self.inlet_mass_frac = customtkinter.CTkEntry(self.inlet_bc_frame,textvariable=self.inlet_mass_frac_entry_var)
        # self.inlet_mass_frac.grid(row=5,column=1,padx=15,pady=10) 

        # ######################   Outlet BC   ######################

        # self.outlet_bc_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        # self.outlet_bc_frame.grid(row=1,column=2,padx=10,pady=10 , sticky = 'new')

        # self.outlet_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Outlet Boundary Condition')
        # self.outlet_label.grid(row=0,column=0,columnspan = 2)

        # self.outlet_press_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Pressure')
        # self.outlet_press_label.grid(row=1,column=0,padx=15,pady=10) 

        # self.outlet_press_entry_var = customtkinter.StringVar()
        # self.outlet_press = customtkinter.CTkEntry(self.outlet_bc_frame,textvariable=self.outlet_press_entry_var)
        # self.outlet_press.grid(row=1,column=1,padx=15,pady=10) 

        # self.outlet_temp_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Temperature')
        # self.outlet_temp_label.grid(row=2,column=0,padx=15,pady=10)  

        # self.outlet_temp_entry_var = customtkinter.StringVar()
        # self.outlet_temp = customtkinter.CTkEntry(self.outlet_bc_frame,textvariable=self.outlet_temp_entry_var)
        # self.outlet_temp.grid(row=2,column=1,padx=15,pady=10) 

        # self.outlet_vel_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Velocity')
        # self.outlet_vel_label.grid(row=3,column=0,padx=15,pady=10) 

        # self.outlet_vel_entry_var = customtkinter.StringVar()
        # self.outlet_vel = customtkinter.CTkEntry(self.outlet_bc_frame,textvariable=self.outlet_vel_entry_var)
        # self.outlet_vel.grid(row=3,column=1,padx=15,pady=10) 

        # self.outlet_rho_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Density')
        # self.outlet_rho_label.grid(row=4,column=0,padx=15,pady=10) 

        # self.outlet_rho_entry_var = customtkinter.StringVar()
        # self.outlet_rho = customtkinter.CTkEntry(self.outlet_bc_frame,textvariable=self.outlet_rho_entry_var)
        # self.outlet_rho.grid(row=4,column=1,padx=15,pady=10) 

        # self.outlet_mass_frac_label = customtkinter.CTkLabel(self.outlet_bc_frame,text='Mass Fraction')
        # self.outlet_mass_frac_label.grid(row=5,column=0,padx=15,pady=10) 

        # self.outlet_mass_frac_entry_var = customtkinter.StringVar()
        # self.outlet_mass_frac = customtkinter.CTkEntry(self.outlet_bc_frame,textvariable=self.outlet_mass_frac_entry_var)
        # self.outlet_mass_frac.grid(row=5,column=1,padx=15,pady=10)

        #####################   Gas Model   ######################
        
        self.physics_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.physics_frame.grid(row=6,column=0 , columnspan = 3 , pady=10,padx=10,sticky='nsew')

        self.gas_model_frame_label = customtkinter.CTkLabel(self.physics_frame,text="Physics")
        self.gas_model_frame_label.grid(row=0 , column= 0 , columnspan = 4) 

        self.gas_model_label = customtkinter.CTkLabel(self.physics_frame , text='Gas Model')
        self.gas_model_label.grid(row=1 , column=0)

        self.gas_model_entry_var = customtkinter.StringVar()
        self.gas_model_options = ['Ideal Gas Model']
        self.gas_model = customtkinter.CTkOptionMenu(self.physics_frame , values=self.gas_model_options)
        self.gas_model.grid(row=1,column=1 , padx = 5 , pady = 10)

        self.flux_scheme_label = customtkinter.CTkLabel(self.physics_frame , text='Flux Scheme')
        self.flux_scheme_label.grid(row=2 , column=0)

        self.flux_scheme_entry_var = customtkinter.StringVar()
        self.flux_scheme_options = ['Roe','Rusanov']
        self.flux_scheme = customtkinter.CTkOptionMenu(self.physics_frame , values=self.flux_scheme_options , variable=self.flux_scheme_entry_var)
        self.flux_scheme.grid(row=2,column=1 , padx = 15 , pady = 10)


        self.limiter_checkbox_check_var = customtkinter.BooleanVar()
        self.limiter_checkbox = customtkinter.CTkCheckBox(self.physics_frame , text='Limiter',variable=self.limiter_checkbox_check_var)
        self.limiter_checkbox.grid(row=1,column=2,columnspan = 2 ,rowspan = 2,padx=5, pady=10)             

        # ######################   Input File  ######################

        self.input_file_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.input_file_frame.grid_columnconfigure(0,weight=1)
        self.input_file_frame.grid_columnconfigure(1,weight=6)
        self.input_file_frame.grid_columnconfigure(2,weight=1)
        self.input_file_frame.grid(row=10,column=0,columnspan=8,pady=10 , sticky = 'new')
        
        self.working_dir_label = customtkinter.CTkLabel(self.input_file_frame , text='Working Directory')
        self.working_dir_label.grid(row=0,column=0,pady=10)

        self.working_dir_entry_var = customtkinter.StringVar()
        self.working_dir = customtkinter.CTkEntry(self.input_file_frame , textvariable=self.working_dir_entry_var)
        self.working_dir.grid(row=0,column=1,pady=10,sticky='ew') 

        self.working_dir_button = customtkinter.CTkButton(self.input_file_frame , text='Open' , command=self.working_dir_callback)
        self.working_dir_button.grid(row=0,column=2)  
        
        self.input_file_label = customtkinter.CTkLabel(self.input_file_frame , text='Input Files')
        self.input_file_label.grid(row=1,column=0,pady=10)   

        self.input_file = customtkinter.CTkEntry(self.input_file_frame)
        self.input_file.grid(row=1,column=1,pady=10,sticky='ew')   

        self.input_file_button = customtkinter.CTkButton(self.input_file_frame , text='Open' , command=self.input_file_open_callback)
        self.input_file_button.grid(row=1,column=2)

        self.FOM_file_label = customtkinter.CTkLabel(self.input_file_frame , text='FOM Result')
        self.FOM_file_label.grid(row=2,column=0,pady=10)  

        self.FOM_file_entry_var = customtkinter.StringVar()
        self.FOM_file = customtkinter.CTkEntry(self.input_file_frame , textvariable=self.FOM_file_entry_var)
        self.FOM_file.grid(row=2,column=1,pady=10,sticky='ew')

        self.FOM_file_button = customtkinter.CTkButton(self.input_file_frame , text='Open' , command=self.FOM_file_open_callback)
        self.FOM_file_button.grid(row=2,column=2)


        ######################  Visualization  ######################

        self.visual_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.visual_frame.grid(row=0,column=5,padx=10,pady=10 , sticky = 'new')

        self.visual_label = customtkinter.CTkLabel(self.visual_frame , text='Visualization')
        self.visual_label.grid(row=0,column=0,columnspan=2)

        self.visual_options = ['None','Density','Pressure','Temprature','Velocity','Heat Release','Mass Fraction']

        self.visual_1_option_label = customtkinter.CTkLabel(self.visual_frame , text='Variable #1')
        self.visual_1_option_label.grid(row=1,column=0,padx=15,pady=10)

        self.visual_1_option_entry_var=customtkinter.StringVar(value=self.visual_options[0])
        self.visual_1_option = customtkinter.CTkOptionMenu(self.visual_frame , values=self.visual_options , variable=self.visual_1_option_entry_var)
        self.visual_1_option.grid(row=1,column=1,padx=15,pady=10)

        self.visual_2_option_label = customtkinter.CTkLabel(self.visual_frame , text='Variable #2')
        self.visual_2_option_label.grid(row=2,column=0,padx=15,pady=10)

        self.visual_2_option_entry_var=customtkinter.StringVar(value=self.visual_options[0])
        self.visual_2_option = customtkinter.CTkOptionMenu(self.visual_frame , values=self.visual_options , variable=self.visual_2_option_entry_var)
        self.visual_2_option.grid(row=2,column=1,padx=15,pady=10)

        self.visual_3_option_label = customtkinter.CTkLabel(self.visual_frame , text='Variable #3')
        self.visual_3_option_label.grid(row=3,column=0,padx=15,pady=10)

        self.visual_3_option_entry_var=customtkinter.StringVar(value=self.visual_options[0])
        self.visual_3_option = customtkinter.CTkOptionMenu(self.visual_frame , values=self.visual_options , variable=self.visual_3_option_entry_var)
        self.visual_3_option.grid(row=3,column=1,padx=15,pady=10)

        self.visual_4_option_label = customtkinter.CTkLabel(self.visual_frame , text='Variable #4')
        self.visual_4_option_label.grid(row=4,column=0,padx=15,pady=10)

        self.visual_4_option_entry_var=customtkinter.StringVar(value=self.visual_options[0])
        self.visual_4_option = customtkinter.CTkOptionMenu(self.visual_frame , values=self.visual_options , variable=self.visual_4_option_entry_var)
        self.visual_4_option.grid(row=4,column=1,padx=15, pady=10)

        self.visual_update_interval_label = customtkinter.CTkLabel(self.visual_frame , text='Update Interval')
        self.visual_update_interval_label.grid(row=5,column=0,padx=15,pady=10)

        self.visual_update_interval_entry_var=customtkinter.StringVar()
        self.visual_update_interval = customtkinter.CTkEntry(self.visual_frame , textvariable=self.visual_update_interval_entry_var)
        self.visual_update_interval.grid(row=5,column=1,padx=15, pady=10)                  

        # ######################  IC  ######################

        # self.ic_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        # self.ic_frame.grid(row=1,column=3,padx=10,pady=10 , sticky = 'new')
        # self.ic_frame.columnconfigure(0,weight=1)

        # self.ic_label = customtkinter.CTkLabel(self.ic_frame , text='Initial Conditions')
        # self.ic_label.grid(row=0,column=0,pady=10)

        # self.ic_listbox = customtkinter.CTkButton(self.ic_frame,text='Configure IC',command=self.IC_conf)
        # self.ic_listbox.grid(row=1,column=0,padx=10,pady=10,sticky='ew')

        ######################  RUN  ######################

        self.run_button = customtkinter.CTkButton(self.set_up_frame , text='RUN !',font=customtkinter.CTkFont(size=30),command=self.run_callback)
        self.run_button.grid(row=13,column=3,columnspan = 2 ,pady=20,sticky='news')

        # self.home_button_callback()
        self.setup_button_callback()
        # self.input_file_open_callback()
        # self.run_callback()

    def select_frame_by_name(self, name):

        self.home_button.configure(fg_color=("gray75", "gray25") if name == "Home" else "transparent")
        self.set_up_page_button.configure(fg_color=("gray75", "gray25") if name == "Setup" else "transparent")
        self.results_page_button.configure(fg_color=("gray75", "gray25") if name == "Results" else "transparent")

        if name == "Home":
            self.home_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.home_frame.grid_forget()
        if name == "Setup":
            self.set_up_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.set_up_frame.grid_forget()
        if name == "Results":
            self.results_page_frame.grid(row=0, column=1, sticky="nsew")
        else:
            self.results_page_frame.grid_forget()

    def home_button_callback(self):

        self.select_frame_by_name("Home")
        self.description_textbox = customtkinter.CTkTextbox(master=self.home_frame , font=customtkinter.CTkFont(size=15, weight="bold"))
        self.description_textbox.grid(row=0, column=1 ,padx = 20 , pady =20 ,sticky="nsew")
        self.description_textbox.insert("0.0", "ROMify is a 1D compressible reacting flow solver with reduced-order-model (ROM) features. It aims to provide a simple and easy-to-use testbed to quickly prototype and examine new data-driven methods")

    def setup_button_callback(self):

        self.select_frame_by_name("Setup")
        
    def results_button_callback(self):

        self.select_frame_by_name("Results")

    def time_scheme_callback(self,time_scheme_entry_var):

        selected_option = time_scheme_entry_var

        if selected_option == 'Explicit - FD Euler':

            self.res_tol.configure(state='disabled')

        elif selected_option == 'Implicit - BD Euler':

            self.res_tol.configure(state='normal')

        
    def solver_mode_checkbox_callback(self,option):

        if option == 'FOM':

            self.fom_mode_checkbox_check_var.set(1)
            self.rom_mode_checkbox_check_var.set(0)
            self.adaptive_rom_mode_checkbox_check_var.set(0)

            self.rom_method.configure(state = 'disabled')
            self.energy_capture.configure(state = 'disabled')
            self.hyper_method_checkbox.configure(state = 'disabled')
            self.hyper_method_checkbox_check_var.set(0)
            self.hyper_method.configure(state = 'disabled')
            self.FOM_file.configure(state = 'disabled')
            self.FOM_file_button.configure(state = 'disabled')
            self.training_window.delete(0, customtkinter.END)
            self.training_window.configure(state = 'disabled')

            

        if option == 'ROM':

            self.rom_mode_checkbox_check_var.set(1)
            self.fom_mode_checkbox_check_var.set(0)
            self.adaptive_rom_mode_checkbox_check_var.set(0)

            self.rom_method.configure(state = 'normal')
            self.energy_capture.configure(state = 'normal')
            self.hyper_method_checkbox.configure(state = 'normal')
            self.hyper_method.configure(state = 'normal')
            self.FOM_file.configure(state = 'normal')
            self.FOM_file_button.configure(state = 'normal')
            self.training_window.delete(0, customtkinter.END)
            self.training_window.configure(state = 'disabled')
            

        if option == 'Adaptive ROM':

            self.adaptive_rom_mode_checkbox_check_var.set(1)
            self.fom_mode_checkbox_check_var.set(0)
            self.rom_mode_checkbox_check_var.set(0)

            self.rom_method.configure(state = 'normal')
            self.adaptive_rom_method.configure(state = 'normal')
            self.energy_capture.configure(state = 'disabled')
            self.hyper_method_checkbox.configure(state = 'disabled')
            self.hyper_method_checkbox_check_var.set(1)
            self.hyper_method.configure(state = 'normal')
            self.FOM_file.configure(state = 'disabled')
            self.FOM_file_button.configure(state = 'disabled')
            self.training_window.configure(state = 'normal')
            self.unsampled_update_freq.configure(state = 'normal')              

    
    def working_dir_callback(self):

        self.working_dir_path = customtkinter.filedialog.askdirectory()
        self.working_dir.delete(0, customtkinter.END)
        self.working_dir.insert(0,string=self.working_dir_path)

    def input_file_open_callback(self):

        self.input_file_path = customtkinter.filedialog.askopenfilename()
        # self.input_file_path = 'C:/GIT_Fork/ROMify/examples/shock_tube/input_file.inp'
        self.input_file.delete(0, customtkinter.END)
        self.input_file.insert(0,string=self.input_file_path)

        try:
            with open(self.input_file_path, 'r') as input_file:
                content = input_file.readlines()

            self.parameters = {}

            for line in content:
                line = line.strip()

                if not line or line.startswith('#'):
                    continue
                keyword_value = line.split('=',1)
                if len(keyword_value)==2:
                    keyword = keyword_value[0].strip()
                    value=keyword_value[1].strip()
                    self.parameters[keyword] = value

            self.fill_gui_from_inp_file()
            self.fill_gui_from_inp_file_IC()
            
        except FileNotFoundError:

            print(f"Error: File not found.")
            return None
    
        except IOError:

            print(f"Error: Unable to read the file.")
            return None
        
        
    def FOM_file_open_callback(self):

        self.FOM_file_path = customtkinter.filedialog.askopenfilename()
        self.FOM_file.delete(0, customtkinter.END)
        self.FOM_file.insert(0,string=self.FOM_file_path)


    def fill_gui_from_inp_file(self):

        parameters = self.parameters
        visual_options = self.visual_options

        for variable in parameters:

            if variable == 'dt':

                self.dt_entry_var.set(parameters['dt']) 

            if variable == 'num_steps':

                self.num_step_entry_var.set(parameters['num_steps'])

            if variable == 'time_scheme':

                if parameters['time_scheme'] == 'explicit_fd_euler':

                    self.time_scheme_entry_var.set(self.time_scheme_options[0])
                
                elif parameters['time_scheme'] == 'implicit_bd_euler':

                    self.time_scheme_entry_var.set(self.time_scheme_options[1])

            if variable == 'dual_time':

                if parameters['dual_time'] == 'true':

                    self.dual_time_entry_var.set(self.dual_time_options[0])
                
                elif parameters['dual_time'] == 'false':

                    self.dual_time_entry_var.set(self.dual_time_options[1])

            if variable == 'res_tol':

                self.res_tol_entry_var.set(parameters['res_tol'])

            if variable == 'x_initial':

                self.mesh_x_init_entry_var.set(parameters['x_initial'])

            if variable == 'x_final':

                self.mesh_x_final_entry_var.set(parameters['x_final'])
            
            if variable == 'cell_number':

                self.cell_number_entry_var.set(parameters['cell_number'])

            if variable == 'press_inlet':

                self.inlet_press_entry_var.set(parameters['press_inlet'])
            
            if variable == 'temp_inlet':

                self.inlet_temp_entry_var.set(parameters['temp_inlet'])

            if variable == 'vel_inlet':

                self.inlet_vel_entry_var.set(parameters['vel_inlet'])

            if variable == 'rho_inlet':

                self.inlet_rho_entry_var.set(parameters['rho_inlet'])                       

            if variable == 'mass_frac_inlet':

                self.inlet_mass_frac_entry_var.set(parameters['mass_frac_inlet'])    

            if variable == 'press_outlet':

                self.outlet_press_entry_var.set(parameters['press_outlet'])
            
            if variable == 'temp_outlet':

                self.outlet_temp_entry_var.set(parameters['temp_outlet'])

            if variable == 'vel_outlet':

                self.outlet_vel_entry_var.set(parameters['vel_outlet'])

            if variable == 'rho_outlet':

                self.outlet_rho_entry_var.set(parameters['rho_outlet'])                  

            if variable == 'mass_frac_outlet':

                self.outlet_mass_frac_entry_var.set(parameters['mass_frac_outlet'])

            if variable == 'gas_model':

                if parameters['gas_model'] == 'ideal gas model':

                    self.gas_model_entry_var.set(self.gas_model_options[0])

            if variable == 'flux_scheme':

                if parameters['flux_scheme'] == 'roe':

                    self.flux_scheme_entry_var.set(self.flux_scheme_options[0])
                    
                elif parameters['flux_scheme'] == 'rusanov':

                    self.flux_scheme_entry_var.set(self.flux_scheme_options[1])

            if variable == 'solver_mode':

                if parameters['solver_mode'] == 'fom':

                    self.solver_mode_checkbox_callback('FOM')

                elif parameters['solver_mode'] == 'rom':

                    self.solver_mode_checkbox_callback('ROM')

                elif parameters['solver_mode'] == 'adaptive_rom':

                    self.solver_mode_checkbox_callback('Adaptive ROM')

            if variable == 'rom_method':

                if parameters['rom_method'] == 'galerkin':

                    self.rom_method_entry_var.set(self.rom_method_options[0])
                
                elif parameters['rom_method'] == 'lspg':

                    self.rom_method_entry_var.set(self.rom_method_options[1])

            if variable == 'pod_energy':

                self.energy_capture_entry_var.set(parameters['pod_energy'])

            if variable == 'hyper':

                if parameters['hyper'] == 'false':

                    self.hyper_method_checkbox_check_var.set(False)

                elif parameters['hyper'] == 'true':
                    
                    self.hyper_method_checkbox_check_var.set(True)

            if variable == 'hyper_method':

                if parameters['hyper_method'] == 'deim':

                    self.hyper_method_entry_var.set(self.hyper_method_options[0])

                if parameters['hyper_method'] == 'qdeim':

                    self.hyper_method_entry_var.set(self.hyper_method_options[1])

                if parameters['hyper_method'] == 'gappypod':

                    self.hyper_method_entry_var.set(self.hyper_method_options[2])

                if parameters['hyper_method'] == 'gappypode':

                    self.hyper_method_entry_var.set(self.hyper_method_options[3])

            if variable == 'init_training_win':

                self.training_window_entry_var.set(parameters['init_training_win'])

            if variable == 'variable1':
                
                indices = [index for index, value in enumerate(visual_options) if value == parameters['variable1']]
                
                self.visual_1_option_entry_var.set(visual_options[indices[0]])

            if variable == 'variable2':
                
                indices = [index for index, value in enumerate(visual_options) if value == parameters['variable2']]
                
                self.visual_2_option_entry_var.set(visual_options[indices[0]])

            if variable == 'variable3':
                
                indices = [index for index, value in enumerate(visual_options) if value == parameters['variable3']]
                
                self.visual_3_option_entry_var.set(visual_options[indices[0]])

            if variable == 'variable4':
                
                indices = [index for index, value in enumerate(visual_options) if value == parameters['variable4']]
                
                self.visual_4_option_entry_var.set(visual_options[indices[0]])

            if variable == 'update_interval':
                
                indices = [index for index, value in enumerate(visual_options) if value == parameters['update_interval']]
                
                self.visual_update_interval_entry_var.set(parameters['update_interval'])

    def fill_gui_from_inp_file_IC(self):

        self.IC_conf()

        parameters =  self.parameters

        num_ic_region = len(ast.literal_eval(parameters['x_interval_ic']))

        for indx in range(0,num_ic_region):

            for variable in parameters:
                            
                if variable == 'x_interval_ic':
                    
                    x_interval = ast.literal_eval(parameters['x_interval_ic'])
                    self.IC_x_init_entry_var.set(x_interval[indx][0])
                    self.IC_x_final_entry_var.set(x_interval[indx][1])

                if variable == 'press_ic':

                    pressure = ast.literal_eval(parameters['press_ic'])
                    self.IC_press_entry_var.set(pressure[indx]) 

                if variable == 'temp_ic':
                    
                    temperature = ast.literal_eval(parameters['temp_ic'])
                    self.IC_temp_entry_var.set(temperature[indx])

                if variable == 'vel_ic':

                    velocity = ast.literal_eval(parameters['vel_ic'])
                    self.IC_vel_entry_var.set(velocity[indx]) 

                if variable == 'rho_ic':
                    
                    velocity = ast.literal_eval(parameters['rho_ic'])
                    self.IC_density_entry_var.set(velocity[indx])

                if variable == 'mass_frac_ic':

                    mass_fraction = ast.literal_eval(parameters['mass_frac_ic'])
                    self.IC_mf_entry_var.set([mass_fraction[indx]])

            self.IC_row_insert()

        self.pass_ic_data()

        

    def IC_conf(self):

        self.new_windows = customtkinter.CTkToplevel()
        self.new_windows.protocol("WM_DELETE_WINDOW",self.pass_ic_data)
        self.new_windows.attributes("-topmost", True)
        self.new_windows_width   = 1430
        self.new_windows_height  = 310
        self.new_windows_x       = (self.screen_width/2)-(self.new_windows_width /2)
        self.new_windows_y       = (self.screen_height/2)-(self.new_windows_height /2)
        self.new_windows.geometry(f'{self.new_windows_width}x{self.new_windows_height}+{int(self.new_windows_x)}+{int(self.new_windows_y)}')
        self.new_windows.title('Initial Condition Configuration')

        ###Treeview Customisation (theme colors are selected)
        bg_color = app._apply_appearance_mode(customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"])
        text_color = app._apply_appearance_mode(customtkinter.ThemeManager.theme["CTkLabel"]["text_color"])
        selected_color = app._apply_appearance_mode(customtkinter.ThemeManager.theme["CTkButton"]["fg_color"])

        treestyle = ttk.Style()
        treestyle.theme_use('default')
        treestyle.configure("Treeview", background=bg_color, foreground=text_color, fieldbackground=bg_color, borderwidth=0)
 
        self.IC_data_table_frame = customtkinter.CTkFrame(self.new_windows)
        self.IC_data_table_frame.columnconfigure(14,weight=1)
        self.IC_data_table_frame.rowconfigure(8,weight=1)
        self.IC_data_table_frame.grid(row=0,column=0,padx = 10 , pady =10 , sticky='ew')

        self.tree = ttk.Treeview(self.IC_data_table_frame, columns=("x_init", "x_final", "Pressure","Temperature","Velocity","Density","Mass Fraction"), show="headings" , selectmode ='browse')
        self.tree.heading("x_init"       , text="x init")
        self.tree.heading("x_final"      , text="x final")
        self.tree.heading("Pressure"     , text="Pressure")
        self.tree.heading("Temperature"  , text="Temperature")
        self.tree.heading("Velocity"     , text="Velocity")
        self.tree.heading("Density"      , text="Density")
        self.tree.heading("Mass Fraction", text="Mass Fraction")
        self.tree.grid(row=0,column=0,columnspan=14)
        
        self.IC_x_init_label = customtkinter.CTkLabel(self.IC_data_table_frame, text="x init:")
        self.IC_x_init_label.grid(row=1,column=0)
        self.IC_x_init_entry_var = customtkinter.StringVar()
        self.IC_x_init_entry = customtkinter.CTkEntry(self.IC_data_table_frame,textvariable=self.IC_x_init_entry_var)
        self.IC_x_init_entry.grid(row=1,column=1)

        self.IC_x_final_label = customtkinter.CTkLabel(self.IC_data_table_frame, text="x final:")
        self.IC_x_final_label.grid(row=1,column=2)
        self.IC_x_final_entry_var = customtkinter.StringVar()        
        self.IC_x_final_entry = customtkinter.CTkEntry(self.IC_data_table_frame,textvariable=self.IC_x_final_entry_var)
        self.IC_x_final_entry.grid(row=1,column=3)

        self.IC_press_label = customtkinter.CTkLabel(self.IC_data_table_frame, text="Pressure:")
        self.IC_press_label.grid(row=1,column=4)
        self.IC_press_entry_var = customtkinter.StringVar()        
        self.IC_press_entry = customtkinter.CTkEntry(self.IC_data_table_frame,textvariable=self.IC_press_entry_var)
        self.IC_press_entry.grid(row=1,column=5)

        self.IC_temp_label = customtkinter.CTkLabel(self.IC_data_table_frame, text="Temperature:")
        self.IC_temp_label.grid(row=1,column=6)
        self.IC_temp_entry_var = customtkinter.StringVar()        
        self.IC_temp_entry = customtkinter.CTkEntry(self.IC_data_table_frame,textvariable=self.IC_temp_entry_var)
        self.IC_temp_entry.grid(row=1,column=7)

        self.IC_vel_label = customtkinter.CTkLabel(self.IC_data_table_frame, text="Velocity:")
        self.IC_vel_label.grid(row=1,column=8)
        self.IC_vel_entry_var = customtkinter.StringVar()        
        self.IC_vel_entry = customtkinter.CTkEntry(self.IC_data_table_frame,textvariable=self.IC_vel_entry_var)
        self.IC_vel_entry.grid(row=1,column=9)

        self.IC_density_label = customtkinter.CTkLabel(self.IC_data_table_frame, text="Density:")
        self.IC_density_label.grid(row=1,column=10)
        self.IC_density_entry_var = customtkinter.StringVar()        
        self.IC_density_entry = customtkinter.CTkEntry(self.IC_data_table_frame,textvariable=self.IC_density_entry_var)
        self.IC_density_entry.grid(row=1,column=11)

        self.IC_mf_label = customtkinter.CTkLabel(self.IC_data_table_frame, text="Mass Fraction:")
        self.IC_mf_label.grid(row=1,column=12)
        self.IC_mf_entry_var = customtkinter.StringVar()        
        self.IC_mf_entry = customtkinter.CTkEntry(self.IC_data_table_frame,textvariable=self.IC_mf_entry_var)
        self.IC_mf_entry.grid(row=1,column=13)

        self.IC_add_button = customtkinter.CTkButton(self.IC_data_table_frame, text="Add",command=self.IC_row_insert)
        self.IC_add_button.grid(row=2,column=2,columnspan=2 ,pady=10,sticky='ew')

        self.IC_remove_button = customtkinter.CTkButton(self.IC_data_table_frame, text="Remove",command=self.IC_row_remove)
        self.IC_remove_button.grid(row=2,column=6,columnspan=2 ,pady=10,sticky='ew')

        self.IC_file_read = customtkinter.CTkButton(self.IC_data_table_frame, text="Read IC File",command=self.IC_file_read_fun)
        self.IC_file_read.grid(row=2,column=10,columnspan=2,pady=10,sticky='ew')

        if hasattr(self, 'ic_data'):

            for row in self.ic_data:
                self.tree.insert("", "end", values=row)


    def IC_row_insert(self):

        xi  = self.IC_x_init_entry.get()
        xf  = self.IC_x_final_entry.get()
        p   = self.IC_press_entry.get()
        T   = self.IC_temp_entry.get()
        v   = self.IC_vel_entry.get()
        rho = self.IC_density_entry.get()
        mf  = self.IC_mf_entry.get()

        if xi and xf and p and T and v and rho and mf :

            self.tree.insert('','end',values=(xi,xf,p,T,v,rho,mf))

            self.IC_x_init_entry.delete(0, customtkinter.END)
            self.IC_x_final_entry.delete(0, customtkinter.END)
            self.IC_press_entry.delete(0, customtkinter.END)
            self.IC_temp_entry.delete(0, customtkinter.END)
            self.IC_vel_entry.delete(0, customtkinter.END)
            self.IC_density_entry.delete(0, customtkinter.END)
            self.IC_mf_entry.delete(0, customtkinter.END)

    def IC_row_remove(self):

        selected_item = self.tree.selection()
        if selected_item:
            self.tree.delete(selected_item)

    def IC_file_read_fun(self):

        self.new_windows.withdraw()
        IC_file_path = customtkinter.filedialog.askopenfilename()
        self.new_windows.deiconify()
        IC_data = pd.read_excel(IC_file_path)
        print('Reading Excel File Sucessfully Completed')
        
        num_ic_region = len(IC_data['press_ic'])

        parameters = ("x_interval_ic", "press_ic","temp_ic","vel_ic","rho_ic","mass_frac_ic")

        for indx in range(0,num_ic_region):

            for variable in parameters:
                            
                if variable == 'x_interval_ic':
                    
                    x_interval = ast.literal_eval(IC_data[variable][indx])
                    self.IC_x_init_entry_var.set(x_interval[0])
                    self.IC_x_final_entry_var.set(x_interval[1])

                if variable == 'press_ic':

                    pressure = IC_data[variable][indx]
                    self.IC_press_entry_var.set(pressure) 

                if variable == 'temp_ic':
                    
                    temperature = IC_data[variable][indx]
                    self.IC_temp_entry_var.set(temperature)

                if variable == 'vel_ic':

                    velocity = IC_data[variable][indx]
                    self.IC_vel_entry_var.set(velocity) 

                if variable == 'rho_ic':
                    
                    density = IC_data[variable][indx]
                    self.IC_density_entry_var.set(density)

                if variable == 'mass_frac_ic':

                    mass_fraction = ast.literal_eval(IC_data[variable][indx])
                    self.IC_mf_entry_var.set([mass_fraction])

            self.IC_row_insert()        


    def pass_ic_data(self):

        data_array = []

        for item in self.tree.get_children():
            data = [self.tree.item(item, "values")]
            data_array.extend(data)

        self.ic_data = data_array
        self.new_windows.destroy()

    def run_callback(self):

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

        ui_solver_bridge.driver(self)


                                                             
if __name__ == "__main__":

    app = UI()  
    app.mainloop()






