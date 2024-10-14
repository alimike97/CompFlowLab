
import customtkinter
import ast
from tkinter import ttk
import importlib
import numpy as np
import cProfile
import pstats


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

        self.window_width = 1350
        self.window_length= 800

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

        self.hybrid_rom_mode_checkbox_check_var = customtkinter.BooleanVar()
        self.hybrid_rom_mode_checkbox = customtkinter.CTkCheckBox(self.solver_mode_frame , text='Hybrid ROM', variable=self.hybrid_rom_mode_checkbox_check_var , 
                                                                  command=lambda: self.solver_mode_checkbox_callback(option='Hybrid ROM'))
        self.hybrid_rom_mode_checkbox.grid(row=1,column=3,padx=5, pady=10)

        self.rom_method_label = customtkinter.CTkLabel(self.solver_mode_frame , text='ROM Method')
        self.rom_method_label.grid(row=2,column=0,padx=5, pady=10)
        
        self.rom_method_entry_var = customtkinter.StringVar()
        self.rom_method_options = ['Galerkin','LSPG']
        self.rom_method = customtkinter.CTkOptionMenu(self.solver_mode_frame , values=self.rom_method_options , state='disabled' , variable=self.rom_method_entry_var)
        self.rom_method.grid(row=2,column=1,padx=5, pady=10)

        self.adaptive_rom_method_entry_var = customtkinter.StringVar()
        self.adaptive_rom_method_options = ['Single-Snapshot','Multi-Snapshot','Direct Adapt','Initiative Adapt']
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
        self.hyper_method_options = ['DEIM','QDEIM','Gappy POD','Gappy POD + Shock','Gappy POD + E']
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


        #####################   Gas Model   ######################
        
        self.physics_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.physics_frame.grid(row=6,column=0 , columnspan = 3 , pady=10,padx=10,sticky='nsew')

        self.gas_model_frame_label = customtkinter.CTkLabel(self.physics_frame,text="Physics")
        self.gas_model_frame_label.grid(row=0 , column= 0 , columnspan = 4) 

        self.gas_model_label = customtkinter.CTkLabel(self.physics_frame , text='Gas Model')
        self.gas_model_label.grid(row=1 , column=0)

        self.gas_model_entry_var = customtkinter.StringVar()
        self.gas_model_options = ['Non-Reacting Air','H2/Air Combustion']
        self.gas_model = customtkinter.CTkOptionMenu(self.physics_frame , values=self.gas_model_options , variable=self.gas_model_entry_var,command=self.gas_model_callback)
        self.gas_model.grid(row=1,column=1 , padx = 5 , pady = 10)

        self.flux_scheme_label = customtkinter.CTkLabel(self.physics_frame , text='Flux Scheme')
        self.flux_scheme_label.grid(row=2 , column=0)

        self.flux_scheme_entry_var = customtkinter.StringVar()
        self.flux_scheme_options = ['1st Order Roe','2nd Order Roe','Rusanov']
        self.flux_scheme = customtkinter.CTkOptionMenu(self.physics_frame , values=self.flux_scheme_options , variable=self.flux_scheme_entry_var)
        self.flux_scheme.grid(row=2,column=1 , padx = 15 , pady = 10)

        self.limiter_checkbox_check_var = customtkinter.BooleanVar()
        self.limiter_checkbox = customtkinter.CTkCheckBox(self.physics_frame , text='Limiter',variable=self.limiter_checkbox_check_var)
        self.limiter_checkbox.grid(row=1,column=2)

        self.viscous_checkbox_check_var = customtkinter.BooleanVar()
        self.viscous_checkbox = customtkinter.CTkCheckBox(self.physics_frame , text='Viscous',variable=self.viscous_checkbox_check_var)
        self.viscous_checkbox.grid(row=2,column=2)     

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
        self.visual_frame.grid(row=0,column=6,padx=10,pady=10 , sticky = 'new')

        self.visual_label = customtkinter.CTkLabel(self.visual_frame , text='Visualization')
        self.visual_label.grid(row=0,column=0,columnspan=2)

        self.visual_options = ['None','Density','Pressure','Temprature','Velocity','Heat Release']

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

        self.fom_plot_check_var = customtkinter.BooleanVar()
        self.fom_plot = customtkinter.CTkCheckBox(self.visual_frame , text='Plot FOM Data',variable=self.fom_plot_check_var)
        self.fom_plot.grid(row=6,column=0)

        self.profile_check_var = customtkinter.BooleanVar()
        self.profile = customtkinter.CTkCheckBox(self.visual_frame , text='Perform Profiling',variable=self.profile_check_var)
        self.profile.grid(row=6,column=1)             

        # ######################  IC  ######################

        self.ic_frame = customtkinter.CTkFrame(self.set_up_frame,corner_radius=20)
        self.ic_frame.grid(row=6,column=6,padx=10,pady=10 , sticky = 'new')
        self.ic_frame.columnconfigure(0,weight=1)

        self.ic_label = customtkinter.CTkLabel(self.ic_frame , text='Initial and Boundary Conditions')
        self.ic_label.grid(row=0,column=0,pady=10)

        self.ic_listbox = customtkinter.CTkButton(self.ic_frame,text='Configure IC',command=self.IC_conf)
        self.ic_listbox.grid(row=1,column=0,padx=10,pady=10,sticky='ew')

        self.bc_listbox = customtkinter.CTkButton(self.ic_frame,text='Configure BC',command=self.BC_conf)
        self.bc_listbox.grid(row=2,column=0,padx=10,pady=10,sticky='ew')
    

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
            self.hybrid_rom_mode_checkbox_check_var.set(0)

            self.rom_method.configure(state = 'disabled')
            self.adaptive_rom_method.configure(state = 'disabled')
            self.energy_capture.configure(state = 'disabled')
            self.hyper_method_checkbox.configure(state = 'disabled')
            self.hyper_method_checkbox_check_var.set(0)
            self.hyper_method.configure(state = 'disabled')
            self.FOM_file.configure(state = 'disabled')
            self.FOM_file_button.configure(state = 'disabled')
            self.training_window.delete(0, customtkinter.END)
            self.training_window.configure(state = 'disabled')
            self.unsampled_update_freq.configure(state = 'disabled')

            

        if option == 'ROM':

            self.rom_mode_checkbox_check_var.set(1)
            self.fom_mode_checkbox_check_var.set(0)
            self.adaptive_rom_mode_checkbox_check_var.set(0)
            self.hybrid_rom_mode_checkbox_check_var.set(0)

            self.rom_method.configure(state = 'normal')
            self.energy_capture.configure(state = 'normal')
            self.adaptive_rom_method.configure(state = 'disabled')
            self.hyper_method_checkbox.configure(state = 'normal')
            self.hyper_method.configure(state = 'normal')
            self.FOM_file.configure(state = 'normal')
            self.FOM_file_button.configure(state = 'normal')
            self.training_window.delete(0, customtkinter.END)
            self.training_window.insert(0,'0')
            self.training_window.configure(state = 'disabled')
            self.unsampled_update_freq.configure(state = 'disabled')
            

        if option == 'Adaptive ROM':

            self.adaptive_rom_mode_checkbox_check_var.set(1)
            self.fom_mode_checkbox_check_var.set(0)
            self.rom_mode_checkbox_check_var.set(0)
            self.hybrid_rom_mode_checkbox_check_var.set(0)

            self.rom_method.configure(state = 'normal')
            self.adaptive_rom_method.configure(state = 'normal')
            self.energy_capture.configure(state = 'disabled')
            self.hyper_method_checkbox.configure(state = 'disabled')
            self.hyper_method_checkbox_check_var.set(1)
            self.hyper_method.configure(state = 'normal')
            self.FOM_file.configure(state = 'normal')
            self.FOM_file_button.configure(state = 'normal')
            self.training_window.configure(state = 'normal')
            self.unsampled_update_freq.configure(state = 'normal')

        if option == 'Hybrid ROM':

            self.hybrid_rom_mode_checkbox_check_var.set(1)
            self.adaptive_rom_mode_checkbox_check_var.set(0)
            self.fom_mode_checkbox_check_var.set(0)
            self.rom_mode_checkbox_check_var.set(0)

            self.rom_method.configure(state = 'normal')
            self.adaptive_rom_method.configure(state = 'normal')
            self.energy_capture.configure(state = 'disabled')
            self.hyper_method_checkbox.configure(state = 'disabled')
            self.hyper_method_checkbox_check_var.set(1)
            self.hyper_method.configure(state = 'normal')
            self.FOM_file.configure(state = 'normal')
            self.FOM_file_button.configure(state = 'normal')
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
            self.fill_gui_from_inp_file_BC()
            
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

            if variable == 'gas_model':

                if parameters['gas_model'] == 'Non-Reacting Air':

                    self.gas_model_entry_var.set(self.gas_model_options[0])

                elif parameters['gas_model'] == 'H2/Air Combustion':

                    self.gas_model_entry_var.set(self.gas_model_options[1])

            if variable == 'x_initial':

                self.mesh_x_init_entry_var.set(parameters['x_initial'])

            if variable == 'x_final':

                self.mesh_x_final_entry_var.set(parameters['x_final'])
            
            if variable == 'cell_number':

                self.cell_number_entry_var.set(parameters['cell_number'])

            if variable == 'flux_scheme':

                if parameters['flux_scheme'] == 'roe1':

                    self.flux_scheme_entry_var.set(self.flux_scheme_options[0])

                elif parameters['flux_scheme'] == 'roe2':

                    self.flux_scheme_entry_var.set(self.flux_scheme_options[1])
                    
                elif parameters['flux_scheme'] == 'rusanov':

                    self.flux_scheme_entry_var.set(self.flux_scheme_options[2])

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

            if variable == 'viscosity':

                if parameters['viscosity'] == 'false':

                    self.viscous_checkbox_check_var.set(False)

                elif parameters['viscosity'] == 'true':
                    
                    self.viscous_checkbox_check_var.set(True)            

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
    
        parameters =  self.parameters

        # read npy file if exist

        if 'ic_path' in parameters:

            IC_file_path = parameters['ic_path']
            self.ic_data = np.load(IC_file_path)
            print('Reading IC File Sucessfully Completed')

        # read the regions manually entered 

        else:

            self.IC_conf()

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
                        self.IC_mf_entry_var.set(str(mass_fraction[indx]))

                self.IC_row_insert()

            self.pass_ic_data()
    
    def fill_gui_from_inp_file_BC(self):

        parameters = self.parameters

        self.BC_conf()

        for variable in parameters:

            if variable == 'inlet_bc':
                
                if parameters['inlet_bc'] == 'periodic':

                    self.inlet_period_checkbox_check_var.set(1)

                if parameters['inlet_bc'] == 'wall':

                    self.inlet_wall_checkbox_check_var.set(1)

            if variable == 'outlet_bc':

                if parameters['outlet_bc'] == 'periodic':
                    
                    self.outlet_period_checkbox_check_var.set(1)

                if parameters['outlet_bc'] == 'wall':

                    self.outlet_wall_checkbox_check_var.set(1)

            if variable == 'rho_inlet':

                if parameters['rho_inlet'] == 'extrap':

                    self.inlet_extrap_density_checkbox_check_var.set(1)

                else:

                    self.inlet_rho_entry_var.set(parameters['rho_inlet']) 

            if variable == 'vel_inlet':

                if parameters['vel_inlet'] == 'extrap':

                    self.inlet_extrap_velocity_checkbox_check_var.set(1)

                else:

                    self.inlet_vel_entry_var.set(parameters['vel_inlet'])                 

            if variable == 'press_inlet':

                if parameters['press_inlet'] == 'extrap':

                    self.inlet_extrap_press_checkbox_check_var.set(1)

                else:

                    self.inlet_press_entry_var.set(parameters['press_inlet'])                

            if variable == 'temp_inlet':

                if parameters['temp_inlet'] == 'extrap':

                    self.inlet_extrap_temp_checkbox_check_var.set(1)

                else:                

                    self.inlet_temp_entry_var.set(parameters['temp_inlet'])

            if variable == 'mass_frac_inlet':

                if parameters['mass_frac_inlet'] == 'extrap':

                    self.inlet_extrap_MF_checkbox_check_var.set(1)

                else:                

                    self.inlet_mass_frac_entry_var.set(parameters['mass_frac_inlet'])  
              
            if variable == 'rho_outlet':

                if parameters['rho_outlet'] == 'extrap':

                    self.outlet_extrap_density_checkbox_check_var.set(1)

                else:                

                    self.outlet_rho_entry_var.set(parameters['rho_outlet'])

            if variable == 'vel_outlet':

                if parameters['vel_outlet'] == 'extrap':

                    self.outlet_extrap_velocity_checkbox_check_var.set(1)

                else:                

                    self.outlet_vel_entry_var.set(parameters['vel_outlet'])            
                
            if variable == 'press_outlet':

                if parameters['press_outlet'] == 'extrap':

                    self.outlet_extrap_press_checkbox_check_var.set(1)

                else:                

                    self.outlet_press_entry_var.set(parameters['press_outlet'])
            
            if variable == 'temp_outlet':

                if parameters['temp_outlet'] == 'extrap':

                    self.outlet_extrap_temp_checkbox_check_var.set(1)

                else:

                  self.outlet_temp_entry_var.set(parameters['temp_outlet'])

            if variable == 'mass_frac_outlet':

                if parameters['mass_frac_outlet'] == 'extrap':

                    self.outlet_extrap_MF_checkbox_check_var.set(1)

                else:

                    self.outlet_mass_frac_entry_var.set(parameters['mass_frac_outlet'])

        self.pass_bc_data()

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

        self.tree = ttk.Treeview(self.IC_data_table_frame, columns=("x_init", "x_final", "Pressure","Temperature","Velocity","Density","Mass Fraction"), show="headings" , selectmode ='extended')
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
        self.IC_add_button.grid(row=2,column=1,columnspan=2 ,pady=10,sticky='ew')

        self.IC_remove_button = customtkinter.CTkButton(self.IC_data_table_frame, text="Remove Selected",command=self.IC_row_remove)
        self.IC_remove_button.grid(row=2,column=4,columnspan=2 ,pady=10,sticky='ew')

        self.IC_remove_all_button = customtkinter.CTkButton(self.IC_data_table_frame, text="Remove All",command=self.IC_row_remove_all)
        self.IC_remove_all_button.grid(row=2,column=9,columnspan=2 ,pady=10,sticky='ew')

        self.IC_file_read = customtkinter.CTkButton(self.IC_data_table_frame, text="Read IC File",command=self.IC_file_read_fun)
        self.IC_file_read.grid(row=2,column=12,columnspan=2,pady=10,sticky='ew')

        if hasattr(self, 'ic_data'):

            for row in self.ic_data:
                self.tree.insert("", "end", values=row)

    def BC_conf(self):

        self.bc_windows = customtkinter.CTkToplevel()
        self.bc_windows.protocol("WM_DELETE_WINDOW",self.pass_bc_data)
        self.bc_windows.attributes("-topmost", True)
        self.bc_windows_width   = 800
        self.bc_windows_height  = 330
        self.bc_windows_x       = (self.screen_width/2)-(self.bc_windows_width /2)
        self.bc_windows_y       = (self.screen_height/2)-(self.bc_windows_height /2)
        self.bc_windows.geometry(f'{self.bc_windows_width}x{self.bc_windows_height}+{int(self.bc_windows_x)}+{int(self.bc_windows_y)}')
        self.bc_windows.title('Boundary Condition Configuration')

        self.inlet_frame = customtkinter.CTkFrame(self.bc_windows,corner_radius=20)
        self.inlet_frame.grid(row=0,column=0,padx=10,pady=10)
        self.inlet_frame.columnconfigure(3,weight=1)

        self.inlet_label = customtkinter.CTkLabel(self.inlet_frame,text='Inlet Boundary Condition')
        self.inlet_label.grid(row=0,column=1,columnspan = 2)

        self.inlet_rho_label = customtkinter.CTkLabel(self.inlet_frame,text='Density')
        self.inlet_rho_label.grid(row=1,column=0,padx=15,pady=10)

        self.inlet_rho_entry_var = customtkinter.StringVar()
        self.inlet_rho = customtkinter.CTkEntry(self.inlet_frame,textvariable=self.inlet_rho_entry_var)
        self.inlet_rho.grid(row=1,column=1,padx=15,pady=10) 

        self.inlet_vel_label = customtkinter.CTkLabel(self.inlet_frame,text='Velocity')
        self.inlet_vel_label.grid(row=2,column=0,padx=15,pady=10) 

        self.inlet_vel_entry_var = customtkinter.StringVar()
        self.inlet_vel = customtkinter.CTkEntry(self.inlet_frame , textvariable=self.inlet_vel_entry_var)
        self.inlet_vel.grid(row=2,column=1,padx=15,pady=10) 

        self.inlet_press_label = customtkinter.CTkLabel(self.inlet_frame,text='Pressure')
        self.inlet_press_label.grid(row=3,column=0,padx=15,pady=10) 

        self.inlet_press_entry_var = customtkinter.StringVar()
        self.inlet_press = customtkinter.CTkEntry(self.inlet_frame , textvariable=self.inlet_press_entry_var)
        self.inlet_press.grid(row=3,column=1,padx=15,pady=10) 

        self.inlet_temp_label = customtkinter.CTkLabel(self.inlet_frame,text='Temperature')
        self.inlet_temp_label.grid(row=4,column=0,padx=15,pady=10)  

        self.inlet_temp_entry_var = customtkinter.StringVar()
        self.inlet_temp = customtkinter.CTkEntry(self.inlet_frame,textvariable=self.inlet_temp_entry_var)
        self.inlet_temp.grid(row=4,column=1,padx=15,pady=10) 

        self.inlet_mass_frac_label = customtkinter.CTkLabel(self.inlet_frame,text='Mass Fraction')
        self.inlet_mass_frac_label.grid(row=5,column=0,padx=15,pady=10) 

        self.inlet_mass_frac_entry_var = customtkinter.StringVar()
        self.inlet_mass_frac = customtkinter.CTkEntry(self.inlet_frame,textvariable=self.inlet_mass_frac_entry_var)
        self.inlet_mass_frac.grid(row=5,column=1,padx=15,pady=10) 

        self.inlet_period_checkbox_check_var = customtkinter.BooleanVar()
        self.inlet_period_checkbox = customtkinter.CTkCheckBox(self.inlet_frame , text='Periodic',variable=self.inlet_period_checkbox_check_var)
        self.inlet_period_checkbox.grid(row=6,column=0,padx=5, pady=10)

        self.inlet_wall_checkbox_check_var = customtkinter.BooleanVar()
        self.inlet_wall_checkbox = customtkinter.CTkCheckBox(self.inlet_frame , text='Wall',variable=self.inlet_wall_checkbox_check_var)
        self.inlet_wall_checkbox.grid(row=6,column=1,padx=5, pady=10)

        self.inlet_extrap_density_checkbox_check_var = customtkinter.BooleanVar()
        self.inlet_extrap_density_checkbox = customtkinter.CTkCheckBox(self.inlet_frame , text='Extrapolate',variable=self.inlet_extrap_density_checkbox_check_var)
        self.inlet_extrap_density_checkbox.grid(row=1,column=4)

        self.inlet_extrap_velocity_checkbox_check_var = customtkinter.BooleanVar()
        self.inlet_extrap_velocity_checkbox = customtkinter.CTkCheckBox(self.inlet_frame , text='Extrapolate',variable=self.inlet_extrap_velocity_checkbox_check_var)
        self.inlet_extrap_velocity_checkbox.grid(row=2,column=4)

        self.inlet_extrap_press_checkbox_check_var = customtkinter.BooleanVar()
        self.inlet_extrap_press_checkbox = customtkinter.CTkCheckBox(self.inlet_frame , text='Extrapolate',variable=self.inlet_extrap_press_checkbox_check_var)
        self.inlet_extrap_press_checkbox.grid(row=3,column=4)

        self.inlet_extrap_temp_checkbox_check_var = customtkinter.BooleanVar()
        self.inlet_extrap_temp_checkbox = customtkinter.CTkCheckBox(self.inlet_frame , text='Extrapolate',variable=self.inlet_extrap_temp_checkbox_check_var)
        self.inlet_extrap_temp_checkbox.grid(row=4,column=4)

        self.inlet_extrap_MF_checkbox_check_var = customtkinter.BooleanVar()
        self.inlet_extrap_MF_checkbox = customtkinter.CTkCheckBox(self.inlet_frame , text='Extrapolate',variable=self.inlet_extrap_MF_checkbox_check_var)
        self.inlet_extrap_MF_checkbox.grid(row=5,column=4)


        ###########################################################

        self.outlet_frame = customtkinter.CTkFrame(self.bc_windows,corner_radius=20)
        self.outlet_frame.grid(row=0,column=1,padx=10,pady=10)
        self.outlet_frame.columnconfigure(3,weight=1)

        self.outlet_label = customtkinter.CTkLabel(self.outlet_frame,text='Outlet Boundary Condition')
        self.outlet_label.grid(row=0,column=6,columnspan = 2)

        self.outlet_rho_label = customtkinter.CTkLabel(self.outlet_frame,text='Density')
        self.outlet_rho_label.grid(row=1,column=6,padx=15,pady=10)

        self.outlet_rho_entry_var = customtkinter.StringVar()
        self.outlet_rho = customtkinter.CTkEntry(self.outlet_frame,textvariable=self.outlet_rho_entry_var)
        self.outlet_rho.grid(row=1,column=7,padx=15,pady=10) 

        self.outlet_vel_label = customtkinter.CTkLabel(self.outlet_frame,text='Velocity')
        self.outlet_vel_label.grid(row=2,column=6,padx=15,pady=10) 

        self.outlet_vel_entry_var = customtkinter.StringVar()
        self.outlet_vel = customtkinter.CTkEntry(self.outlet_frame , textvariable=self.outlet_vel_entry_var)
        self.outlet_vel.grid(row=2,column=7,padx=15,pady=10) 

        self.outlet_press_label = customtkinter.CTkLabel(self.outlet_frame,text='Pressure')
        self.outlet_press_label.grid(row=3,column=6,padx=15,pady=10) 

        self.outlet_press_entry_var = customtkinter.StringVar()
        self.outlet_press = customtkinter.CTkEntry(self.outlet_frame , textvariable=self.outlet_press_entry_var)
        self.outlet_press.grid(row=3,column=7,padx=15,pady=10) 

        self.outlet_temp_label = customtkinter.CTkLabel(self.outlet_frame,text='Temperature')
        self.outlet_temp_label.grid(row=4,column=6,padx=15,pady=10) 

        self.outlet_temp_entry_var = customtkinter.StringVar()
        self.outlet_temp = customtkinter.CTkEntry(self.outlet_frame,textvariable=self.outlet_temp_entry_var)
        self.outlet_temp.grid(row=4,column=7,padx=15,pady=10) 

        self.outlet_mass_frac_label = customtkinter.CTkLabel(self.outlet_frame,text='Mass Fraction')
        self.outlet_mass_frac_label.grid(row=5,column=6,padx=15,pady=10) 

        self.outlet_mass_frac_entry_var = customtkinter.StringVar()
        self.outlet_mass_frac = customtkinter.CTkEntry(self.outlet_frame,textvariable=self.outlet_mass_frac_entry_var)
        self.outlet_mass_frac.grid(row=5,column=7,padx=15,pady=10) 

        self.outlet_period_checkbox_check_var = customtkinter.BooleanVar()
        self.outlet_period_checkbox = customtkinter.CTkCheckBox(self.outlet_frame , text='Periodic',variable=self.outlet_period_checkbox_check_var)
        self.outlet_period_checkbox.grid(row=6,column=6,padx=5, pady=10)

        self.outlet_wall_checkbox_check_var = customtkinter.BooleanVar()
        self.outlet_wall_checkbox = customtkinter.CTkCheckBox(self.outlet_frame , text='Wall',variable=self.outlet_wall_checkbox_check_var)
        self.outlet_wall_checkbox.grid(row=6,column=7,padx=5, pady=10)

        self.outlet_extrap_density_checkbox_check_var = customtkinter.BooleanVar()
        self.outlet_extrap_density_checkbox = customtkinter.CTkCheckBox(self.outlet_frame , text='Extrapolate',variable=self.outlet_extrap_density_checkbox_check_var)
        self.outlet_extrap_density_checkbox.grid(row=1,column=10)

        self.outlet_extrap_velocity_checkbox_check_var = customtkinter.BooleanVar()
        self.outlet_extrap_velocity_checkbox = customtkinter.CTkCheckBox(self.outlet_frame , text='Extrapolate',variable=self.outlet_extrap_velocity_checkbox_check_var)
        self.outlet_extrap_velocity_checkbox.grid(row=2,column=10)
        
        self.outlet_extrap_press_checkbox_check_var = customtkinter.BooleanVar()
        self.outlet_extrap_press_checkbox = customtkinter.CTkCheckBox(self.outlet_frame , text='Extrapolate',variable=self.outlet_extrap_press_checkbox_check_var)
        self.outlet_extrap_press_checkbox.grid(row=3,column=10)

        self.outlet_extrap_temp_checkbox_check_var = customtkinter.BooleanVar()
        self.outlet_extrap_temp_checkbox = customtkinter.CTkCheckBox(self.outlet_frame , text='Extrapolate',variable=self.outlet_extrap_temp_checkbox_check_var)
        self.outlet_extrap_temp_checkbox.grid(row=4,column=10)

        self.outlet_extrap_MF_checkbox_check_var = customtkinter.BooleanVar()
        self.outlet_extrap_MF_checkbox = customtkinter.CTkCheckBox(self.outlet_frame , text='Extrapolate',variable=self.outlet_extrap_MF_checkbox_check_var)
        self.outlet_extrap_MF_checkbox.grid(row=5,column=10)

        if hasattr(self, 'bc_data'):

            bc_array = self.bc_data

            inlet_att_list = ['inlet_rho_entry_var',
                            'inlet_vel_entry_var',
                            'inlet_press_entry_var',
                            'inlet_temp_entry_var',
                            'inlet_mass_frac_entry_var']

            outlet_att_list = ['outlet_rho_entry_var',
                            'outlet_vel_entry_var',
                            'outlet_press_entry_var',
                            'outlet_temp_entry_var',
                            'outlet_mass_frac_entry_var']
            
            inlet_extrap_list= ['inlet_extrap_density_checkbox_check_var',
                                'inlet_extrap_velocity_checkbox_check_var',
                                'inlet_extrap_press_checkbox_check_var',
                                'inlet_extrap_temp_checkbox_check_var',
                                'inlet_extrap_MF_checkbox_check_var']

            outlet_extrap_list = ['outlet_extrap_density_checkbox_check_var',
                                'outlet_extrap_velocity_checkbox_check_var',
                                'outlet_extrap_press_checkbox_check_var',
                                'outlet_extrap_temp_checkbox_check_var',
                                'outlet_extrap_MF_checkbox_check_var']

            condition_met = False

            if np.any(bc_array[:,0]) == 'periodic':
                
                self.inlet_period_checkbox_check_var.set(1)
                condition_met = True
                
            elif np.any(bc_array[:,0]) == 'wall':
            
                self.inlet_wall_checkbox_check_var.set(1)
                condition_met = True
            
            if not condition_met:

                for item in range(len(inlet_att_list)):

                    if bc_array[item,0] == 'extrapolate':
        
                        getattr(self, inlet_extrap_list[item]).set(1)
                        
                    else : 

                        getattr(self, inlet_att_list[item]).set(bc_array[item,0])

            condition_met = False

            if np.any(bc_array[:,1]) == 'periodic':

               self.outlet_period_checkbox_check_var.set(1)
               condition_met = True 

            elif np.any(bc_array[:,1]) == 'wall':
            
                self.outlet_wall_checkbox_check_var.set(1)
                condition_met = True
            
            if not condition_met:

                for item in range(len(outlet_att_list)):

                    if bc_array[item,1] == 'extrapolate':
        
                        getattr(self, outlet_extrap_list[item]).set(1)

                    else : 

                        getattr(self, outlet_att_list[item]).set(bc_array[item,1])

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

            for item in selected_item:

                self.tree.delete(item)
    
    def IC_row_remove_all(self):

        for item in self.tree.get_children():
            self.tree.delete(item)

    def IC_file_read_fun(self):
        
        self.new_windows.withdraw()
        IC_file_path = customtkinter.filedialog.askopenfilename()
        self.new_windows.deiconify()
        self.ic_data = np.load(IC_file_path)
        print('Reading IC File Sucessfully Completed')

    def pass_ic_data(self):

        data_array = []

        for item in self.tree.get_children():
            data = [self.tree.item(item, "values")]
            data_array.extend(data)

        self.ic_data = data_array
        self.new_windows.destroy()

    def pass_bc_data(self):
        
        inlet_att_list = ['inlet_rho_entry_var',
                          'inlet_vel_entry_var',
                          'inlet_press_entry_var',
                          'inlet_temp_entry_var',
                          'inlet_mass_frac_entry_var']

        outlet_att_list = ['outlet_rho_entry_var',
                          'outlet_vel_entry_var',
                          'outlet_press_entry_var',
                          'outlet_temp_entry_var',
                          'outlet_mass_frac_entry_var']
        
        inlet_extrap_list= ['inlet_extrap_density_checkbox_check_var',
                            'inlet_extrap_velocity_checkbox_check_var',
                            'inlet_extrap_press_checkbox_check_var',
                            'inlet_extrap_temp_checkbox_check_var',
                            'inlet_extrap_MF_checkbox_check_var']

        outlet_extrap_list = ['outlet_extrap_density_checkbox_check_var',
                              'outlet_extrap_velocity_checkbox_check_var',
                              'outlet_extrap_press_checkbox_check_var',
                              'outlet_extrap_temp_checkbox_check_var',
                              'outlet_extrap_MF_checkbox_check_var']

        bc_array = np.empty((5,2) , dtype='object')

        condition_met = False

        if self.inlet_period_checkbox_check_var.get():

            bc_array[:,0] = 'periodic'
            condition_met = True

        elif self.inlet_wall_checkbox_check_var.get():
            
            bc_array[:,0] = 'wall'
            condition_met = True

        if not condition_met:

            for item in range(len(inlet_att_list)):

                att     = getattr(self, inlet_att_list[item]).get()
                att_exp = getattr(self, inlet_extrap_list[item]).get()

                if att_exp:
                    
                    bc_array[item,0] = 'extrapolate'

                else : 

                    bc_array[item,0] = att

        condition_met = False

        if self.outlet_period_checkbox_check_var.get():

            bc_array[:,1] = 'periodic'
            condition_met = True

        elif self.outlet_wall_checkbox_check_var.get():
            
            bc_array[:,1] = 'wall'
            condition_met = True

        if not condition_met:

            for item in range(len(outlet_att_list)):

                att     = getattr(self, outlet_att_list[item]).get()
                att_exp = getattr(self, outlet_extrap_list[item]).get()

                if att_exp:
                    
                    bc_array[item,1] = 'extrapolate'

                else : 

                    bc_array[item,1] = att
    
        self.bc_data = bc_array
        self.bc_windows.destroy()

    def gas_model_callback(self,gas_model_entry_var):

        if gas_model_entry_var == 'H2/Air Combustion':

            self.viscous_checkbox_check_var.set(1)

        else:

            self.viscous_checkbox_check_var.set(0)

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

        if self.profile_check_var.get() == True:

            with cProfile.Profile() as pr:

                ui_solver_bridge.driver(self)
            
            stats = pstats.Stats(pr)
            stats.sort_stats(pstats.SortKey.TIME)
            stats.dump_stats(filename=self.working_dir_entry_var.get() +'/'+ 'profiling_results.prof')

        else: 

            ui_solver_bridge.driver(self)


                                                             
if __name__ == "__main__":

    app = UI()  
    app.mainloop()

