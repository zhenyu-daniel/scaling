import torch
import tomography_toolkit_dev.generator_module as generators
import tomography_toolkit_dev.discriminator_module as discriminators
import tomography_toolkit_dev.theory_module as theories
import tomography_toolkit_dev.experimental_module as experiments
import tomography_toolkit_dev.eventselection_module as filters
import tomography_toolkit_dev.expdata_module as data_parsers
from tomography_toolkit_dev.core.workflow_core import Workflow
from tomography_toolkit_dev.utils.data_science_mon.torch_gradient_monitor import Weight_Gradient_Monitor
from tomography_toolkit_dev.utils.data_science_mon.torch_gan_performance_monitor import GAN_Performance_Monitor
from tomography_toolkit_dev.utils.data_science_mon.torch_gan_performance_plotter import GAN_Performance_Plotter
from tomography_toolkit_dev.utils.helper_scripts.output_data_handler import Output_Data_Handler
from tomography_toolkit_dev.utils.physics_mon.torch_physics_monitor import Physics_Monitor
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import argparse
from datetime import datetime

class seq_workflow(Workflow):
    """
    Class to run the entire GAN training workflow in a sequence (i.e. not ditributed). The core function .run() has been divided into 6 sub-routines, in order to make this class a bit more accesible
    and easier to debug. The sub-routines are listed below:

    1.) .load_modules() #--> Load all required modules, such as theory, experiment, generator, discriminator, etc. Any new modules should be put there
    2.) .load_tools() #--> Load loss monitoring tools and other helpful functions (e.g. directory creator or visualization)
    3.) .load_experimental_data() #--> Load the experimental data from a .npy array. Any changes to the experimental data format / outputs / etc. should be done here
    4.) .handle_output_data_storage() #--> This just creates a main directory with a few sub-directories where all the results from the GAN training are stored. Go here if you wish to add / remove a directory
    5.) .fit() #--> This runs the actual training loop. Any updates / changes to the training procedure are done here
    6.) .visualize_and_store_results() #--> This evaluates the data that have been collected during the training. If you wish to add more or remove plots, then this is the place to go
    """

    # INITIALIZE:

    #*****************************************************
    def __init__(self, module_names,
                       generator_config,
                       discriminator_config,
                       theory_config,
                       experimental_config,
                       training_config,
                       data_config,
                       devices="cpu"):

        # Get the basic settings:

        # Configurations:
        self.module_names = module_names
        self.generator_config = generator_config
        self.discriminator_config = discriminator_config
        self.theory_config = theory_config
        self.experimental_config = experimental_config
        self.training_config = training_config
        self.data_config = data_config

        # Define an argparser, so that we can overwrite specific configuration settings:
        workflow_parser = argparse.ArgumentParser(description='Proxy Sequential Workflow',prog='driver.py')
        
        # set the random seed:
        workflow_parser.add_argument('--seed',type=int,default=self.read_settings_from_config("seed",training_config,-1),metavar='S',help='Random Seed')
        # batch size:
        workflow_parser.add_argument('--batch_size',type=int,default=self.read_settings_from_config("batch_size",training_config,64),metavar='N',help='Batch size for training')
        # number of events that are generated for each predicted parameter:
        workflow_parser.add_argument('--num_events_per_parameter_set',type=int,default=self.read_settings_from_config("num_events_per_parameter_set", training_config, 1),metavar='N',help='Number of events generated for each predicted parameter')
        # number of epochs:
        workflow_parser.add_argument('--num_epochs',type=int,default=self.read_settings_from_config("num_epochs",training_config,100000),metavar='N',help='Number of training epochs')
        # number of events to use analyze / use for training:
        workflow_parser.add_argument('--n_events_to_analyze',type=int,default=self.read_settings_from_config("n_events_to_analyze",training_config,-1),metavar='N',help='Number of events that shall be used for training')
        # number of samples in the final analysis:
        workflow_parser.add_argument('--n_final_analysis_samples',type=int,default=self.read_settings_from_config("n_final_analysis_samples",training_config,50000),metavar='N',help='Number of samples in final analysis')
        # number of samples in the snapshot analysis:
        workflow_parser.add_argument('--n_snapshot_samples',type=int,default=self.read_settings_from_config("n_snapshot_samples",training_config,5000),metavar='N',help='Number of samples to analyize during one snapshot')
        # snapshot epoch:
        workflow_parser.add_argument('--snapshot_epoch',type=int,default=self.read_settings_from_config("snapshot_epoch",training_config,0),metavar='N',help='Snapshot is taken every Nth epoch')
        # print info epoch:
        workflow_parser.add_argument('--print_info_epoch',type=int,default=self.read_settings_from_config("print_info_epoch",training_config,1000),metavar='N',help='Print info every Nth epoch')
        # read performance epoch:
        workflow_parser.add_argument('--read_performance_epoch',type=int,default=self.read_settings_from_config("read_performance_epoch",training_config,1000),metavar='N',help='Read out performance metrics every Nth epoch')
        # name of the result folder (i.e. where all the exciting stuff is stored):
        workflow_parser.add_argument('--result_folder',type=str,default=self.read_settings_from_config("result_folder",training_config,"results_GAN_training"),metavar='Folder',help='Name of the folder where all results are stored')
        # name of the directory where the results shall be stored:
        workflow_parser.add_argument('--output_loc',type=str,default=self.read_settings_from_config("output_loc",training_config,""),metavar='Dir',help='Name of directory where the result folder will be stored')
        # define the full path where the training data is stored:
        workflow_parser.add_argument('--path_to_data',type=str,default=self.data_config["path"],metavar='Dir',help='Full path to training data')
        # use optional time stamp when saving models:
        workflow_parser.add_argument('--use_timestamp',action='store_true', default=self.read_settings_from_config("use_timestamp",training_config,False),help='Store models with timestamp')
        # decide if you want to store the models only and nothing else:
        workflow_parser.add_argument('--store_models_only',action='store_true',default=self.read_settings_from_config("store_models_only",training_config,False),help='Write only models to file')

        workflow_args = workflow_parser.parse_args()

        # Training:
        self.devices = devices
        self.batch_size = workflow_args.batch_size
        self.n_events_to_analyze = workflow_args.n_events_to_analyze
        self.do_test_run = self.read_settings_from_config("do_test_run",training_config,False)
        self.num_epochs = workflow_args.num_epochs
        self.print_info_epoch = workflow_args.print_info_epoch
        self.read_performance_epoch = workflow_args.read_performance_epoch
        self.n_final_analysis_samples = workflow_args.n_final_analysis_samples
        self.snapshot_epoch = workflow_args.snapshot_epoch
        self.n_snapshot_samples = workflow_args.n_snapshot_samples
        self.noise_mean = self.read_settings_from_config("noise_mean",training_config,0.0)
        self.noise_std = self.read_settings_from_config("noise_std",training_config,1.0)
        self.noise_dim = self.read_settings_from_config("input_size",generator_config,10)
        self.watch_gradients = self.read_settings_from_config("watch_gradients",training_config,False)
        self.num_events_per_parameters = workflow_args.num_events_per_parameter_set
        self.data_dim = len(self.read_settings_from_config("observable_names", data_config, ["o1", "o2"]))
        self.seed = workflow_args.seed
        self.use_timestamp = workflow_args.use_timestamp
        self.store_models_only = workflow_args.store_models_only

        # Make sure that we monitor something during training:
        if self.read_performance_epoch <= 0:
            self.read_performance_epoch = 1

        # Decide if and where to store results / information:
        self.output_loc = workflow_args.output_loc
        self.result_folder = workflow_args.result_folder
        self.store_results_as_npy = self.read_settings_from_config("store_results_as_npy",training_config,False)
        self.write_cfg_to_json = self.read_settings_from_config("write_cfg_to_json",training_config,False)
        self.write_cfg_to_py = self.read_settings_from_config("write_cfg_to_py",training_config,False)
        self.get_software_info = self.read_settings_from_config("get_software_info",training_config,False)

        # Experimental data:
        self.true_params = self.read_settings_from_config("truth",data_config,None)
        self.observable_names = self.read_settings_from_config("observable_names",data_config,[str(k) for k in range(1,1+discriminator_config['input_size'])])
        # Overwrite the full path to the available data:
        self.data_config["path"] = workflow_args.path_to_data


        # Networks:
        # TODO: This has not been implemented yet, the ability to change the loss function of each network. In the current version, both networks
        # use the mean squared error, but this will change in the future. For now, we use the MSE as default.
        self.disc_loss_fn = self.read_settings_from_config("loss_fn",discriminator_config,"MSE",disable_info=True)
        self.gen_loss_fn = self.read_settings_from_config("loss_fn",generator_config,"MSE",disable_info=True)

        # Name of this workflow
        self.name = "sequential_workflow_v0"

        # Set the seed (if specified)
        if self.seed > 0:
            torch.manual_seed(self.seed) 
            # According to the documentation here: https://pytorch.org/docs/stable/notes/randomness.html
            # This sets the seed for all devices...

        # Print an intro:
        print(" ")
        print("******************************")
        print("*                            *")
        print("*   Sequential Workflow v0   *")
        print("*                            *")
        print("******************************")
        print(" ")
    #*****************************************************

    # READ IN THE SETTINGS FROM A CONFIGURATION:

    #*****************************************************
    def read_settings_from_config(self,setting_str,dict,alternative_setting,disable_info=False):
        if setting_str in dict:
            return dict[setting_str]
        else:
            if disable_info == False:
               print(" ")
               print(">>> INFO: " + setting_str + " not found in config. Going to use default value: " + str(alternative_setting) + " <<<")
               print(" ")

            return alternative_setting
    #*****************************************************

    # GET TIME STAMP FOR ADDITIONAL INFO / NAMING CONVENTION

    #*****************************************************
    def get_timestamp_extension(self):
        if self.use_timestamp == True:
            return '_' + datetime.now().strftime('%dd_%mm_%yy_%Hh_%Mm_%Ss')
        return ''
    #*****************************************************

    # WRITE MODEL TO FILE:

    #*****************************************************
    def write_model_to_file(self,model,full_path):
        model_state_dict = model.state_dict()
        
        if self.use_timestamp == True:
            model_state_dict['timestamp'] = time.time()

        torch.save(model_state_dict,full_path)
    #*****************************************************


    # DEFINE ANALYSIS PIPELINES:

    #*****************************************************
     # Define the pipeline for inside the training loop:
    def training_pipeline(self,params):
        fake_events, _, _ = self.module_chain_excluding_gen[0].forward(params, self.num_events_per_parameters)
        fake_events = torch.reshape(torch.transpose(fake_events, 1, 2), (-1, self.data_dim))
        fake_events = torch.squeeze(fake_events) #--> Remove unnecessary dimensions

        if len(self.module_chain_excluding_gen) > 2:
            for module in self.module_chain_excluding_gen[1:-1]:
                fake_events = module.forward(fake_events) #Here we use .forward(), because we are inside the training loop
                    # i.e. we are updating the discriminator and generator.

        return fake_events

    #-------------------------------------------------

    # Define the exact same pipeline, but for usage outside the training loop:
    def analysis_pipeline(self,params):
        fake_events, _, _ = self.module_chain_excluding_gen[0].apply(params)
        fake_events = torch.squeeze(fake_events) #--> Remove unnecessary dimensions

        if len(self.module_chain_excluding_gen) > 2:
            for module in self.module_chain_excluding_gen[1:-1]:
                fake_events = module.apply(fake_events)  # We are currently not updating the models, so we are using .apply() here

        return fake_events

    # Note: The difference between .forward() and .apply() is mainly, that .forward() is used during the GAN training stage, whereas
    # .apply() is used for inference. Depending on the module, there might be no difference between these two, i.e. .forward() == .apply()
    # However, there might be modules that require a different behavior in .forward() and .apply(), in order to guarantee differentiability.
    #*****************************************************

    # LOAD THE INDIVIDUAL MODULES:

    #*****************************************************
    def load_modules(self):
        # Theory:
        self.theory = theories.make(self.module_names['theory'], config=self.theory_config, devices=self.devices)
        # Experiment:
        self.experiment = experiments.make(self.module_names['experiment'],config=self.experimental_config,devices=self.devices)
        # Event selection: (aka filter)
        self.filter = filters.make(self.module_names['event_selection'],config=self.experimental_config,devices=self.devices)
        # Discriminator
        self.discriminator = discriminators.make(self.module_names['discriminator'], config=self.discriminator_config, devices=self.devices)

        # Put everything together:
        self.module_chain_excluding_gen = [self.theory,self.experiment,self.filter,self.discriminator]
        # Now set the generator
        self.generator = generators.make(self.module_names['generator'], config=self.generator_config, module_chain=self.module_chain_excluding_gen,devices=self.devices)

        # Get the data module which loads and handles the experimental data:
        self.data_parser = data_parsers.make(self.module_names['data_parser'],config=self.data_config,module_chain=self.module_chain_excluding_gen,devices=self.devices)
    #*****************************************************

    # LOAD HELPFUL TOOLS (MAINLY MONITORING):

    #*****************************************************
    def load_tools(self):
        # TODO: The plots produced by each monitoring tool can be adjusted by the user, via a visualization config file.
        # All tools, except the physics monitor, do not require an explicitly config file, i.e. they use default settings. 
        # For now, the visualization config is set here, but we might want to move it to the main cfg...

        gan_performance_plotter_cfg = {
           'observable_names':self.observable_names, #--> Names of the features in the real data
           'parameter_residual_format': [1,6]
        }

        physics_monitor_cfg = {
            'x_labels': ['x','x'],
            'y_labels': [r'$f_{u}(x,Q^{2})$',r'$f_{d}(x,Q^{2})$'],
            'dist_names': ['u','d'],
            'pdf_format': [1,2]
        }

        # i) Gradient monitors:
        self.disc_grad_mon = Weight_Gradient_Monitor(self.discriminator)
        self.gen_grad_mon = Weight_Gradient_Monitor(self.generator)
        # ii) Performance monitor:
        self.performance_monitor = GAN_Performance_Monitor(
            generator=self.generator, #--> Generator network
            discriminator=self.discriminator, #--> Discriminator network
            data_pipeline=self.analysis_pipeline, #--> Analysis pipeline which translates params to events
            generator_noise_mean=self.noise_mean, #--> Mean of the gaussian noise
            generator_noise_std=self.noise_std, #--> Std. dev of the gaussian noise
            n_features=self.noise_dim, #--> Dimension of the the noise / parameters
            disc_loss_fn_str=self.disc_loss_fn, #--> String for the discriminator loss (allowed values are: "mse", "mae" and "bce")
            gen_loss_fn_str=self.gen_loss_fn, #--> String for the generator loss (allowed values are: "mse", "mae" and "bce")
            device=self.devices #--> CPU vs GPU
        )
        # iii) Performance plotter, i.e. a class that translates the results fromt the performance monitor into plots and figures
        self.performance_plotter = GAN_Performance_Plotter(config=gan_performance_plotter_cfg)
        # iv) Load a class that helps to handle the storage of the produced output data:
        self.out_dat_handler = Output_Data_Handler()

        # v) Get meaningful distributions from the theory module:
        # TODO: This might need a better solution. For now, we implement this function manually. But in the future, one might want this to be part
        # of the theory module. Something like: params_to_physics_fn = theory.get_params_to_physics_fn ....
        def get_ud(p,n_points=100):
            xmin, xmax = 0.1, 0.99999
            dx = (xmax - xmin) / n_points

            x_full_range = torch.arange(xmin,xmax,dx).to(self.devices)
            u = p[0]*torch.pow(x_full_range, p[1]) * torch.pow((1-x_full_range), p[2]).to(self.devices)
            d = p[3]*torch.pow(x_full_range, p[4]) * torch.pow((1-x_full_range), p[5]).to(self.devices)
            return u, d, x_full_range

        # Now we can load the physics monitor:
        self.physics_monitor = Physics_Monitor(
            params_to_physics_func=get_ud, #--> We use the function that we just defined to translate the generator parameters to a meaningful physics distribution
            generator=self.generator,
            true_params=torch.tensor(self.true_params),
            noise_mean=self.noise_mean,
            noise_std=self.noise_std,
            config=physics_monitor_cfg,
            device=self.devices
        )
    #*****************************************************

    # LOAD THE EXPERIMENTAL DATA:

    #*****************************************************
    def load_experimental_data(self):
        # Load data packages from data module:
        training_data_pkg = self.data_parser.return_data(is_training=True,additional_observable_names=['norm1','norm2']) #--> This is the data we are going to use for training
        analysis_data_pkg = self.data_parser.return_data(is_training=False,additional_observable_names=['norm1','norm2']) #--> This is the data we are going to use for analysis (e.g. making nice plots)

        # The elements in the data package are stored in a dictionary
        self.training_data = training_data_pkg['parsed_data']
        self.analysis_data = analysis_data_pkg['parsed_data']
        # Get the clean data, i.e. the data before all the modules:
        self.clean_data = analysis_data_pkg['original_data']

        if self.n_events_to_analyze > 0:
            self.training_data = self.data_parser.get_random_batch_from_data(self.training_data,self.n_events_to_analyze)
            self.analysis_data = self.data_parser.get_random_batch_from_data(self.analysis_data,self.n_events_to_analyze)
            self.clean_data = self.data_parser.get_random_batch_from_data(self.clean_data,self.n_events_to_analyze)

        self.norms = analysis_data_pkg['norm1'].repeat(self.batch_size), analysis_data_pkg['norm2'].repeat(self.batch_size)
    #*****************************************************

    # OUTPUT DATA STORAGE:

    #*****************************************************
    def handle_output_data_storage(self):
        # Define main directory:
        self.main_dir = self.out_dat_handler.create_output_data_folder(folder_name=self.result_folder,top_level_dir=self.output_loc)

        # Directory for configurations: (if required)
        self.config_dir = None
        if self.write_cfg_to_json == True or self.write_cfg_to_py == True:
           self.config_dir = self.out_dat_handler.create_output_data_folder(folder_name="configs",top_level_dir=self.main_dir)

        # Directory for storing software related information:
        self.software_info_dir = None
        if self.get_software_info == True:
            self.software_info_dir = self.out_dat_handler.create_output_data_folder(folder_name="software",top_level_dir=self.main_dir)

        # Define directories where all the plots are stored:
        self.data_science_dir = self.out_dat_handler.create_output_data_folder(folder_name="data_science_mon",top_level_dir=self.main_dir)
        self.physics_dir = self.out_dat_handler.create_output_data_folder(folder_name="physics_mon",top_level_dir=self.main_dir)
        self.snapshot_dir = None
        if self.snapshot_epoch > 0 and self.store_models_only == False:
           self.snapshot_dir = self.out_dat_handler.create_output_data_folder(folder_name="snaphots",top_level_dir=self.main_dir)

        # Store the data as numpy arrays (if requested):
        self.data_science_npy_dir = None
        self.physics_npy_dir = None
        self.snapshot_npy_dir = None
        if self.store_results_as_npy == True and self.store_models_only == False:
           self.data_science_npy_dir = self.out_dat_handler.create_output_data_folder(folder_name="data_science_npy",top_level_dir=self.main_dir)
           self.physics_npy_dir = self.out_dat_handler.create_output_data_folder(folder_name="physics_npy",top_level_dir=self.main_dir)

           if self.snapshot_epoch > 0:
               self.snapshot_npy_dir = self.out_dat_handler.create_output_data_folder(folder_name="snapshot_npy",top_level_dir=self.main_dir)

        # Store the model (stages) for post training analysis:
        self.model_dir = self.out_dat_handler.create_output_data_folder(folder_name="model_stages",top_level_dir=self.main_dir)
    #*****************************************************

    # TRAINING

    #*****************************************************
    def fit(self):

        t_start = time.perf_counter() #--> Using this with GPU and improper synchronization might give inacurrate results (see this post: https://discuss.pytorch.org/t/why-time-time-in-python-is-inaccurte/94274)
        # TODO: This should be replaced by a more proper time keeping procedure.
        # However, we use this simple approach for now, in order to get a rough estimate for the time during training. In summary: please use the time information with caution!
        t_unit = "s"
        total_time = 0.0

        start_epoch = 1
        if self.do_test_run == True:
            start_epoch = 0
        #+++++++++++++++++++++++++++++++++++
        for epoch in range(start_epoch,1+self.num_epochs):

            if epoch > 0:
               # >>> One training cycle: <<<
               #----------------------------------------------------

               # Retreive (random) batch of the real data:
               real_events = self.data_parser.get_random_batch_from_data(data=self.training_data,batch_size=self.batch_size*self.num_events_per_parameters)

               # Generate the fake data:
               noise = torch.normal(self.noise_mean, self.noise_std, size=(self.batch_size,self.noise_dim),device=self.devices)
               fake_parameters = self.generator.generate(noise)
               fake_events = self.training_pipeline(fake_parameters)

               # Train each network:
               d_losses = self.discriminator.train(real_events, fake_events)
               g_losses = self.generator.train(noise, self.norms)
               #----------------------------------------------------


               # >>> Collect performance metrics and other information: <<<
               #----------------------------------------------------
               # a) Loss and accuracy:

               # (We call the watch_() function for every batch / every update in the training data)
               self.performance_monitor.watch_losses_and_accuracy_per_batch(
                  real_data=real_events,
                  fake_data=fake_events,
                  gen_loss=g_losses[0],
                  disc_real_loss=d_losses[0],
                  disc_fake_loss=d_losses[1])

            # b) Gradients:
            if epoch > 0:
               if self.watch_gradients == True:
                  self.disc_grad_mon.watch_gradients_per_batch(sample_size=self.read_performance_epoch)
                  self.gen_grad_mon.watch_gradients_per_batch(sample_size=self.read_performance_epoch)

            # Read out the accumulated metrics / information:
            if epoch % self.read_performance_epoch == 0 and epoch > 0:
                # a) Losses and accuracy:
                self.performance_monitor.collect_losses_and_accuracy_per_epoch()
                # b) Residuals
                self.performance_monitor.collect_residual_information_per_epoch()
                # c) And the gradients:
                if self.watch_gradients == True:
                    self.disc_grad_mon.collect_gradients_per_epoch()
                    self.gen_grad_mon.collect_gradients_per_epoch()
            #----------------------------------------------------


            # >>> Take a snapshot of the current model predictions: <<<
            #----------------------------------------------------
            if self.snapshot_epoch > 0 and epoch % self.snapshot_epoch == 0:
                epoch_str = str(epoch) + 'epochs'
                epoch_str = epoch_str.zfill(6 + len(str(self.num_epochs)))

                # Unlike in previous versions, we are NOT producing any plots during the training.
                # Instead, we store the models themselves for each training stage. After running the training,
                # we can then analyze every model for every epoch. This might not be the best solution, but it helps to disentangle the resources needed for the actual training
                # and those needed for simply producing nice plots
                if epoch < self.num_epochs:
                    self.write_model_to_file(self.generator,self.model_dir+'/generator_'+epoch_str+'.pt')
                    self.write_model_to_file(self.discriminator,self.model_dir+'/discriminator_'+epoch_str+'.pt')

                #    torch.save(self.generator.state_dict(),self.model_dir+'/generator_'+epoch_str+'.pt')
                #    torch.save(self.discriminator.state_dict(),self.model_dir+'/discriminator_'+epoch_str+'.pt')
            #----------------------------------------------------


            # >>> Print out some basic information: <<<
            #----------------------------------------------------
            if epoch % self.print_info_epoch == 0:
                t_end = time.perf_counter()
                dt = t_end - t_start
                total_time += dt

                if dt >= 60.0 and dt < 3600.0:
                    t_unit = "min"
                    dt /= 60.0
                elif dt >= 3600.0:
                    t_unit = "h"
                    dt /= 3600.0

                print(" ")
                print("Epoch: " + str(epoch) + "/" + str(self.num_epochs))
                self.performance_monitor.print_losses_and_accuracy()
                print(">>> Time needed for " + str(self.print_info_epoch) + " epochs: ~" + str(round(dt,3)) + " " + t_unit + " (Caution: This is a crude estimate!) <<<")
                print(" ")

                t_start = time.perf_counter()
                t_unit = "s"
            #----------------------------------------------------

            # Fin! Thats all folks!
        #+++++++++++++++++++++++++++++++++++

        # Store the 'final' models:
        epoch_str = str(self.num_epochs) + 'epochs'
        self.write_model_to_file(self.generator,self.model_dir+'/generator_'+epoch_str+'.pt')
        self.write_model_to_file(self.discriminator,self.model_dir+'/discriminator_'+epoch_str+'.pt')

        # torch.save(self.generator.state_dict(),self.model_dir+'/generator_'+epoch_str+'.pt')
        # torch.save(self.discriminator.state_dict(),self.model_dir+'/discriminator_'+epoch_str+'.pt')

        return total_time
    #*****************************************************


    # VISUALIZE AND STORE THE RESULTS

    #*****************************************************
    # Load the model weights for each training stage:
    def load_model_weights(self,current_epoch):
        epoch_str = str(current_epoch) + 'epochs'
        
        if current_epoch < self.num_epochs:
            epoch_str = epoch_str.zfill(6 + len(str(self.num_epochs)))

        discriminator_state = torch.load(self.model_dir+'/discriminator_'+epoch_str+'.pt')
        generator_state = torch.load(self.model_dir+'/generator_'+epoch_str+'.pt')
        
        self.discriminator.load_state_dict(discriminator_state)
        self.generator.load_state_dict(generator_state)

        return epoch_str
    
    #----------------------------------

    # Run snapshot analysis:
    def snaphsot_analysis(self):
        # We loop over all models that have been stored (if snapshot_epoch > 0):
        #++++++++++++++++++++++++
        for epoch in range(1,1+self.num_epochs):
            if epoch % self.snapshot_epoch == 0:
                epoch_str = self.load_model_weights(epoch)

                # Take a small fraction of the available data to run some basic monitoring tests:
                monitoring_data_chunk = self.data_parser.get_random_batch_from_data(data=self.analysis_data,batch_size=self.n_snapshot_samples)
                current_params, current_gen_data, current_residuals = self.performance_monitor.watch_residual_information_per_batch(real_data=monitoring_data_chunk)

                # Take a snapshot of the current observables:
                gen_data_plot_dict = self.performance_plotter.plot_observables_and_residuals(
                    real_data=monitoring_data_chunk.cpu().numpy(),
                    generated_data=current_gen_data.cpu().numpy(),
                    residuals=current_residuals.cpu().numpy(),
                    )
                #++++++++++++++++++++
                for key in gen_data_plot_dict:
                    current_plot = gen_data_plot_dict[key]

                    if current_plot is not None:
                        current_fig = current_plot[0]
                        current_fig.suptitle('Snapshot for Epoch: ' + str(epoch))
                        save_name = self.snapshot_dir + '/' + key + '_' + epoch_str + '.png'
                        current_fig.savefig(save_name)
                        plt.close(current_fig) 
                #++++++++++++++++++++

                # Take a snapshot of the parameter residuals:
                pred_param_dict = self.performance_plotter.plot_parameter_residuals(
                    true_parameters=np.array(self.true_params),
                    pred_parameters=current_params.cpu().numpy(),
                    )
                
                #++++++++++++++++++++
                for key in pred_param_dict:
                    current_plot = pred_param_dict[key]

                    if current_plot is not None:
                        save_name = self.snapshot_dir + '/' + key + '_' + epoch_str + '.png'
                        current_fig  = current_plot[0]
                        current_fig.suptitle('Snapshot for Epoch: ' + str(epoch))
                        current_fig.savefig(save_name)
                        plt.close(current_fig)
                #++++++++++++++++++++  

                # Take a snapshot of the pdfs:
                pdf_plots = self.physics_monitor.compare_true_and_generated_distributions(
                    true_npy_data=self.physics_monitor.create_true_distributions(),
                    generated_npy_data = self.physics_monitor.create_gen_distributions(self.n_snapshot_samples)
                )

                #++++++++++++++++++++ 
                for key in pdf_plots:
                    current_plot = pdf_plots[key]
                    if current_plot is not None:
                        current_fig = current_plot[0]
                        save_name = self.snapshot_dir + '/' + key + '_' + epoch_str + '.png'
                        current_fig.suptitle('Snapshot for Epoch: ' + str(epoch))
                        current_fig.savefig(save_name)
                        plt.close(current_fig)
                #++++++++++++++++++++ 

                # Store the snapshot data as npy arrays:
                if self.store_results_as_npy:
                    np.save(self.snapshot_npy_dir + '/generated_data_' + epoch_str + '.npy',current_gen_data.cpu().numpy())
                    np.save(self.snapshot_npy_dir + '/residuals_' + epoch_str + '.npy',current_residuals.cpu().numpy())

                    param_residual = np.array(self.true_params) - current_params.cpu().numpy()
                    np.save(self.snapshot_npy_dir + '/parameter_residuals_' + epoch_str + '.npy',param_residual)
        #++++++++++++++++++++++++

    #----------------------------------

    # Run the post training analysis:
    def post_training_analysis(self,n_samples):
          # Important note: All the plots / histograms that are used here, are saved in dictionaries where the keys (more or less) represent the plotted quantity.
          # Each entry in the dictionary returns a list, where the first elemtent is a pyplot figure and the second element is a pyplot axis.
          # The figure element can be used to store the resutls, whereas the axis element can be used to alter the plot (e.g. change the axis label, font size, etc.)
        
          # Loss and accuracy:
          loss_accuracy_dict = self.performance_monitor.read_out_losses_and_accuracy()
         
        # Plot losses and accuracy as a function of the training epoch:
          loss_plots = self.performance_plotter.plot_losses_and_accuracy(
             disc_real_loss=loss_accuracy_dict['disc_real_loss'], #--> Discriminator loss on real data
             disc_fake_loss=loss_accuracy_dict['disc_fake_loss'], #--> Discriminator loss on fake data
             gen_loss=loss_accuracy_dict['gen_loss'], #--> Generator loss,
             disc_real_accuracy=loss_accuracy_dict['disc_real_acc'], #--> Discriminator accuracy on real data
             disc_fake_accuracy=loss_accuracy_dict['disc_fake_acc'], #--> Discriminator accuracy on fake data
             x_label = 'Training Epoch per ' + str(self.read_performance_epoch) #--> Set the x-label properly
          )

          fig_loss = loss_plots['loss_and_accuracy_plots'][0]
          fig_loss.savefig(self.data_science_dir + '/losses_and_accuracies.png')
          plt.close(fig_loss)

        
          # Calculate actual residuals:
          params,generated_data, residuals = self.performance_monitor.predict_observables_and_residuals(real_data=self.analysis_data,n_samples=n_samples)
          monitoring_data = residuals + generated_data

          # Get some physics distributions (here: pdfs)
          true_pdfs = self.physics_monitor.create_true_distributions()
          generated_pdfs = self.physics_monitor.create_gen_distributions(n_samples)

           # Store the results to .npy files, if wanted:
          if self.store_results_as_npy == True and self.store_models_only == False:

            # Losses and accuracies:
            np.save(self.data_science_npy_dir + '/discriminator_real_loss.npy',loss_accuracy_dict['disc_real_loss'])
            np.save(self.data_science_npy_dir + '/discriminator_fake_loss.npy',loss_accuracy_dict['disc_fake_loss'])
            np.save(self.data_science_npy_dir + '/generator_loss.npy',loss_accuracy_dict['gen_loss'])
            np.save(self.data_science_npy_dir + '/discriminator_real_accuracy.npy',loss_accuracy_dict['disc_real_acc'])
            np.save(self.data_science_npy_dir + '/discriminator_fake_accuracy.npy',loss_accuracy_dict['disc_fake_acc'])
            
            # Residuals:
            np.save(self.data_science_npy_dir + '/generated_data.npy',generated_data.cpu().numpy())
            np.save(self.data_science_npy_dir + '/residuals.npy',residuals.cpu().numpy())

            # Parameter residuals:
            parameter_residuals = np.array(self.true_params) - params.cpu().numpy()
            np.save(self.data_science_npy_dir + '/parameter_residuals.npy',parameter_residuals)

            # Write pdfs to file:
            #++++++++++++++++++++++++
            for j in range(len(true_pdfs)-1):
                np.save(self.physics_npy_dir + '/true_' + self.physics_monitor.dist_names[j] + '.npy',true_pdfs[j])
                np.save(self.physics_npy_dir + '/generated_' + self.physics_monitor.dist_names[j] + '.npy',generated_pdfs[j])
            #++++++++++++++++++++++++
            np.save(self.physics_npy_dir + '/pdf_x.npy',true_pdfs[len(true_pdfs)-1])

          # Check the gradient flow:
          if self.watch_gradients == True:
            disc_gradients = self.disc_grad_mon.read_out_gradients()
            gen_gradients = self.gen_grad_mon.read_out_gradients()

            disc_grad_plots = self.disc_grad_mon.show_gradients(gradient_dict=disc_gradients,model_name='Discriminator',xlabel='Epoch per ' + str(self.read_performance_epoch))
            gen_grad_plots = self.gen_grad_mon.show_gradients(gradient_dict=gen_gradients,model_name='Generator',xlabel='Epoch per ' + str(self.read_performance_epoch))

            disc_grad_fig = disc_grad_plots['gradient_flow_Discriminator'][0]
            gen_grad_fig = gen_grad_plots['gradient_flow_Generator'][0]

            disc_grad_fig.savefig(self.data_science_dir + '/discriminator_gradient_flow.png')
            plt.close(disc_grad_fig)

            gen_grad_fig.savefig(self.data_science_dir + '/generator_gradient_flow.png')
            plt.close(gen_grad_fig)

          # Now plot the full residuals:
          gen_data_plots = self.performance_plotter.plot_observables_and_residuals(
             real_data=monitoring_data.cpu().numpy(), #--> Analysis data
             generated_data=generated_data.cpu().numpy(), #--> Generated / predicted data
             residuals=residuals.cpu().numpy(), #--> Residuals = Training data - Generated data
          )

          #+++++++++++++++++++++++
          for key in gen_data_plots:
            current_plot = gen_data_plots[key]

            if current_plot is not None:
                current_fig = current_plot[0]
                save_name = self.data_science_dir + '/' + key + '.png'
                current_fig.savefig(save_name)
                plt.close(current_fig)
          #+++++++++++++++++++++++

          # Check the residuals between the true parameters and the predicted ones:
          param_residual_plots = self.performance_plotter.plot_parameter_residuals(
            true_parameters=np.array(self.true_params),
            pred_parameters=params.cpu().numpy())
        
          #+++++++++++++++++++++
          for key in param_residual_plots:
            current_plot = param_residual_plots[key]

            if current_plot is not None:
                save_name =  self.data_science_dir + '/' + key + '.png'
                current_fig = current_plot[0]
                current_fig.savefig(save_name)
                plt.close(current_fig)
          #+++++++++++++++++++++

          # Finally, we can look at some physics distributions:
          pdf_plots = self.physics_monitor.compare_true_and_generated_distributions(
            true_npy_data=true_pdfs,
            generated_npy_data = generated_pdfs)

          #+++++++++++++++++++++
          for key in pdf_plots:
            current_plot = pdf_plots[key]
            if current_plot is not None:
                current_fig = current_plot[0]
                current_fig.suptitle('Comparing True and Generated PDFs')
                current_fig.savefig(self.physics_dir + '/' + key + '.png')
                plt.close(current_fig)
          #+++++++++++++++++++++
    #*****************************************************

    # STORE CONFIGURATION FILES, CONDA ENVIRONMENTS AND OTHER SOFTWARE RELATED INFORMATION

    #*****************************************************
    # Configurations:
    def store_configurations(self):
        # Write the configurations to .json files, if wanted:
        if self.write_cfg_to_json == True:
            # Theory:
            self.out_dat_handler.write_config_to_json_file(self.config_dir + '/theory_cfg',self.theory_config)
            # Experiment & event selection
            self.out_dat_handler.write_config_to_json_file(self.config_dir + '/experiment_cfg',self.experimental_config)
            # Discriminator
            self.out_dat_handler.write_config_to_json_file(self.config_dir + '/discriminator_cfg',self.discriminator_config)
            # Generator:
            self.out_dat_handler.write_config_to_json_file(self.config_dir + '/generator_cfg',self.generator_config)
            # Data:
            self.out_dat_handler.write_config_to_json_file(self.config_dir + '/data_cfg',self.data_config)
            # Training:
            self.out_dat_handler.write_config_to_json_file(self.config_dir + '/training_cfg',self.training_config)
            # Module names:
            self.out_dat_handler.write_config_to_json_file(self.config_dir + '/module_names',self.module_names)

        # Write entire configuration to .py file, if wanted:
        if self.write_cfg_to_py == True:
            # Create a list with all the configurations we would like to store:
            conf_list = [
               ['training_config',self.training_config],
               ['generator_config',self.generator_config],
               ['discriminator_config',self.discriminator_config],
               ['theory_config',self.theory_config],
               ['module_names',self.module_names],
               ['data_config',self.data_config],
               ['experimental_config',self.experimental_config]
            ]

            self.out_dat_handler.write_config_to_py_file(self.config_dir + '/configuration',conf_list)

    #-----------------------------

    # Software:
    def store_software_info(self):
        # Conda environment:
        write_cond_env = "conda env export > " + self.software_info_dir + "/environment.yaml"
        os.system(write_cond_env)

        # Git hashes:
        full_hash = self.out_dat_handler.get_git_revision_hash()
        short_hash = self.out_dat_handler.get_git_revision_short_hash()

        with open(self.software_info_dir + "/git_hash.txt","w") as hash_file:
            hash_file.write("# revision hash \n")
            hash_file.write(full_hash + "\n")
            hash_file.write("\n")
            hash_file.write("# revision short hash \n")
            hash_file.write(short_hash + "\n")
            hash_file.write("\n")
            hash_file.close()
    #*****************************************************

    # NOW PUT EVERYHTING TOGETHER:

    #*****************************************************
    def run(self):
        # 1.) Load the modules:
        print("Load modules...")

        self.load_modules()

        print("...done!")
        print(" ")

        # 2.) Load tools:
        print("Load tools...")

        self.load_tools()

        print("...done!")
        print(" ")

        # 3.) Get the experimental data:
        print("Gather experimental data...")

        self.load_experimental_data()

        print("...done!")
        print(" ")

        # 4.) Set important directories:
        print("Set directories for output data...")

        self.handle_output_data_storage()

        print("...done!")
        print(" ")

        # 5.) Train the GAN:
        print("Train GAN...")

        t_fit = self.fit()

        t_fit_unit = "s"
        if t_fit >= 60.0 and t_fit < 3600.0:
            t_fit /= 60.0
            t_fit_unit = "min"
        elif t_fit >= 3600.0:
            t_fit /= 3600.0
            t_fit_unit = "h"

        print("...done! Finished GAN training in: ~" + str(round(t_fit,3)) + " " + t_fit_unit)
        print(" ")

        # 6.) Run post analysis:
        print("Collect and visualize results...")

        if self.snapshot_epoch > 0 and self.store_models_only == False:
              self.snaphsot_analysis()

        self.post_training_analysis(self.n_final_analysis_samples)

        print("...done!")
        print(" ")


        # 7.) Optional: Write out conda environment, config files, etc.
        if self.config_dir is not None:
            print("Store configuration file(s)...")

            self.store_configurations()

            print("...done!")
            print(" ")

        if self.software_info_dir is not None:
            print("Store software specific information...")

            self.store_software_info()

            print("...done!")
            print(" ")

        print("You are all done! Have a great day!")
        print(" ")
    #*****************************************************








