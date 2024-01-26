import os
this_file_path = os.path.dirname(__file__)

training_config = {
    "batch_size":512,
    "num_epochs":200000,
    "print_info_epoch": 10000, #--> When to print the loss and accuracy values (here: every 10000 epochs, a result is printed)
    "read_performance_epoch": 1000, #--> When to record performance metrics, such as loss and accuracy (here: every 1000 epochs, the loss and accuracy are recorded)
    "n_final_analysis_samples": 50000, #--> How many events to generate for the final analysis (i.e. residuals, pdfs, etc.)
    "snapshot_epoch": 0, #--> Set this to a value > 0, if you want to take a snaphot of the parameter and residual evolution (e.g. 10 would mean, that every 10 epochs a snapshot is taken)
    "n_snapshot_samples": 5000, #--> How many events to generate for the snapshot evaluation. This setting has only impact, if snapshot_epoch > 0
    "noise_mean":0.5,
    "noise_std": 0.2,
    "output_loc": os.getcwd(), #--> location where the results shall be stored
    "result_folder": "gan_training_results", #--> Name of the folder (created at output_loc) which contains the results
    "watch_gradients": False, #--> Set to true if you want to monitor the gradients during training
    "store_results_as_npy": False, #--> Set to True if you want to store all monitored quantities at .npy arrays
    "write_cfg_to_json": False, #--> Set to True if you wish to store the configurations as individual .json files
    "write_cfg_to_py": False, #--> Set to True if you wish to store the entire configuration as .py file
    "get_software_info": False, #--> Set to True if you wish to store software related information (e.g. conda environment, git hash,...)
}

generator_config={
    "num_layers": 4,
    "num_nodes": [128,128,128,128],
    "activation": ["LeakyRelu", "LeakyRelu", "LeakyRelu", "LeakyRelu"],
    "useBias"   : [True,  True,  False, True ],
    "batchNorms": [False, False, True,  False],
    "dropout_percents": [0., 0., 0., 0.],
    "applyCustomWeightsInit": True,
    "input_mean" : training_config["noise_mean"],
    "input_std"  : training_config["noise_std"],
    "input_size" : 6,
    "output_size" : 6,
    "learning_rate": 1e-5,
    "clipGradientMagn": False
}

discriminator_config = {
    "num_layers": 4,
    "num_nodes": [128,128,128,128],
    "activation": ["LeakyRelu", "LeakyRelu", "LeakyRelu", "LeakyRelu"],
    "useBias"   : [True,  True,  False, True ],
    "batchNorms": [False, False, True,  False],
    "applyCustomWeightsInit": True,
    "input_size" : 2,
    "output_size" : 1,
    "learning_rate": 1e-4,
    "clipGradientMagn": 1.0
}

theory_config = {
    "n_parameters":6,
    "parmin":[0.0, -1.0, 0.0, 0.0, -1.0, 0.0],
    "parmax":[3.0, 1.0, 5.0, 3.0, 1.0, 5.0],
    "n_events":2
}

module_names = {
    "generator": "torch_mlp_generator_v0",
    "discriminator": "torch_mlp_discriminator_v0",
    "theory": "torch_proxy_theory_v0",
    "data_parser": "numpy_data_parser",
    "experiment": "simple_det", #--> Simplified detector
    "event_selection": "rectangular_cuts" #--> Rectangualr box cut, i.e. a <= x <= b, where a,b are the limits
}

data_config = {
    "path": os.path.join(this_file_path,'../sample_data/events_data.pkl.npy'),
    "truth": [0.72916667, 0.25, 0.6, 0.36458333, 0.25, 0.8],
    "observable_names":["Sigma1","Sigma2"],
    "transpose_dim": [0,1], #--> This ensures that our output data is in the right format, because the proxy theory returns the format: N_features x N_events
    "use_exp_module": True, #--> Decide if the experimental module shall be used on the training data
    "use_evtsel_module": True #--> Decide if the event selection module shall be used on the training data
}

experimental_config = {
    "smearing_parameters": [0.08,0.15], #--> Smearing parameters. Length of this list must be the same as the dimension of the observables (here: 2)
    "correlation_parameters": [0.03,0.2], #--> Handle correlations. First element: Introduces constant part. Second element: Introduces random part
    "correlation_asymmetry": [-1.0,0.3], #--> Introduce additional asymmetry in correlation matrix
    "exp_module_off": False, #--> Decide if we want to turn the experimental module 'off'
    # 'turned off' here means that the input data without any alterations is returned
    "rect_evt_filter_minimum": [0.0,0.0], #--> Set lower limits for your observable (here: cross sections). Length must match observable dimension (here: 2)
    "rect_evt_filter_maximum": [1.0,1.0], #--> Set upper limits for your observable (here: cross sections). Length must match observable dimension (here: 2)
    "evtsel_module_off": False, #--> Decide if we want to turn the event selection module 'off'. 'turned off' here means that the input data without any alterations is returned
    "rect_evt_filter_force_hard_cut_in_training": False, #--> Set this to True, in case you want to enforce a hard_cut during training
    # "rect_evt_filter_shift": -1.0 #--> Uncomment this line to fine tune the soft cut approximation
}
