import torch
import numpy as np
import matplotlib.pyplot as plt
import time 
import copy
import argparse
import os
from mlp_generator import Generator


# BASIC SETTINGS
#////////////////////////////////////////////////
ana_parser = argparse.ArgumentParser(description='Mock Analysis for Sync Studies',prog='run_mock_analysis.py')
ana_parser.add_argument('--ensemble_size',type=int,default=30,metavar='N',help='Number of GANs in the ensemble')
ana_parser.add_argument('--ensembles_to_show',type=int,default=15,metavar='N',help='Number of GANs to show in dt plot')
ana_parser.add_argument('--num_epochs',type=int,default=20000,metavar='N',help='Number of training epochs per GAN')
ana_parser.add_argument('--snapshot_epoch',type=int,default=10,metavar='N',help='Nth epoch when snapshot was taken')
ana_parser.add_argument('--n_noise_samples',type=int,default=5000,metavar='N',help='Number of noise samples')
ana_parser.add_argument('--stop_epoch',type=int,default=20000,metavar='N',help='Max. epoch to look for')
ana_parser.add_argument('--t_scan',type=int,default=5,metavar='N',help='Frequency (s) to scan for updates')
ana_parser.add_argument('--rel_t_smear',type=float,default=0.0001,metavar='F',help='Additonal relative time smearing (%)')
ana_parser.add_argument('--use_t_smear',action='store_true',default=False,help='Include time smearing')
ana_parser.add_argument('--result_folder',type=str,default='mock_sync_ana_results',help='Name of folder where results are stored')
ana_parser.add_argument('--data_loc',type=str,default=os.getcwd(),metavar='Dir',help='Location where data is stored')

ana_args = ana_parser.parse_args()

ensemble_size = ana_args.ensemble_size #--> Number of GANs in the ensemble
ensembles_to_show = ana_args.ensembles_to_show #--> How many GANs to show in the dt plot. This helps to make the plot less messy
num_epochs = ana_args.num_epochs #--> Number or training epochs per GAN
snapshot_epoch = ana_args.snapshot_epoch #--> At which epoch a snapshot was taken (e.g. every 10th, or 100th, etc.)
use_t_smear = ana_args.use_t_smear #--> Decide if you want to use the t-distribution for each GAN, including the resolution
rel_t_smear = ana_args.rel_t_smear #--> Additonal time smear one may or may not apply 
n_samples = ana_args.n_noise_samples #--> Number of samples we wish to generate
noise_mean = 0.0 #--> Noise mean
noise_std = 1.0 #--> Noise sigma

t_scan = ana_args.t_scan #--> Check for new results every 't_scan' time units
stop_epoch = ana_args.stop_epoch #--> When to stop looking for new results. Ideally this value should be equal to 'num_epochs'
# However, for sake of testing, one might choose a smaller value, in order to run this code faster

result_folder = ana_args.result_folder #--> Folder in which all results will be stored
data_loc = ana_args.data_loc #--> Location where all the ensembles are stored

ensemble_names = [data_loc + '/gan_ensemble_sync_study'+str(i)+'_np4' for i in range(1,1+ensemble_size)] #--> Names of all GANs in the ensemble
model_dir = 'model_stages' #--> Folder where the weights / models are stored
true_params = np.array([0.72916667, 0.25, 0.6, 0.36458333, 0.25, 0.8]) #--> True parameters

# Configuration of the generator model, used for each GAN in the ensemble:
generator_config={
    'num_layers': 4, 
    'num_nodes': [128, 128, 128, 128], 
    'activation': ['LeakyRelu', 'LeakyRelu', 'LeakyRelu', 'LeakyRelu'], 
    'dropout_percents': [0.0, 0.0, 0.0, 0.0], 
    'input_size': 6, 
    'output_size': 6, 
    'learning_rate': 1e-05, 
    'weight_initialization': ['uniform', 'uniform', 'uniform', 'uniform', 'uniform'], 
    'bias_initialization': ['normal', 'normal', 'normal', 'normal', 'normal']
}

#////////////////////////////////////////////////

# Have an intro
print(" ")
print("************************************************")
print("*                                              *")
print("*   Mock Analysis for syncronization Studies   *")
print("*                                              *")
print("************************************************")
print(" ")

# Load generator base class and create result folder:

#**********************************
print("Set up generator base class and create result folder...")

generator_base = Generator(generator_config,[],"cpu")

if use_t_smear == False:
   result_folder += '_t_fix'
else:
    result_folder += '_t_smear'

if os.path.exists(result_folder) == False:
    os.mkdir(result_folder)

print("...done!")
print(" ")
#**********************************


# Determine the individual time distributions:
#**********************************
print("Determine average training time for each GAN within the ensemble...")

plt.rcParams.update({'font.size':20})
figt,axt = plt.subplots(figsize=(12,8))

# Loop through all GAN files and extract the state dictionary which was stored during training:
ensemble_time_collection = []
#++++++++++++++++++++++++
for k in range(ensemble_size):
    current_gan_models = ensemble_names[k] + '/' + model_dir
    
    t_raw_recorded = []
    #++++++++++++++++++++++++++++
    for epoch in range(1,1+num_epochs):
        epoch_str = str(epoch) + 'epochs'
        epoch_str = epoch_str.zfill(6 + len(str(num_epochs)))
        if epoch % snapshot_epoch == 0:
           # Here is where we load the state dictionary
           current_generator_dict = torch.load(current_gan_models+'/generator_'+epoch_str+'.pt',map_location=torch.device('cpu'))
           
           # The timestamp is stored inside the state dictionary. This way, we ensure that this information does not get lost
           # e.g. when moving the GAN files somehwere else...
           raw_time = current_generator_dict['timestamp']
           t_raw_recorded.append(raw_time)
    #++++++++++++++++++++++++++++

    # Correct each time for the absolute start time:
    t_recorded = [t_raw_recorded[j+1]-t_raw_recorded[j] for j in range(len(t_raw_recorded)-1)]
    ensemble_time_collection.append(np.expand_dims(np.array(t_recorded),axis=0))
    
    # Show the time distributions for only a few GANs (optional)
    if k < ensembles_to_show:
      current_label = 'GAN ' + str(k)
      axt.hist(t_recorded,100,histtype='step',linewidth=3.0,label=current_label)
#++++++++++++++++++++++++

axt.set_xlabel('Time per ' + str(snapshot_epoch) + ' Epochs [s]')
axt.set_ylabel('Entries')
axt.grid(True)
axt.legend()

figt.savefig(result_folder+'/avg_training_time.png')
plt.close(figt)

ensemble_times = np.concatenate(ensemble_time_collection,axis=0)
t_ensemble_mean = np.mean(ensemble_times,axis=1)
t_ensemble_std = np.std(ensemble_times,axis=1)

print("...done!")
print(" ")
#**********************************


# Now run a mock-analysis:
#**********************************
print("Run mock analysis...")

# Array to check when the training of each GAN has finished, i.e. when the maximum epoch is reached
training_finished = np.zeros(ensemble_size)

generator_ensemble = [] #--> All generator models that are avaialble after so and so many scans
accumulated_scan_time = 0 #--> Accumulated time during scanning
accumulated_ensemble_time = np.copy(t_ensemble_mean) #--> Accumulated time for ecach indiviudal GAN workflow to release a new model
trained_epochs = np.array([snapshot_epoch]*ensemble_size)

# Ensemble results:
param_residual_mean = []
param_residual_std = []

# Individual results:
indiv_param_residual_mean = []

# Accumulated scanning time:
training_time = []

while np.prod(training_finished) == 0.0:
    # Wait t_scan time units and then check for results:
    # time.sleep(t_scan) #--> Uncommend this to make the code a bit more 'realistic'
    accumulated_scan_time += t_scan 

    # Now go through all ensembles and load every model that is available after t_scan:
    #++++++++++++++++++++++++++++++
    for k in range(ensemble_size):
        current_t = accumulated_ensemble_time[k]

        if accumulated_scan_time > current_t:
            # Check for the available epochs:
            epoch_str = str(trained_epochs[k]) + 'epochs'
            epoch_str = epoch_str.zfill(6 + len(str(num_epochs)))
            
            # Load the generator dict:
            current_gan_models = ensemble_names[k] + '/' + model_dir
            current_generator_dict = torch.load(current_gan_models+'/generator_'+epoch_str+'.pt',map_location=torch.device('cpu'))
            # Remove the time stamp (not permanently, just here, so that torch does not complain)
            current_generator_dict.pop('timestamp')
            generator_base.load_state_dict(current_generator_dict) #--> Load the dictionary into the base network
            generator_ensemble.append(copy.deepcopy(generator_base)) #--> And make a copy
            

            if trained_epochs[k] == stop_epoch: #--> Once the last model (for one workflow) has been written to file, we define this workflow to be finished
                training_finished[k] = 1.0
            
            # Once we retreived the model, we advance in epoch / time
            trained_epochs[k] += snapshot_epoch
            if trained_epochs[k] > stop_epoch:
                trained_epochs[k] = stop_epoch
            
            # Decide if we advance in time via a constant value or a gaussian smeared value:
            if use_t_smear == False:
               accumulated_ensemble_time[k] += t_ensemble_mean[k]
            else:
               accumulated_ensemble_time[k] += ( np.random.normal(loc=t_ensemble_mean[k],scale=t_ensemble_std[k],size=(1)) * np.random.normal(loc=1.0,scale=rel_t_smear,size=(1)) )
    #++++++++++++++++++++++++++++++ 

    # Read out all models that have been collected so far:
    n_collected_models = len(generator_ensemble)
    if n_collected_models >= ensemble_size:
       
        # Calculate the noise:
        noise = torch.normal(mean=noise_mean,std=noise_std,size=(n_samples,6))

        # Pass the noise through every generator:
        params_list = []
        individual_mean = -100.0 * np.ones((ensemble_size,6))
        #+++++++++++++++++++++++++
        for g in range(ensemble_size):
            gen = generator_ensemble[g]
            
            # Ensemble:
            pred_params = gen.forward(noise)
            p_residuals = true_params - pred_params.detach().numpy()
            params_list.append(p_residuals)

            # Individual:
            if g < ensemble_size:
                t = np.mean(p_residuals,axis=0)
                individual_mean[g,:] = t
        #+++++++++++++++++++++++++

        # Determine the mean and sigma:
        params = np.concatenate(params_list,axis=0)
        params_mean = np.mean(params,axis=0)
        params_std = np.std(params,axis=0)

        param_residual_mean.append(np.expand_dims(params_mean,axis=0))
        param_residual_std.append(np.expand_dims(params_std,axis=0))
        training_time.append(accumulated_scan_time)
        indiv_param_residual_mean.append(np.expand_dims(individual_mean,axis=0))

        # Once we have an entire ensemble of N GANs, we reset the list and restart the collection 
        if n_collected_models >= ensemble_size:
            generator_ensemble = generator_ensemble[ensemble_size:]

print("...done!")
print(" ")
#**********************************

            
# Once the 'training loop' has finished, we can evaluate the collected data and produce some pretty plots:
#**********************************
print("Visualize results...")

collected_param_mean = np.concatenate(param_residual_mean,axis=0)
collected_param_std = np.concatenate(param_residual_std,axis=0)

individual_results = np.concatenate(indiv_param_residual_mean,axis=0)

figf,axf = plt.subplots(6,1,sharex=True,figsize=(15,8))

#+++++++++++++++++++++
for p in range(6):
    axf[p].errorbar(x=training_time,y=collected_param_mean[:,p],yerr=collected_param_std[:,p],fmt='ko',linewidth=3.0,capsize=5,label='GAN Ensemble')
    axf[p].grid(True)
    axf[p].legend(fontsize=10)
    
    r = p + 1
    current_y_label = 'R' + str(r)
    axf[p].set_ylabel(current_y_label)
    axf[p].set_ylim(-0.5,0.5)

    # Add the individual results:
    i_res = individual_results[:,:,p]
    
    #++++++++++++++++++++++++++++++++++++++++
    for g in range(ensemble_size):
        acc = i_res[:,g] > -100.0
        axf[p].plot(np.array(training_time)[acc],i_res[:,g][acc],linewidth=3.0) 
    #++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++
axf[5].set_xlabel('Accumulated ' + r'$t_{scan}$' + ' [s]')

figf.savefig(result_folder+'/param_residuals_vs_t_scan.png')
plt.close(figf)

print("...done!")
print(" ")
print("All done! Have a great day!")
print(" ")
#**********************************





