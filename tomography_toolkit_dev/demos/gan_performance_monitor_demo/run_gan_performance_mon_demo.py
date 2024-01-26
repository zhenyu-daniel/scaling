import torch
from torch import nn
import numpy as np
import time
import matplotlib.pyplot as plt
from tomography_toolkit_dev.utils.helper_scripts.torch_mlp_architecture import MLP_Architecture

import tomography_toolkit_dev.generator_module as generators
import tomography_toolkit_dev.discriminator_module as discriminators
from tomography_toolkit_dev.utils.data_science_mon.torch_gan_performance_monitor import GAN_Performance_Monitor
from tomography_toolkit_dev.utils.data_science_mon.torch_gan_performance_plotter import GAN_Performance_Plotter
from tomography_toolkit_dev.utils.helper_scripts.output_data_handler import Output_Data_Handler

"""
Demo to run the GAN performance monitoring and GAN architecture tool. Here, we are training a GAN to fit a 2D data set (x,y), where:
(x/a)^2 + (y/b)^2 = 1

The GAN does not fit the data directly, i.e. the generator predicts the ellipse parameters a and b which are then (via a data pipeline)
translated to a (x,y) pair.

The performance monitoring tool helps to track the losses, accuracy and residual information during the training.

All results produced by this script are stored as .png files inside a folder called 'exampleGAN'.

Important Note: The GAN model here is NOT tuned. So the results obtained here are not perfect.
"""

# Define a fake theory class, that produces ellipse data, based on the provided set of parameters
#-------------------------------------------------------------------------------------------------
class ellipse_theory(object):
     
    # Load ellipse parameters: 
    #************************
    def __init__(self,config,devices='cpu'):
         parmin = config['parmin'] if 'parmin' in config else [0.0,0.0]
         parmax = config['parmax'] if 'parmax' in config else [1.0,1.0]
         
         self.parmin = torch.as_tensor(parmin,device=devices)
         self.parmax = torch.as_tensor(parmax,device=devices)

         self.devices = devices
    #************************

    # Scale parameters:
    #************************
    def scale_params(self,params):
        scaled_params = (self.parmax - self.parmin) * params + self.parmin
        return scaled_params.to(self.devices)
    #************************
    
    # Generate the data:
    #************************
    def gen_events(self,params,n_events):
        if n_events <= 1:
           t = 2.0*np.pi * torch.rand(size=(params.size()[0],1))
        else:
           t = 2.0*np.pi * torch.rand(size=(n_events,1))


        dat = torch.cat([
            torch.cos(t),
            torch.sin(t)
        ],dim=1).to(devices)

        gen_data = params * dat
        return gen_data, torch.zeros((1,),device=devices,requires_grad=True), torch.zeros((1,),device=devices,requires_grad=True)
    #************************
    
    # Define forward path:
    #************************
    def forward(self,params):
         scaled_params = self.scale_params(params) 
         return self.gen_events(scaled_params,1)
    #************************

    # Define apply path:
    #************************
    def apply(self,params):
         scaled_params = self.scale_params(params) 
         return self.gen_events(scaled_params,1)
    #************************
#-------------------------------------------------------------------------------------------------

print("  ")
print("***************************************")
print("*                                     *")
print("*   GAN Performance Monitoring Demo   *")
print("*                                     *")
print("***************************************")
print("  ")

# Set the devices:
devices = 'cpu'

# Basic settings:
#///////////////////////////////////////////////////
# Number of events for the training data:
n_evts = 10000
true_parameters = np.array([5.0,2.0]) #--> True ellipse parameters

# Number of training epochs:
n_epochs = 5000
# Batch size:
batch_size = 512
# Print out some basic results, every mon_epoch:
mon_epoch = 1000
# When to read out loss and accuracy values: (i.e. the loss / accuracy / etc. will be updated every read_out_loss_epoch)
read_out_loss_epoch = 20
# Define if and when to take snapshots during the training:
snapshot_epoch = 1000 #--> Set this to zero if you are not interested in taking snapshots

# Mean of generator noise:
gen_noise_mean = 0.0
# Std. Dev. of generator noise:
gen_noise_std = 1.0

# Generator settings:
# Architecture:
num_gen_layers = 4
generator_cfg = {
    'num_layers': num_gen_layers,
    'num_nodes': [70] * num_gen_layers,
    'activations': ['LeakyReLU'] * num_gen_layers,
    'input_size': 2,
    'output_size': 2,
    'learning_rate': 1e-4
}
                                                  
# Discriminator settings:
# Architecture:
num_disc_layers = 3
discriminator_cfg = {
    'num_layers': num_disc_layers,
    'num_nodes': [50] * num_disc_layers,
    'activations': ['LeakyReLU'] * num_disc_layers,
    'input_size': 2,
    'output_size': 1,
    'learning_rate': 1e-4
}
#///////////////////////////////////////////////////


# Define toy data set:
#*********************************************************
print("Create training data set...")

# Load the 'theory' module:
theory = ellipse_theory(config={'parmin':[2.0,-1.0],'parmax':[8.0,5.0]},devices=devices)
# Generate trainig data
training_data, norm1, norm2 = theory.gen_events(torch.as_tensor(true_parameters,device=devices,dtype=torch.float32),n_evts)

n_features = training_data.size()[1]
observable_names = ['X','Y'] #--> Name of observables that we try to fit
n_parameters = true_parameters.shape[0]
par_names = ['a','b']

# Norms. The proxy GAN workflow requires those, but this workflow doesnt, so we simply set them to zero...
real_norms = norm1.repeat(batch_size), norm2.repeat(batch_size)

print("...done!")
print(" ")
#*********************************************************

# Define a data pipeline, like for the GAN worklfow.
# This pipeline translates the generator predictions (aka parameters) to meaningful data
# Here, the generator predicts the ellipse parameters a and b which are then translated to a (x,y) pair
def mock_data_pipeline(params):
       gen_events,_,_ = theory.apply(params)
       return gen_events

# Set up the discriminator and generator:
#*********************************************************
print("Set discriminator and generator...")

discriminator = discriminators.make("torch_mlp_discriminator_v0",config=discriminator_cfg).to(devices)

# Define module chain for the generator:
generator_module_chain = [theory,discriminator]

generator = generators.make("torch_mlp_generator_v0",config=generator_cfg,module_chain=generator_module_chain)

print("...done!")
print(" ")
#*********************************************************

# Get performance monitoring tools:
#*********************************************************
print("Load performance monitoring and plotting tools...")

# Load the performance monitor:
performance_monitor = GAN_Performance_Monitor(
    generator=generator, #--> Generator model
    discriminator=discriminator, #--> Discriminator model
    data_pipeline=mock_data_pipeline, #--> Data pipeline which usually translates generated parameters to physics events
    generator_noise_mean=gen_noise_mean, #--> Mean of the input noise
    generator_noise_std=gen_noise_std, #--> Std dev. of the input noise
    n_features=n_features, #--> Dimension of feature space (here: x and y --> 2)
    disc_loss_fn_str="mse", #--> Loss function string for discriminator, needed to assign the proper norm
    gen_loss_fn_str="mse", #--> Loss function string for generator, needed to assign the proper norm
    device=devices #--> Device where this script is running on
)

# Load the performance plotter:
performance_plotter = GAN_Performance_Plotter()

# Load the output data handler:
out_dat_handler = Output_Data_Handler()

print("...done!")
print(" ")
#*********************************************************

# Create a directory to store some results:
#*********************************************************
print("Create folder(s) to store training results...")

main_folder = out_dat_handler.create_output_data_folder("example_GAN")
training_folder = out_dat_handler.create_output_data_folder("training_results",main_folder)
snapshot_folder = None
if snapshot_epoch > 0:
   snapshot_folder = out_dat_handler.create_output_data_folder("training_snapshots",main_folder)

print("...done!")
print(" ")
#*********************************************************

# Run the training: 
#*********************************************************
print("Run GAN training...")

training_start_time = time.time()
#++++++++++++++++++++++++++++
for epoch in range(1,1+n_epochs):

    # We do not divide out data in batches here. We just sample from the real data
    # Where the sample size corresponds to the batch size 
    sample_idx = torch.randint(high=training_data.size()[0],size=(batch_size,))
    real_events = training_data[sample_idx].to(devices) 

    noise = torch.normal(mean=gen_noise_mean,std=gen_noise_std,size=(batch_size,training_data.size()[1]),device=devices)
    params = generator.generate(noise).to(devices)
    
    fake_events = mock_data_pipeline(params).to(devices)

    # Train the discriminator:
    d_losses = discriminator.train(real_events,fake_events)

    # Train the generator:
    g_losses = generator.train(noise,real_norms)

    # Monitor the losses, accuracy and residuals:
    # (We call the watch_() function for every batch / every update in the training data)
    performance_monitor.watch_losses_and_accuracy_per_batch(
        real_data=real_events,
        fake_data=fake_events,
        gen_loss=g_losses[0],
        disc_real_loss=d_losses[0],
        disc_fake_loss=d_losses[1])

    # We define a monitoring data set which has enough statistics so that we get a good representation of the residual information:
    current_params, current_gen_data, current_residuals = performance_monitor.watch_residual_information_per_batch(real_data=training_data,sample_size=5000)

    # Read out the monitored quantities so that we can calculate the weighted average of the loss, accuracy and residual info:
    # Important Note(s): 
    # (i) Not calling the collect_() function might cause inaccurate results
    # (ii) The collect_() function is called every 'read_out_loss_epoch'
    # (iii) Alternatively, one could write a 'classical' training loop, i.e. loop over epochs and over batched samples separately 
    # This way, one would call the collect_() function after each epoch and the watch_() function for each batch
    if epoch % read_out_loss_epoch == 0:
            # Losses and accuracy:
            performance_monitor.collect_losses_and_accuracy_per_epoch()
            # Residuals
            performance_monitor.collect_residual_information_per_epoch() 

    # Print out the losses and accuracies
    if epoch % mon_epoch == 0:
        # Print losses and accuracy
        print(" ")
        print("Epoch: " + str(epoch) + "/" + str(n_epochs))
        performance_monitor.print_losses_and_accuracy()
        print(" ")

    # Optional: Take a snapshot of the residuals to monitor the parameter evolution:
    if snapshot_epoch > 0 and epoch % snapshot_epoch == 0:
        # Take a snapshot of the current observables:
        mon_data = current_residuals + current_gen_data
        gen_data_plot_dict = performance_plotter.plot_observables_and_residuals(
            real_data=mon_data.cpu().numpy(),
            generated_data=current_gen_data.cpu().numpy(),
            residuals=current_residuals.cpu().numpy(),
            observable_names=observable_names)
        
        #++++++++++++++++++++ 
        for i in range(len(observable_names)):
            key_name = 'real_vs_gen_data_plots_' + observable_names[i]
            current_fig = gen_data_plot_dict[key_name][0]

            current_fig.suptitle('Comparing real and generated Data for Epoch: ' + str(epoch))
            current_fig.savefig(snapshot_folder + '/real_vs_gen_data_' + observable_names[i] + '_epoch' + str(epoch) + '.png')
            plt.close(current_fig)
        #++++++++++++++++++++ 

        # Take a snapshot of the parameter residuals:
        pred_param_dict = performance_plotter.plot_parameter_residuals(
            true_parameters=true_parameters,
            pred_parameters=current_params.cpu().numpy(),
            parameter_names=par_names)
        
        #++++++++++++++++++++ 
        for p in range(true_parameters.shape[0]):
            key_name = 'parameter_residual_plots_' + par_names[p]
            current_fig = pred_param_dict[key_name][0]

            current_fig.suptitle('Parameter Evolution Epoch: ' + str(epoch))
            current_fig.savefig(snapshot_folder + '/parameter_residual_' + par_names[p] + '_epoch' + str(epoch) + '.png')
            plt.close(current_fig)
        #++++++++++++++++++++ 

#++++++++++++++++++++++++++++
training_end_time = time.time()

training_time = (training_end_time - training_start_time)
unit = "s"

if training_time >= 60.0 and training_time < 3600.0:
     unit = "min"
     training_time /= 60.0
elif training_time >= 3600:
     unit = "h"
     training_time /= 3600.0

print("...done! Finished GAN training in: " + str(round(training_time,3)) + unit)
print(" ")
#*********************************************************

# Get the results:
#*********************************************************
print("Retrieve results...")

# Loss and accuracy:
loss_accuracy_dict = performance_monitor.read_out_losses_and_accuracy()
# Residual info during training
residual_info_dict = performance_monitor.read_out_residual_information()
# Calculate actual residuals:
params,generated_data, residuals = performance_monitor.predict_observables_and_residuals(real_data=training_data)

print("...done!")
print(" ")
#*********************************************************

# Plot everything:
#*********************************************************
print("Visualize performance...")

# Plot losses and accuracy as a function of the training epoch:
loss_plots = performance_plotter.plot_losses_and_accuracy(
    disc_real_loss=loss_accuracy_dict['disc_real_loss'], #--> Discriminator loss on real data
    disc_fake_loss=loss_accuracy_dict['disc_fake_loss'], #--> Discriminator loss on fake data
    gen_loss=loss_accuracy_dict['gen_loss'], #--> Generator loss,
    disc_real_accuracy=loss_accuracy_dict['disc_real_acc'], #--> Discriminator accuracy on real data
    disc_fake_accuracy=loss_accuracy_dict['disc_fake_acc'], #--> Discriminator accuracy on fake data
    x_label = 'Training Epoch per ' + str(read_out_loss_epoch) #--> Set the x-label properly
)

fig_loss = loss_plots['loss_and_accuracy_plots'][0]                                          
fig_loss.savefig(training_folder + '/GAN_loss_and_accurcay.png')
plt.close(fig_loss)
                       
# Plot the residual mean and std. dev as a function of training epoch:
residual_info_plots= performance_plotter.plot_residual_information(
    residual_mean=residual_info_dict['residual_mean'],
    residual_stddev=residual_info_dict['residual_stddev'],
    x_label = 'Training Epoch per ' + str(read_out_loss_epoch)
)

fig_res_info = residual_info_plots['residual_mean_and_stddev_plots'][0]
fig_res_info.savefig(training_folder + '/GAN_residual_info.png')
plt.close(fig_res_info)

# Now plot the actual residuals:
gen_data_plots = performance_plotter.plot_observables_and_residuals(
    real_data=training_data.cpu().numpy(), #--> Training data
    generated_data=generated_data.cpu().numpy(), #--> Generated / predicted data
    residuals=residuals.cpu().numpy(), #--> Residuals = Training data - Generated data
    observable_names=observable_names
)

#+++++++++++++++++++++++
for i in range(n_features):
    key_name = 'real_vs_gen_data_plots_' + observable_names[i]
    current_fig = gen_data_plots[key_name][0]

    save_name = training_folder + '/GAN_compare_' + observable_names[i] + '_prediction.png'
    current_fig.savefig(save_name)
    plt.close(current_fig)

    #+++++++++++++++++++++++
    for j in range(n_features):
        if j > i:
            key_name = 'data_correlation_plots_' + observable_names[j] + '_vs_' + observable_names[i]
            current_fig = gen_data_plots[key_name][0]
            
            save_name =  training_folder + '/GAN_predicted_' + observable_names[i] + observable_names[j] + '_correlation.png'
            current_fig.savefig(save_name)
            plt.close(current_fig)
    #+++++++++++++++++++++++ 

#+++++++++++++++++++++++

# Finally, check the residuals between the true parameters and the predicted ones:

param_residual_plots = performance_plotter.plot_parameter_residuals(
    true_parameters=true_parameters,
    pred_parameters=theory.scale_params(params).cpu().numpy(),
    parameter_names=par_names)

#+++++++++++++++++++++
for k in range(n_parameters):
    key_name = 'parameter_residual_plots_' + par_names[k]
    current_fig = param_residual_plots[key_name][0]

    save_name =  training_folder + '/GAN_parameter_' + par_names[k] + '_residual.png'
    current_fig.savefig(save_name)
    plt.close(current_fig)
#+++++++++++++++++++++

current_fig = param_residual_plots['parameter_comparison_plots'][0]
current_fig.savefig(training_folder + '/GAN_parameter_prediction.png')
plt.close(current_fig)

print("...done!")
print(" ")
print("Have a great day!")
print(" ")
#*********************************************************





