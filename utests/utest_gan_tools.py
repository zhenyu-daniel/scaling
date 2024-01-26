import unittest
import torch
from torch import nn
import numpy as np
import tomography_toolkit_dev.generator_module as generators
import tomography_toolkit_dev.discriminator_module as discriminators
from tomography_toolkit_dev.utils.data_science_mon.torch_gradient_monitor import Weight_Gradient_Monitor
from tomography_toolkit_dev.utils.data_science_mon.torch_gan_performance_monitor import GAN_Performance_Monitor

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
        ],dim=1).to(self.devices)

        gen_data = params * dat
        return gen_data, torch.zeros((1,),device=self.devices,requires_grad=True), torch.zeros((1,),device=self.devices,requires_grad=True)
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


class TestGANTools(unittest.TestCase):
 
    # Run a mini GAN training where we test various performance monitoring tools:
    #************************************
    # Define a little helper function, do quickly evaluate a dictionary:
    def eval_dict(self,dict,n_dim0,n_dim1,check_nan=True):
        pass_test = 1.0
        #+++++++++++++++++++++++++
        for key in dict:
            current_array = dict[key]
     
            # Ensure that the first dimension has the expected number of entries:
            if n_dim0 > 0:
                if current_array.shape[0] == n_dim0:
                    pass_test *= 1.0
                else:
                    print(">>> WARNING for " + key + ": dimension 0 not matched <<< ")
                    pass_test *= 0.0
            
            # Ensure that the second dimension has the expected number of entries:
            if n_dim1 > 0:
                if current_array.shape[1] == n_dim1:
                    pass_test *= 1.0
                else:
                    print(">>> WARNING for " + key + ": dimension 1 not matched <<<")
                    pass_test *= 0.0

            # Finally, check that there are non NaN values:
            if check_nan:
                if ~np.isnan(np.sum(current_array)):
                    pass_test *= 1.0
                else:
                    print(">>> WARNING: " + key + ": detected NaN <<<")
                    pass_test *= 0.0
        #+++++++++++++++++++++++++
        
        return pass_test
    
    #----------------------------------------------------------

    # Run the test:
    def test_GAN_performance_monitoring(self):
        devices = 'cpu'
        batch_size = 64
        n_epochs = 200
   
        # Define true parameters and 'theory' module:
        n_evts = 10000
        true_parameters = np.array([5.0,2.0])

        theory = ellipse_theory(config={'parmin':[2.0,-1.0],'parmax':[8.0,5.0]},devices=devices)
        # Generate trainig data
        training_data, norm1, norm2 = theory.gen_events(torch.as_tensor(true_parameters,device=devices,dtype=torch.float32),n_evts)
        n_features = training_data.size()[1]
        
        # Norms. The proxy GAN workflow requires those, but this workflow doesnt, so we simply set them to zero...
        real_norms = norm1.repeat(batch_size), norm2.repeat(batch_size)

        # Define a data pipeline, like for the GAN worklfow.
        # This pipeline translates the generator predictions (aka parameters) to meaningful data
        # Here, the generator predicts the ellipse parameters a and b which are then translated to a (x,y) pair
        def mock_data_pipeline(params):
            gen_events,_,_ = theory.apply(params)
            return gen_events

        # Get the discriminator:
        num_disc_layers = 3
        discriminator_cfg = {
            'num_layers': num_disc_layers,
            'num_nodes': [30] * num_disc_layers,
            'activations': ['LeakyReLU'] * num_disc_layers,
            'input_size': 2,
            'output_size': 1,
            'learning_rate': 1e-4
        }
        discriminator = discriminators.make("torch_mlp_discriminator_v0",config=discriminator_cfg).to(devices)

        # Define module chain for the generator:
        generator_module_chain = [theory,discriminator]

        # And now get the generator:
        num_gen_layers = 4
        generator_cfg = {
            'num_layers': num_gen_layers,
            'num_nodes': [50] * num_gen_layers,
            'activations': ['LeakyReLU'] * num_gen_layers,
            'input_size': 2,
            'output_size': 2,
            'learning_rate': 1e-4
        }

        generator = generators.make("torch_mlp_generator_v0",config=generator_cfg,module_chain=generator_module_chain).to(devices)

        # Load the performance monitor:
        performance_monitor = GAN_Performance_Monitor(
           generator=generator, #--> Generator model
           discriminator=discriminator, #--> Discriminator model
           data_pipeline=mock_data_pipeline, #--> Data pipeline which usually translates generated parameters to physics events
           generator_noise_mean=0.0, #--> Mean of the input noise
           generator_noise_std=1.0, #--> Std dev. of the input noise
           n_features=n_features, #--> Dimension of feature space (here: x and y --> 2)
           disc_loss_fn_str="mse", #--> Loss function string for discriminator, needed to assign the proper norm
           gen_loss_fn_str="mse", #--> Loss function string for generator, needed to assign the proper norm
           device=devices #--> Device where this script is running on
        )

        # Load gradient monitors:
        disc_grad_mon = Weight_Gradient_Monitor(discriminator)
        gen_grad_mon = Weight_Gradient_Monitor(generator)

        print(" ")
        print("Run GAN training: ")
        print(" ")
        #++++++++++++++++++++++++++++
        for epoch in range(1,1+n_epochs):

            if epoch % 50 == 0:
                print(" ")
                print("Epoch: " + str(epoch) + "/" + str(n_epochs))
                performance_monitor.print_losses_and_accuracy()
                print(" ")

            sample_idx = torch.randint(high=training_data.size()[0],size=(batch_size,))
            real_events = training_data[sample_idx].to(devices) 

            noise = torch.normal(mean=0.0,std=1.0,size=(batch_size,2))
            params = generator.generate(noise)
            fake_events = mock_data_pipeline(params)

            # Train the discriminator:
            d_losses = discriminator.train(real_events,fake_events)
            
            # Monitor the discriminator gradients:
            disc_grad_mon.watch_gradients_per_batch(batch_size)

            # Train the generator:
            g_losses = generator.train(noise,real_norms)

            # Monitor the generator gradients:
            gen_grad_mon.watch_gradients_per_batch(batch_size)

            # Monitor the losses, accuracy and residuals:
            performance_monitor.watch_losses_and_accuracy_per_batch(real_events,fake_events,g_losses[0],d_losses[0],d_losses[1])
            performance_monitor.watch_residual_information_per_batch(real_events)

            # Read out the gradients every batch size, so that we can calculate the average mean:
            # Similarly for the losses, accuracy and residuals:
            if epoch % batch_size == 0:
                # Gradients:
                disc_grad_mon.collect_gradients_per_epoch()
                gen_grad_mon.collect_gradients_per_epoch()

                # Losses and accuracy:
                performance_monitor.collect_losses_and_accuracy_per_epoch()
                # Residuals
                performance_monitor.collect_residual_information_per_epoch() 
        #++++++++++++++++++++++++++++

        # Evaluate the results:

        # i) Gradients:
        disc_gradient_dict = disc_grad_mon.read_out_gradients()
        gen_gradient_dict = gen_grad_mon.read_out_gradients()
        # Each dictionary contains three arrays: average, minimum and maximum gradient

        # We monitored the gradients every 'batch_size' epoch, thus we expect the first dimension 
        # of each array to be: int(epoch / batch_size)
        # The second dimentsion of each array has to be exactly equal the number of layers in each model (plus the output layer)
        # (Because we check the gradient for each layer)
        # On top of that, we check that there are no NaN values (this might happen in general, but this toy data is simple enough that we do not expect any NaNs)
        expected_length = int(n_epochs / batch_size)

        # Check discrimiantor gradients:
        pass_disc_grad_test = self.eval_dict(disc_gradient_dict,expected_length,num_disc_layers+1)
        # Check generator gradients:
        pass_gen_grad_test = self.eval_dict(gen_gradient_dict,expected_length,num_gen_layers+1)

        # ii) Losses and accuracy:
        loss_dict = performance_monitor.read_out_losses_and_accuracy()

        # Check that the dimensionality is correct --> Expect that each array has the length: int(epoch / batch_size)
        pass_loss_acc_test = self.eval_dict(loss_dict,expected_length,-1)   

        # iii) Residual information:
        residual_info_dict = performance_monitor.read_out_residual_information()
        
        # The residual information is stored as averaged mean and stdev, so the above dictionary should contain two arrays
        # with length int(epoch/batch_size)
        pass_residual_info_test = self.eval_dict(residual_info_dict,expected_length,-1) 

        self.assertEqual(pass_disc_grad_test * pass_gen_grad_test * pass_loss_acc_test * pass_residual_info_test,1.0)
    #************************************




if __name__ == "__main__":
    # You can run this file using:   python unittest_example.py
    unittest.main()



