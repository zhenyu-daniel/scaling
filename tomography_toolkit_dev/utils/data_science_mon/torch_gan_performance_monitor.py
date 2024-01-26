import torch
from torchmetrics import MeanMetric
from torchmetrics.classification import Accuracy
import numpy as np


class GAN_Performance_Monitor(object):
    """
    Class that summarizes monitoring tools that shall help to characterize / diagnose a GAN training cycle. The following quantities are monitored:

    (i) The generator, discriminator (real & fake) losses
    (ii) The discriminator accuracy on real / fake data
    (iii) The mean and std. dev. of the residuals: real data - fake data (Note: In order to simplify / generalize the computation, we only look at the overall mean and std. dev.)

    This class also allows to calculate the residuals for the observables (i.e. the quantities that we are trying to fit) and the parameters (i.e. the generator predictions)
    All results are returned as numpy arrays which may be stored or visualized via matplotlib.

    The quantities are tracked via the torchmetric package which allows to easily track a metric, implement weights or even define your own metric. 
    """
    
    # Initialize:
    #************************************
    def __init__(self,generator,discriminator,data_pipeline,generator_noise_mean,generator_noise_std,n_features,disc_loss_fn_str,gen_loss_fn_str,accuracy_threshold=0.5,device='cpu'):
        self.generator = generator
        self.discriminator = discriminator
        self.data_pipeline = data_pipeline
        self.generator_noise_mean = generator_noise_mean
        self.generator_noise_std = generator_noise_std
        self.n_features = n_features
        self.device = device
        
        # Get the norm for each loss:
        self.disc_loss_norm = 1.0 / self.get_loss_norm(disc_loss_fn_str)
        self.gen_loss_norm = 1.0 / self.get_loss_norm(gen_loss_fn_str)
        
        # Watch loss and accuracy for each data batch:
        self.watched_generator_loss = MeanMetric().to(self.device)
        self.watched_real_discriminator_loss = MeanMetric().to(self.device)
        self.watched_fake_discriminator_loss = MeanMetric().to(self.device)
        self.watched_real_accuracy = Accuracy(task='binary',threshold=accuracy_threshold).to(self.device)
        self.watched_fake_accuracy = Accuracy(task='binary',threshold=accuracy_threshold).to(self.device)

        # Collect losses and accuracy at each training epoch:
        self.collected_generator_loss = []
        self.collected_real_discriminator_loss = []
        self.collected_fake_discriminator_loss = []
        self.collected_real_accuracy = []
        self.collected_fake_accuracy = []

        # Watch mean and std deviation of the residuals for each data batch:
        self.watched_residual_mean = MeanMetric().to(self.device)
        self.watched_residual_std = MeanMetric().to(self.device)

        # Collect mean and std dev at each training epoch:
        self.collected_residual_mean = []
        self.collected_residual_stddev = []
    #************************************

    # LOSSES AND ACCURACY:

    # Get the norm for a specific loss function:
    #************************************
    def get_loss_norm(self,loss_fn_str):
        loss_norm = 1.0

        if loss_fn_str.lower() == "bce":
            loss_norm = -np.log(0.5)
        if loss_fn_str.lower() == "mse":
            loss_norm = 0.25
        if loss_fn_str.lower() == "mae" or loss_fn_str.lower() == "l1":
            loss_norm = 0.5

        return loss_norm
    #************************************
  
    # Watch loss and accuracy during each batch:
    #************************************
    def watch_losses_and_accuracy_per_batch(self,real_data,fake_data,gen_loss,disc_real_loss,disc_fake_loss):
        norm = torch.as_tensor( (1.0 / float(real_data.size()[0]) ),device=self.device)

        # Losses:
        self.watched_generator_loss.update(value=gen_loss.detach() * self.gen_loss_norm,weight=norm)
        self.watched_real_discriminator_loss.update(value=disc_real_loss.detach() * self.disc_loss_norm,weight=norm)
        self.watched_fake_discriminator_loss.update(value=disc_fake_loss.detach() * self.disc_loss_norm,weight=norm)
        
        # Accuracy:
        disc_real = self.discriminator(real_data)
        pos_labels = torch.ones(disc_real.size(),device=self.device,dtype=torch.int32)
        self.watched_real_accuracy.update(preds=disc_real,target=pos_labels)

        disc_fake = self.discriminator(fake_data)
        neg_labels = torch.zeros(disc_fake.size(),device=self.device,dtype=torch.int32)
        self.watched_fake_accuracy.update(preds=disc_fake,target=neg_labels)
    #************************************

    # Collect losses and accuracy during one epoch:
    #************************************
    def collect_losses_and_accuracy_per_epoch(self):

        # Collect losses first:
        self.collected_generator_loss.append(self.watched_generator_loss.compute().cpu().numpy().item(0))
        self.collected_real_discriminator_loss.append(self.watched_real_discriminator_loss.compute().cpu().numpy().item(0))
        self.collected_fake_discriminator_loss.append(self.watched_fake_discriminator_loss.compute().cpu().numpy().item(0))

        # Get the accuracy:
        self.collected_real_accuracy.append(self.watched_real_accuracy.compute().cpu().numpy().item(0))
        self.collected_fake_accuracy.append(self.watched_fake_accuracy.compute().cpu().numpy().item(0))

        # Reset watchers:
        self.watched_generator_loss.reset()
        self.watched_real_discriminator_loss.reset()
        self.watched_fake_discriminator_loss.reset()
        self.watched_real_accuracy.reset()
        self.watched_fake_accuracy.reset()
    #************************************

    # Print out losses and accuracry, if needed:
    #************************************
    def print_losses_and_accuracy(self,rounding=4):
        if len(self.collected_generator_loss) > 0:
           print(">>> G_Loss: " + str(round(self.collected_generator_loss[-1],rounding)) + " <<<")
           print(">>> D_Loss(real): " + str(round(self.collected_real_discriminator_loss[-1],rounding)) + " <<<")
           print(">>> D_Loss(fake): " + str(round(self.collected_fake_discriminator_loss[-1],rounding)) + " <<<")
           print(">>> Accuracy(real): " + str(round(self.collected_real_accuracy[-1],rounding)) + " <<<")
           print(">>> Accuracy(fake): " + str(round(self.collected_fake_accuracy[-1],rounding)) + " <<<")

        else:
            print(">>> Loss information currently not available. Still collecting data <<<")
    #************************************
    
    # Read out losses and accuracy during training:
    #************************************
    def read_out_losses_and_accuracy(self):

        # Collect info:
        generator_loss = np.array(self.collected_generator_loss)
        discriminator_real_loss = np.array(self.collected_real_discriminator_loss)
        discriminator_fake_loss = np.array(self.collected_fake_discriminator_loss)

        discriminator_real_accuracy = np.array(self.collected_real_accuracy)
        discriminator_fake_accuracy = np.array(self.collected_fake_accuracy)

        # Clean up everything:
        self.collected_generator_loss = []
        self.collected_real_discriminator_loss = []
        self.collected_fake_discriminator_loss = []

        self.collected_real_accuracy = []
        self.collected_fake_accuracy = []

        return {
            'gen_loss': generator_loss,
            'disc_real_loss': discriminator_real_loss,
            'disc_fake_loss': discriminator_fake_loss,
            'disc_real_acc': discriminator_real_accuracy,
            'disc_fake_acc': discriminator_fake_accuracy
        }
    #************************************

    # RESIDUALS:

    # Predict observables via the generator:
    #************************************
    def predict_observables_and_residuals(self,real_data,n_samples=0.0):
       with torch.no_grad(): 
          monitoring_data = real_data
        
          # If a number of samples is specified, we may up / downsample the data we want to analyze:
          if n_samples > 0.0:
            mon_sample_idx = torch.randint(real_data.size()[0],size=(n_samples,),device=self.device)
            monitoring_data = real_data[mon_sample_idx].to(self.device)

          n_gen_events = int(monitoring_data.size()[0])
          noise = torch.normal(mean=self.generator_noise_mean,std=self.generator_noise_std,size=(n_gen_events,self.n_features),device=self.device)
          params = self.generator.generate(noise)

          # Now we use the data pipeline to generate data:
          out_data = self.data_pipeline(params).to(self.device)
          
          # Depending on the modules involved, the number of events in the generated data might be more / less after passing through all modules
          # than the corresponding number of events in the real data. This is why we simply randomly upsample / down sample the
          # number of generated events:
          sample_idx = torch.randint(out_data.size()[0],size=(monitoring_data.size()[0],),device=self.device)
          generated_data = out_data[sample_idx]
          
          residuals = monitoring_data - generated_data
          return params, generated_data, residuals
    #************************************
    
    # Watch residual mean / std for each data batch:
    # This function returns the generated parameters and observable residuals
    #************************************  
    def watch_residual_information_per_batch(self,real_data,sample_size=0.0): 
        norm = torch.as_tensor( (1.0 / float(real_data.size()[0])),device=self.device )

        params, generated_data, residuals = self.predict_observables_and_residuals(real_data,n_samples=sample_size)
        
        # We are looking at the mean / std only, because we would like to keep the number of monitired quantities low
        mean = torch.mean(residuals)
        stddev = torch.std(residuals)

        self.watched_residual_mean.update(value=mean,weight=norm)
        self.watched_residual_std.update(value=stddev,weight=norm)

        return params, generated_data, residuals
    #************************************
    
    # Collect residual mean / std for each training epoch:
    #************************************
    def collect_residual_information_per_epoch(self):
        self.collected_residual_mean.append(self.watched_residual_mean.compute().cpu().numpy().item(0))
        self.collected_residual_stddev.append(self.watched_residual_std.compute().cpu().numpy().item(0))

        # Reset everything:
        self.watched_residual_mean.reset()
        self.watched_residual_std.reset()
    #************************************
    
    # Read out residual information during training:
    #************************************
    def read_out_residual_information(self):
        
        # Collect info:
        residual_mean = np.array(self.collected_residual_mean)
        residual_stddev = np.array(self.collected_residual_stddev)
        
        # Clean up:
        self.collected_residual_mean = []
        self.collected_residual_stddev = []

        return {
            'residual_mean': residual_mean,
            'residual_stddev': residual_stddev
        }
    #************************************

    # MODEL STORAGE:

    # Store the models, if wanted:
    #************************************
    def store_models(self,current_training_epoch,disc_store_path,gen_store_path):
        # Discriminator:
        torch.save({
                 'epoch': current_training_epoch,
                 'discriminator_state': self.discriminator.state_dict(),
                 'discriminator_real_loss': self.collected_real_discriminator_loss[-1],
                 'discriminator_fake_loss': self.collected_fake_discriminator_loss[-1]
               },
        disc_store_path)

        # Generator:
        torch.save({
                 'epoch': current_training_epoch,
                 'generator_state': self.generator.state_dict(),
                 'generator_loss': self.collected_generator_loss[-1]
               },
        gen_store_path)
    #************************************






