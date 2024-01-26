import torch
from torchmetrics import MeanMetric
import math
import numpy as np


class AE_Performance_Monitor(object):
    """
    Class that summarizes monitoring tools that shall help to characterize / diagnose a MC dropout AE training cycle. The following quantities are monitored:

    (i) The generator losse
    (ii) The learned dropout probability
    

    This class also allows to calculate the residuals for the observables (i.e. the quantities that we are trying to fit) and the parameters (i.e. the generator predictions)
    All results are returned as numpy arrays which may be stored or visualized via matplotlib.

    The quantities are tracked via the torchmetric package which allows to easily track a metric, implement weights or even define your own metric. 
    """
    
    # Initialize:
    #************************************
    def __init__(self,generator,data_pipeline,device='cpu'):
        self.generator = generator
        self.data_pipeline = data_pipeline
        self.device = device
        
        # Watch loss and dropout rate for each data batch:
        self.watched_generator_loss = MeanMetric().to(self.device)
        self.watched_dropout_rate = MeanMetric().to(self.device)
        
        # Collect losses and accuracy at each training epoch:
        self.collected_generator_loss = []
        self.collected_dropout_rate = []
    #************************************

    # LOSS AND DROPOUT RATE:

    # Watch loss and dropout rate during each batch:
    #************************************
    def watch_loss_and_dropout_rate_per_batch(self,real_data,gen_loss):
        norm = torch.as_tensor( (1.0 / float(real_data.size()[0]) ),device=self.device)

        # Loss:
        self.watched_generator_loss.update(value=gen_loss.detach(),weight=norm)
        
        # Dropout rate:
        mc_dropout_rate = self.generator.layers['Dropout0'].p_train
        self.watched_dropout_rate.update(mc_dropout_rate,weight=norm)
    #************************************

    # Collect losses and accuracy during one epoch:
    #************************************
    def collect_loss_and_dropout_rate_per_epoch(self):

        # Collect the loss first:
        self.collected_generator_loss.append(self.watched_generator_loss.compute().cpu().numpy().item(0))
    
        # Get the mc dropout rate:
        dropout_rate_log = self.watched_dropout_rate.compute().cpu().numpy().item(0)

        dropout_rate = 1.0 / (1.0 + math.exp(-dropout_rate_log))
        self.collected_dropout_rate.append(dropout_rate)

        # Reset watchers:
        self.watched_generator_loss.reset()
        self.watched_dropout_rate.reset()
    #************************************

    # Print out losses and accuracry, if needed:
    #************************************
    def print_loss_and_dropout_rate(self,rounding=4):
        if len(self.collected_generator_loss) > 0:
           print(">>> Loss: " + str(round(self.collected_generator_loss[-1],rounding)) + " <<<")
           print(">>> MC dropout rate: " + str(round(self.collected_dropout_rate[-1],rounding)) + " <<<")
        else:
            print(">>> Loss information currently not available. Still collecting data <<<")
    #************************************
    
    # Read out losses and accuracy during training:
    #************************************
    def read_out_loss_and_dropout_rate(self):

        # Collect info:
        generator_loss = np.array(self.collected_generator_loss)
        dropout_rate = np.array(self.collected_dropout_rate)

        # Clean up everything:
        self.collected_generator_loss = []
        self.collected_dropout_rate = []

        return {
            'gen_loss': generator_loss,
            'dropout_rate': dropout_rate
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

          params = self.generator.generate(monitoring_data)

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






