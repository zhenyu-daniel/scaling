import torch
import random
import functorch
from tomography_toolkit_dev.core.eventselection_core import EventSelection

class Rectangular_Cuts(EventSelection):
   """
   Class to apply a rectangular box cut on a specified data set. Let X be the input tensor, then a box cut on tensor element i is defined as:
   a_i <= X_i <= b_i where a_i,b_i are user defined limits

   Input: Tensor with shape: N_events x N_features
   Output non-training mode: Tensor with shape N_acc x N_features, where N_acc <= N_events
   Output training mode: Tensor with shape: N_events x N_features

   This particular filter runs in two modes:

   i) Non-Training mode: A conditional tensor (i.e. containing boolean elements) is defined via:
   cond = a_i <= X_i <= b_i  which then leads to the output tensor: X_acc = X[cond]
   A disadvantage of this method is that it might have a negative impact on the GAN training, since an if-condition is not
   differentiable

   ii) Training mode: Instead of using a hard cut, like in i), we use a soft approximation of the Heaviside function H(x):
   https://en.wikipedia.org/wiki/Heaviside_step_function

   H(x) ~ 0.5 + 0.5 tanh(kx) = [1 + exp(-2kx)]^(-1) where k is a scaling factor
   """ 

   # Initialize:
   #**********************************************
   def __init__(self,config,devices="cpu"):
        self.devices = devices

        self.minimum_limits = config["rect_evt_filter_minimum"] if "rect_evt_filter_minimum" in config else []
        self.maximum_limits = config["rect_evt_filter_maximum"] if "rect_evt_filter_maximum" in config else []
        self.scale_t = config["rect_evt_filter_scale"] if "rect_evt_filter_scale" in config else 50.0
        self.shift_t = config["rect_evt_filter_shift"] if "rect_evt_filter_shift" in config else 0.0
        self.is_blank = config["evtsel_module_off"] if "evtsel_module_off" in config else False
        
        self.n_min_limits = len(self.minimum_limits)
        self.n_max_limits = len(self.maximum_limits)

        self.run_blank_condition = self.n_min_limits * self.n_max_limits #--> Check if conditions are set

        # In case we want to enforce the 'hard_cut' during training:
        self.disable_training_behavior = config["rect_evt_filter_force_hard_cut_in_training"] if "rect_evt_filter_force_hard_cut_in_training" in config else False
        
        assert self.n_max_limits == self.n_min_limits, "The number of minium and maximum boundaries needs to be equal"
   #**********************************************

   # This is the 'classical' implementation of
   # A rectangular box cut:
   #**********************************************
   def apply_hard_box_cut(self,data):
       hard_box_cut = torch.ones(size=(data.size()[0],),dtype=torch.bool,device=self.devices)

       #+++++++++++++++++++++++++++
       for k in range(self.n_max_limits):
          min_cond = data[:,k] >= self.minimum_limits[k]
          max_cond = data[:,k] <= self.maximum_limits[k]

          min_cond = min_cond.to(self.devices)
          max_cond = max_cond.to(self.devices)

          hard_box_cut *= min_cond * max_cond
       #+++++++++++++++++++++++++++ 

       return hard_box_cut.to(self.devices)
   #**********************************************
   
   # Now we formualte the 'soft' version of a box cut, by using the heaside approximation
   # Depending on the underlying optimization problem, the functions below may or may not be used
   # However, their continuity and differentiability helps to support the gradient flow in the GAN model
   # It should be noted though, that this approach might, depending on the underlying data set, cause an
   # over-abundance of zeros which may or may not have a negative impact on the training
   # Alternative approaches are currently under investigation. For now, we will use this approach which has
   # been tested under different conditions and turned out to work smoothly 
   # In order to counter a possible zero-abundance, a shift is introduced
   #**********************************************
   # Reject / shift values according to a lower bound threshold:
   def apply_low_soft_threshold_v1(self,data,value,scale,shift):
       dX = data - torch.as_tensor(value,device=self.devices)
       f = torch.sigmoid(scale*dX) * (1.0 - shift) + shift
       f.to(self.devices)
       return f

   #-----------------------
   
   # Reject / shift values according to an upper bound threshold:
   def apply_high_soft_threshold_v1(self,data,value,scale,shift):
        dX = torch.as_tensor(value,device=self.devices) - data
        f = torch.sigmoid(scale*dX) * (1.0 - shift) + shift
        f.to(self.devices)
        return f

   #-----------------------

   # Now we combine the two functions above to define a soft box cut
   # The output of this function is a condition that can be directly plugged in to the filter function below
   def apply_soft_box_cut_v1(self,data):
       f_min = self.apply_low_soft_threshold_v1(data,self.minimum_limits,self.scale_t,self.shift_t)
       f_max = self.apply_high_soft_threshold_v1(data,self.maximum_limits,self.scale_t,self.shift_t)

       f_prod = torch.mul(f_min,f_max).to(self.devices)
       soft_conditions = torch.prod(f_prod,dim=1).to(self.devices)
       if self.n_max_limits > 1:
         soft_conditions = soft_conditions.unsqueeze(-1)

       return soft_conditions.to(self.devices)
   #**********************************************

   # Apply the filter, based on the provided condition:
   #**********************************************
   def filter(self,data,is_training):
      if self.run_blank_condition == 0 or self.is_blank == True:#--> if no limits are provided, then nothing should be done
            return data

      if is_training == True:
         conditions = self.apply_soft_box_cut_v1(data)
         return torch.mul(data,conditions).to(self.devices)
      
      conditions = self.apply_hard_box_cut(data)
      return data[conditions]
   #**********************************************

   # Forward path: Used at training time
   # This is when we use the soft approximation of the heaviside function
   #**********************************************
   def forward(self,data):
       if self.disable_training_behavior == True:
          filtered_data = self.filter(data,is_training=False)
          
          # For the purpose of consistency, we must upsample this data, so that it has the exact same dimension as the real data:
          sample_idx = torch.randint(high=filtered_data.size()[0],size=(data.size()[0],),device=self.devices)
          return filtered_data[sample_idx]

       return self.filter(data,is_training=True)
   #**********************************************

   # Apply: Just apply this function outside the training loop
   # 'Everyhting goes' here, i.e. we do not need to worry about differentiability
   #**********************************************
   def apply(self,data):
       return self.filter(data,is_training=False)
   #**********************************************
       

   
