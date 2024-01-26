import torch
import functorch
import numpy as np
from tomography_toolkit_dev.core.expdata_core import ExpData

class Numpy_Data_Parser(ExpData):
   """
   Class to load (and parse) experimental data that is stored in a single .npy file

   Input: .npy file which may or may not contain multiple array objects. Only rule: The first array has to be the one containing the experimental
   data

   Output: array object with the following structure:
   (i) The first element is the exp data after the pre-processing (e.g. after passing through some models)
   (ii) The second element is the experimental data after reshaping (i.e. getting the dimensions right)
   (iii) Remaining elements from the input file
   """

   # Initialize:
   #******************************************
   def __init__(self,config,module_chain,devices="cpu"):
       self.devices = devices
       self.module_list = module_chain
       self.data_path = config["path"] if "path" in config else ""
       self.transpose_dim = config["transpose_dim"] if "transpose_dim" in config else None
       self.use_exp_module = config["use_exp_module"] if "use_exp_module" in config else False
       self.use_evtsel_module = config["use_evtsel_module"] if "use_evtsel_module" in config else False
       self.data_type = config["data_type"] if "data_type" in config else torch.float32
       self.use_full_data_for_training = config['use_full_data_for_training'] if 'use_full_data_for_training' in config else False

       assert self.data_path != "", f"Exp Data Module: No path for the data set provided"

       if self.transpose_dim is not None:
          len_transpose_dim = len(self.transpose_dim)
          assert len_transpose_dim == 2, f"Exp Data Module: Provided dimension for transpose operation (={len_transpose_dim}) is larger than 2"

       if self.use_full_data_for_training == True:
          print(" ")
          print(">>> Exp Data Module: Full Data sample is used for training <<<")
          print(" ")
   #******************************************

   # Load the data:
   #******************************************
   def load_data(self):
       if isinstance(self.data_path,list):
           return [np.load(p,allow_pickle=True) for p in self.data_path]
       elif '.npy' in self.data_path:
           return np.load(self.data_path,allow_pickle=True)
       else:
           return np.load(self.data_path + '.npy',allow_pickle=True)
   #******************************************

   # Now pass the data throug the indifivudal moduls:
   # (if requested)
   #******************************************
   # Pass data through a single module and check if training is active or not:
   def pass_data_through_single_modules(self,data,module,is_training,not_active):
       if not_active == False:
          return data

       if is_training:
           if 'disable_training_behavior' in dir(module):
               if module.disable_training_behavior == True:
                  return module.apply(data)

           return module.forward(data)
       else:
           return module.apply(data)

    #----------------------------------

   # Pipeline to create training data:
   def pass_data_through_modules(self,in_data,is_training,extra_observable_names):
       out_data = None

       # Check if the provided data consists of one ore multiple elements:
       is_object = False
       if isinstance(in_data,tuple) or isinstance(in_data,list): #--> Check for tuple/list
          out_data = torch.as_tensor(in_data[0],dtype=self.data_type,device=self.devices)
          is_object = True
       elif isinstance(in_data,object): #--> Check for object
          out_data = torch.as_tensor(in_data[0],dtype=self.data_type,device=self.devices)
          is_object = True
       else:
          out_data = torch.as_tensor(in_data,dtype=self.data_type,device=self.devices)

       if self.transpose_dim is not None:
          out_data = torch.transpose(out_data,dim0=self.transpose_dim[0],dim1=self.transpose_dim[1])

       out_data = torch.squeeze(out_data)
       # Get the original data:
       original_data = out_data

       if len(self.module_list) > 2:

          # Pass the data through the detector module, if requested:
          out_data = self.pass_data_through_single_modules(out_data,self.module_list[1],is_training,self.use_exp_module)

          # Pass data through event selection module, if requested:
          out_data = self.pass_data_through_single_modules(out_data,self.module_list[2],is_training,self.use_evtsel_module)
       else:
          print(" ")
          print(">>> Exp Data Module: No experimental and event selection modules detected <<<")
          print(" ")

       data_package = {
         'parsed_data': out_data,
         'original_data': original_data
       }

       if is_object:
           i = 0
           #+++++++++++++++++
           for el in in_data[1:]:
               d_name = "extra_observable_" + str(i)
               if len(extra_observable_names) > 0:
                  d_name = extra_observable_names[i]

               data_package[d_name] = torch.as_tensor(el,dtype=self.data_type,device=self.devices)

               i += 1
           #+++++++++++++++++

       return data_package
   #******************************************

   # Now return the data:
   #******************************************
   def return_data(self,is_training,additional_observable_names=[]):
       # Load data:
       input_data = self.load_data()

       # Make some alterations:
       output_data = self.pass_data_through_modules(input_data,is_training,additional_observable_names)

       # Make the data accessible:
       return output_data
   #******************************************

   # Get random batch from (training) data:
   #******************************************
   def get_random_batch_from_data(self,data,batch_size):
       if self.use_full_data_for_training == True:
           return data
       
       rand_idx = torch.randint(high=data.size()[0],size=(batch_size,),device=self.devices)
       return data[rand_idx].to(self.devices)
   #******************************************













