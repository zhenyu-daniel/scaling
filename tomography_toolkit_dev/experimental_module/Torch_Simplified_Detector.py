import torch
import functorch
import numpy as np
from tomography_toolkit_dev.core.simulation_core import Simulation

class Simplified_Detector(Simulation):
   """
   Class that mimics a detector with simplified detortions which are implemented as follows:
   X_det = M * X_ideal
   where: X_idal is the undistorted event vector (i.e. after the physics module)

   The matrix M summarizes detector effects and will be defined here as follows:
   M = M_G + c1 * M_c + c2 * (M_r + M_r.T)/2
   with: M_G = diagonal matrix where each diagonal element i is ~N(1,sigma_i)
   M_c: symmetric matrix with: M_c = 0, if i = j and M_c = 1 else
   M_r: symetric matrix with random entries ~U(r1,r2) and 0 on the diagonal
   c1 and c2 are parameters to adjust the magnitude of the correlations
   r1 and r2 adjust the symmetry / asymetry of the correlation matrix

   This implementation might appear cumbersome, but it easy to understand and allows to study different 
   effects seperately (i.e. relative smearing vs. correlations)

   The parameters: sigma_i, c1, c2, r1 and r2 are provided by the user  

   Input: Tensor with shape: N_events x N_features
   Output: Tensor with shape: N_events x N_features

   Note: If the input vector has a different shape, the user may transpose the input vector, by providing a tuple / list
   to the apply_detector_response function which will transpose the input vector according to the torch function
   explained here: https://pytorch.org/docs/stable/generated/torch.transpose.html
   """ 

   # Initialize:
   #**************************************************************
   def __init__(self,config,devices="cpu"):
        self.devices = devices
        self.smearing_parameters = config['smearing_parameters']
        self.transpose_dim = config["transpose_dim"] if "transpose_dim" in config else None
        self.off_diagonal = config['correlation_parameters'] if "correlation_parameters" in config else [0.0,0.0]
        self.m_rnd_asym = config['correlation_asymmetry'] if "correlation_asymmetry" in config else [-1.0,1.0]        
        self.is_blank = config['exp_module_off'] if "exp_module_off" in config else False
      
        self.disable_training_behavior = False #--> This flag is only needed when we wish that .forward() diverges from its behavior 
        # during training. We basically force that .forward() does the same as .apply()
        # This flag is not used here, but there might be cases where we have to have this option

        # Make sure that (some) smearing parameters and the feature dimension are provided
        assert len(self.smearing_parameters) > 0, f"You need to provide a list with smearing values. Lenght of this list has to be equal to the feature dimension"
        self.n_features = len(self.smearing_parameters)

        # Calculate M_c here, since it is just constant
        self.M_c = np.ones((self.n_features,self.n_features))
        np.fill_diagonal(self.M_c,0.0)

        # Calculate a diagonal matrix which simplifies the propagation of detector effects:
        self.M_id = np.zeros((self.n_features,self.n_features))
        np.fill_diagonal(self.M_id,1.0)
   #**************************************************************
        
        

   # Calculate the smearing matrix M:
   #**************************************************************
   def calc_smearing_matrix(self,X):
       # Define the input shape of the matrices, assuming that X has the shape: N_events x N_features:
       matrix_shape = (X.size()[0],self.n_features,self.n_features)

       # Calculate M_r first which gives us a N_events dim tensor where the elements are N_features x N_features matrices:
       rnd_M = np.random.uniform(low=self.m_rnd_asym[0],high=self.m_rnd_asym[1],size=matrix_shape) 
       M_r = (rnd_M + np.transpose(rnd_M,axes=(0,2,1))) / 2 # --> Make everything symmetric

       # Calculate M_G which again results in a N_events dim tensor where the elements are N_features x N_features matrices:
       M_G = np.random.normal(loc=np.ones(self.n_features),scale=self.smearing_parameters,size=matrix_shape)

       # Now compute the full matrix, by folding M_G with the identity and M_rnd with M_c:
       compute_matrix = lambda m_g,m_r: np.multiply(self.M_id,m_g) + self.off_diagonal[0]*self.M_c + self.off_diagonal[1]*np.multiply(self.M_c,m_r)
       M = compute_matrix(M_G,M_r)

       return torch.as_tensor(M,dtype=torch.float32,device=self.devices)
   #**************************************************************
  
   # Apply smearing matrix to one single event:
   #**************************************************************
   def smear_single_event(self,M_smear,X_evt):
       M_smear = M_smear.to(self.devices)
       X_evt = X_evt.to(self.devices)
       return torch.matmul(M_smear,X_evt).to(self.devices)
   #**************************************************************
     
   # Finally, alter the entire data:
   #**************************************************************
   def apply_detector_response(self,data):
        if self.is_blank == True:
            return data

        if self.transpose_dim is not None:
            data_t = torch.transpose(data,dim0=self.transpose_dim[0],dim1=self.transpose_dim[1])

            M = self.calc_smearing_matrix(data_t)
            return functorch.vmap(self.smear_single_event,in_dims=0,randomness="different")(M,data_t).to(self.devices)
       
        M = self.calc_smearing_matrix(data)
        return functorch.vmap(self.smear_single_event,in_dims=0,randomness="different")(M,data).to(self.devices)
   #**************************************************************

   # Use the forward function for consistency inside the GAN
   # training cycle:
   #**************************************************************
   def forward(self,data):
       return self.apply_detector_response(data)
   #**************************************************************

   # Use the apply function for handling the data 'outside' 
   # the training loop:
   #**************************************************************
   def apply(self,data):
       return self.apply_detector_response(data)
   #**************************************************************
   

     

    

        




            




