from abc import ABC, abstractmethod 


class Experimental(ABC):
   # This is a skeleton for a module core class.
   # Inherits from abstract base class for mandating functionality (pure virtual functions). 


   def __init__(self,**kwargs): 
      pass 


   # We define the .forward() function which is used for data processing during the GAN training 
   @abstractmethod 
   def forward(self): 
      return NotImplemented 


   # We define the .apply() function which is used for data processing outside the GAN training 
   @abstractmethod 
   def apply(self): 
      return NotImplemented 


