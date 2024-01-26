from abc import ABC, abstractmethod


class ExpData(ABC):
    """
    Experimental data extraction / parsing base class: Inherits from abstract base class for mandating
    functionality (pure virtual functions).
    """

    def __init__(self,**kwargs):
        pass

    @abstractmethod
    def load_data(self):
        return NotImplemented
    
    @abstractmethod
    def return_data(self):
        return NotImplemented
