from abc import ABC, abstractmethod


class EventSelection(ABC):
    """
    Event selection base class: Inherits from abstract base class for mandating
    functionality (pure virtual functions).
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def filter(self,data):
        return NotImplemented
        
    # Apply function during training:
    @abstractmethod
    def forward(self,data):
        return NotImplemented
    
    # Apply function outside of training loop:
    @abstractmethod
    def apply(self,data):
        return NotImplemented
