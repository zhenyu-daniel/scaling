from abc import ABC, abstractmethod

class Discriminator(ABC):
    """
    Generator base class: Inherits from abstract base class for mandating
    functionality (pure virtual functions).
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def forward(self):
        return NotImplemented

    @abstractmethod
    def train(self, real, fake):
        return NotImplemented
