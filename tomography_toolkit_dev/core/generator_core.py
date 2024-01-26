from abc import ABC, abstractmethod


class Generator(ABC):
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
    def generate(self):
        return NotImplemented

    @abstractmethod
    def train(self, noise):
        return NotImplemented
