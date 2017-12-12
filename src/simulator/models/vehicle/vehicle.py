from abc import ABCMeta, abstractmethod

class Vehicle(metaclass=ABCMeta):

    @abstractmethod
    def step(self):
        pass

