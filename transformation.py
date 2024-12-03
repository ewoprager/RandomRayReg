import torch
from abc import ABC, abstractmethod


class Transformation(ABC):
    @abstractmethod
    def randomise(self):
        pass

    @abstractmethod
    def get(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_matrix(self) -> torch.Tensor:
        """
        :return: the transpose of the transformation matrix corresponding to the transformation's parameters
        """
        pass

    @abstractmethod
    def enable_grad(self):
        pass

    @abstractmethod
    def disable_grad(self):
        pass


class Rotation1D(Transformation):
    def __init__(self, value: torch.Tensor=torch.tensor([0.0])):
        self.value = value

    def randomise(self):
        self.value = torch.pi * (-1. + 2. * torch.rand(1))

    def get(self) -> torch.Tensor:
        return self.value

    def get_matrix(self) -> torch.Tensor:
        return torch.cat((torch.cat((torch.cos(self.value), torch.sin(self.value)))[None, :], torch.cat((-torch.sin(self.value), torch.cos(self.value)))[None, :]))

    def enable_grad(self):
        self.value.grad = torch.zeros_like(self.value)
        self.value.requires_grad_(True)

    def disable_grad(self):
        self.value.requires_grad_(False)