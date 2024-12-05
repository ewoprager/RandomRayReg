import torch
from abc import ABC, abstractmethod
import kornia


class Transformation(ABC):
    @abstractmethod
    def randomise(self):
        pass

    @abstractmethod
    def get(self) -> torch.Tensor:
        pass

    @abstractmethod
    def inverse(self):
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

    @abstractmethod
    def __format__(self, fmt):
        pass


class Rotation1D(Transformation):
    def __init__(self, value: torch.Tensor=torch.tensor([0.0])):
        self.value = value

    def randomise(self):
        self.value = torch.pi * (-1. + 2. * torch.rand(1))

    def get(self) -> torch.Tensor:
        return self.value

    def inverse(self):
        return Rotation1D(-self.value)

    def get_matrix(self) -> torch.Tensor:
        return torch.cat((torch.cat((torch.cos(self.value), torch.sin(self.value)))[None, :], torch.cat((-torch.sin(self.value), torch.cos(self.value)))[None, :]))

    def enable_grad(self):
        self.value.grad = torch.zeros_like(self.value)
        self.value.requires_grad_(True)

    def disable_grad(self):
        self.value.requires_grad_(False)

    def __format__(self, fmt) -> str:
        return self.value.item().__str__()


class SE3(Transformation):
    def __init__(self, value: torch.Tensor=torch.zeros(6)):
        self.value = value

    def randomise(self):
        orientation = -1. + 2. * torch.rand(3)
        while torch.norm(orientation) > 1.:
            orientation = -1. + 2. * torch.rand(3)
        self.value = torch.cat((-.25 + 0.5 * torch.rand(3), torch.pi * orientation))

    def get(self) -> torch.Tensor:
        return self.value

    def inverse(self):
        return SE3(-self.value)

    def get_matrix(self) -> torch.Tensor:
        r = kornia.geometry.conversions.axis_angle_to_rotation_matrix(self.value[None, 3:])[0]
        t = self.value[:3]
        return torch.cat((torch.cat((torch.transpose(r, 1, 0), t[None, :])), torch.tensor([0., 0., 0., 1.])[:, None]), dim=1)

    def enable_grad(self):
        self.value.grad = torch.zeros_like(self.value)
        self.value.requires_grad_(True)

    def disable_grad(self):
        self.value.requires_grad_(False)

    def __format__(self, fmt) -> str:
        return self.value.__str__()