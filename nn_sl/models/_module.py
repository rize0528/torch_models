import torch.nn as nn
from abc import abstractmethod, ABC
from typing import Optional
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score


class Module(nn.Module, ABC):

    def __init__(self, learning_rate: Optional[float] = 0.0001,
                 epoch: Optional[int] = 50,
                 batch_size: Optional[int] = 64) -> None:
        super(Module, self).__init__()

        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size

    @abstractmethod
    def forward(self, x) -> Tensor:
        pass

    @abstractmethod
    def run(self, data_loader: DataLoader, model: nn.Module):
        pass
