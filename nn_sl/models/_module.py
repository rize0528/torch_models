import numpy as np
import torch.nn as nn
from abc import abstractmethod, ABC
from typing import Optional
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix


class Module(nn.Module, ABC):

    def __init__(self, learning_rate: Optional[float] = 0.0001,
                 epoch: Optional[int] = 50,
                 batch_size: Optional[int] = 64) -> None:
        super(Module, self).__init__()

        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size

    def result_evaluation_mini_batch(self, pred: Tensor, target: Tensor):
        assert pred.shape == target.shape
        _target, _pred = target.numpy(), pred.numpy()
        rtn_dict = {
            'accuracy': accuracy_score(_target, _pred),
            'confusion': confusion_matrix(_target, _pred)
        }
        return rtn_dict

    def metrics(self, confusion_matrix: np.array):
        rtn = {}
        if confusion_matrix.shape[0] > 2:
            _correct = np.sum(confusion_matrix.diagonal())
            _wrong = np.sum(confusion_matrix) - _correct
            rtn['accuracy'] = _correct/(_correct+_wrong)
        else:
            pass
        return rtn

    @abstractmethod
    def forward(self, x) -> Tensor:
        pass

    @abstractmethod
    def run(self, data_loader: DataLoader, model: nn.Module):
        pass
