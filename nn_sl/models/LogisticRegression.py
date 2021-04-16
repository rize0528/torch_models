import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._module import Module
from torch.utils.data import DataLoader
from sklearn.metrics import recall_score, precision_score


class LogisticRegression(Module):

    def __init__(self, input_size, num_classes, **kwargs):
        super(LogisticRegression, self).__init__(**kwargs)
        self.linear = nn.Linear(input_size, num_classes)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x) -> torch.Tensor:
        out = F.softmax(self.linear(x), dim=1)
        return out

    def run(self, data_loader: DataLoader, model: nn.Module):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        loss_fun = nn.CrossEntropyLoss()

        for epoch in range(self.epoch):
            result_confusion_mx = []
            for batch_data, batch_label in data_loader:
                optimizer.zero_grad()
                out = model(batch_data)
                out_label = torch.argmax(out, dim=1)
                _results_batch = self.result_evaluation_mini_batch(out_label, batch_label)
                loss = loss_fun(out, batch_label)
                loss.backward()
                optimizer.step()
                result_confusion_mx.append(_results_batch['confusion'])
            if epoch % 10 == 0:
                wrap_up_conf_mx = np.sum(result_confusion_mx, axis=0)
                ans = self.metrics(wrap_up_conf_mx)
                print(f'epoch:{epoch:02d}, accuracy:{ans["accuracy"]:.4f}, loss:{loss.data.item()}')
        #