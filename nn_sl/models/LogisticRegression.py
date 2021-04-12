import torch
import torch.nn as nn
import torch.nn.functional as F
from ._module import Module
from torch.utils.data import DataLoader


class LogisticRegression(Module):

    def __init__(self, input_size, num_classes, **kwargs):
        super(LogisticRegression, self).__init__(**kwargs)
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x) -> torch.Tensor:
        out = F.softmax(self.linear(x))
        return out

    def run(self, data_loader: DataLoader, model: nn.Module):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        loss_fun = nn.CrossEntropyLoss()

        for epoch in range(self.epoch):
            for batch_data, batch_label in data_loader:
                optimizer.zero_grad()
                out = model(batch_data)
                loss = loss_fun(out, batch_label)
                loss.backward()
                optimizer.step()
            print(f'epoch:{epoch:02d}, loss:{loss.data.item()}')
