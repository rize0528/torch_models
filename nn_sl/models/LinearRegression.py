import torch
import torch.nn as nn
from ._module import Module


class LinearRegression(Module):

    def __init__(self, input_size, output_size, **kwargs):
        super(LinearRegression, self).__init__(**kwargs)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out

    def run(self, loader, model):

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        loss_fun = nn.MSELoss()

        for epoch in range(self.epoch):
            for batch_data, batch_label in loader:
                optimizer.zero_grad()
                out = model(batch_data)
                loss = loss_fun(out, batch_label)
                loss.backward()
                print(f'epoch:{epoch:02d}, loss: {loss.data.item()}')
                optimizer.step()
