import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ._module import Module


class Convolution2D(Module):

    def __init__(self, **kwargs):
        super(Convolution2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_drop = nn.Dropout2d(p=0.25)
        self.fc1 = nn.Linear(980, 64)
        self.fc2 = nn.Linear(64, 10)

        self.epoch = kwargs.get('n_epoch', 10)
        self.log_per_batch = kwargs.get('log_per_batch', 100)
        self.train_batch_size = kwargs.get('train_batch_size', 32)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(len(x), -1)
        x = torch.sigmoid(self.fc1(x))
        x = F.dropout(x, p=0.25)
        x = self.fc2(x)

        return F.log_softmax(x)

    def run(self, train: DataLoader, test: DataLoader):

        learning_rate = 0.01
        momentum = 0.5

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        loss_fun = nn.MSELoss()

        for epoch in range(self.epoch+1):
            self.train()
            for batch_index, (batch_data, batch_label) in enumerate(train):
                optimizer.zero_grad()
                out = self(batch_data)
                loss = F.nll_loss(out, batch_label)
                loss.backward()
                optimizer.step()
                if batch_index % self.log_per_batch == 0:
                    fstring = 'Epoch: {epoch} [{batch_status}/{total_samples}]({progress_p:.2f}%), loss: {loss:.5f}'
                    print(fstring.format(epoch=epoch+1,
                                         batch_status=(batch_index+1)*train.batch_size,
                                         total_samples=len(train.dataset),
                                         progress_p=100. * batch_index/len(train),
                                         loss=loss.item()))
            #
            self.eval()
            loss = 0
            correct = 0
            with torch.no_grad():
                for batch_data, batch_label in test:
                    output = self(batch_data)
                    loss += F.nll_loss(output, batch_label, reduction='sum').item()
                    pred = output.data.max(1, keepdim=True)[1]  # 0: value, 1: index
                    correct += pred.eq(batch_label.view_as(pred)).sum()
            loss = loss / len(test.dataset)
            print('==> Evaluation result: /{ correctness: [{}/{}], loss: {} /}'.format(
                correct, len(test.dataset), loss
            ))
