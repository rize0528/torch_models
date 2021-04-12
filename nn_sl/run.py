import sys
import argparse
import models
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.datasets import load_boston, load_iris


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, choices=models.available_models, dest='model_name')
    args = parser.parse_args()

    if args.model_name == "LinearRegression":
        model = models.LinearRegression(1, 1, learning_rate=0.005)
        x_train = np.array([[2.3], [4.4], [3.7], [6.1], [7.3], [2.1], [5.6], [7.7], [8.7], [4.1],
                            [6.7], [6.1], [7.5], [2.1], [7.2],
                            [5.6], [5.7], [7.7], [3.1]], dtype=np.float32)
        y_train = np.array([[3.7], [4.76], [4.], [7.1], [8.6], [3.5], [5.4], [7.6], [7.9], [5.3],
                            [7.3], [7.5], [8.5], [3.2], [8.7],
                            [6.4], [6.6], [7.9], [5.3]], dtype=np.float32)
        t_x = Tensor(x_train)
        t_y = Tensor(y_train)
        tensor_dataset = TensorDataset(t_x, t_y)
        data_loader = DataLoader(tensor_dataset, batch_size=32)
        model.run(data_loader, model)
    elif args.model_name == "LogisticRegression":
        X_train, y_train = load_iris(return_X_y=True)
        t_x, t_y = torch.tensor(X_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.long)
        tensor_dataset = TensorDataset(t_x, t_y)

        num_classes = len(set(y_train))
        input_dim = X_train.shape[1]

        data_loader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)
        model = models.LogisticRegression(input_dim, num_classes, learning_rate=0.01)
        model.run(data_loader, model)


if __name__ == "__main__":
    sys.exit(main())
