import sys
import argparse
import models
import numpy as np
import torchvision
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.datasets import load_boston


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


if __name__ == "__main__":
    sys.exit(main())
