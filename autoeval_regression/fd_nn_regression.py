import os

import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.utils.data


used_model = "resnet"
# used_model = "repvgg"

epochs = 200

if used_model == "resnet":
    dims = 64
elif used_model == "repvgg":
    dims = 1280
else:
    raise ValueError("Invalid model")


class FdRegressor(nn.Module):
    def __init__(self, dims, hidden_layer_sizes=(100,)):
        super(FdRegressor, self).__init__()
        # proj1 is responsible for projecting the 
        # variance(sigma) from RxR to Rx1
        self.dims = dims
        self.proj1 = nn.Linear(dims, 1)
        hidden_layers = []
        previous_hidden_size = 2 * self.dims + 1
        for hidden_layer_size in hidden_layer_sizes:
            hidden_layers.append(
                nn.Linear(previous_hidden_size, hidden_layer_size)
            )
            hidden_layers.append(
                nn.ReLU(inplace=True)
            )
            previous_hidden_size = hidden_layer_size
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.final_regression = nn.Linear(previous_hidden_size, 1)

    def forward(self, mu, var, fd):
        # batch size is the same
        assert mu.shape[0] == var.shape[0] == fd.shape[0]
        # dimension check
        assert mu.shape[1] == var.shape[1] == var.shape[2] == self.dims
        assert fd.shape[1] == 1
        # projection of variance
        var = self.proj1(var).squeeze(2)
        # put features of neural network regression together
        nn_feature = torch.cat((fd, mu, var), dim=1)
        nn_feature = self.hidden_layers(nn_feature)
        return self.final_regression(nn_feature)


class FdDataset(torch.utils.data.Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        if self.is_train:
            self.mu, self.var, self.fd, self.y = get_train_data()
        else:
            self.mu, self.var, self.fd, self.y = get_test_data()
        # cast to float32 and convert to torch Tensor
        self.mu = torch.from_numpy(self.mu.astype(np.float32))
        self.var = torch.from_numpy(self.var.astype(np.float32))
        self.fd = torch.from_numpy(self.fd.astype(np.float32).reshape(-1, 1))
        self.y = torch.from_numpy(self.y.astype(np.float32))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, index):
        return self.mu[index], self.var[index], self.fd[index], self.y[index]


def get_train_data():
    # CIFAR-10-TRANSFORMED is used as training data
    train_set = "cifar10-transformed"
    metaset_folder = f"dataset_{used_model}_feature/{train_set}"
    acc_file = f"dataset_{used_model}_ACC/{train_set}.npy"

    dataset_ranges = len(os.listdir(metaset_folder)) // 3
    train_mu = np.zeros((dataset_ranges, dims))
    train_var = np.zeros((dataset_ranges, dims, dims))
    train_fd = np.load(f"dataset_{used_model}_FD/{train_set}.npy")

    for i, dataset_name in enumerate(range(dataset_ranges)):
        train_mu[i] = np.load(f"{metaset_folder}/{dataset_name}_mean.npy")
        train_var[i] = np.load(f"{metaset_folder}/{dataset_name}_variance.npy")

    train_y = np.load(acc_file)
    return train_mu, train_var, train_fd, train_y


def get_test_data():
    # NOTE: For test data, currently we include CIFAR-10.1, CIFAR-10.1-C, CIFAR-10-F
    test_sets = ["cifar-10.1", "cifar-10.1-c", "cifar10-f-32"]
    test_mu, test_var, test_fd, test_y = [], [], [], []

    for test_set in test_sets:
        metaset_folder = f"dataset_{used_model}_feature/{test_set}"
        acc_file = f"dataset_{used_model}_ACC/{test_set}.npy"

        dataset_ranges = len(os.listdir(metaset_folder)) // 3
        test_dataset_mu = np.zeros((dataset_ranges, dims))
        test_dataset_var = np.zeros((dataset_ranges, dims, dims))
        test_dataset_fd = np.load(f"dataset_{used_model}_FD/{test_set}.npy")

        for i, dataset_name in enumerate(range(dataset_ranges)):
            test_dataset_mu[i] = np.load(f"{metaset_folder}/{dataset_name}_mean.npy")
            test_dataset_var[i] = np.load(f"{metaset_folder}/{dataset_name}_variance.npy")
        acc = np.load(acc_file)
        if not acc.shape:
            acc = acc.reshape(1)

        test_mu.append(test_dataset_mu)
        test_var.append(test_dataset_var)
        test_fd.append(test_dataset_fd)
        test_y.append(acc)
    
    test_mu = np.concatenate(test_mu)
    test_var = np.concatenate(test_var)
    test_fd = np.concatenate(test_fd)
    test_y = np.concatenate(test_y)
    return test_mu, test_var, test_fd, test_y


if __name__ == "__main__":
    train_data = FdDataset(True)
    test_data = FdDataset(False)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=16,
        shuffle=False
    )

    regressor = FdRegressor(dims=dims, hidden_layer_sizes=(200, 200))
    optim = torch.optim.SGD(regressor.parameters(), lr=1e-4)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (mu, var, fd, y) in enumerate(train_loader):
            predicted = regressor(mu, var, fd).squeeze(1)
            loss = nn.functional.mse_loss(predicted, y)
            loss.backward()
            optim.step()

            running_loss += loss.item()
        print(f"Epoch {epoch} running loss: {running_loss / i}")

    prediction = []
    for mu, var, fd, _ in iter(test_loader):
        predicted = regressor(mu, var, fd).squeeze(1).detach()
        prediction.append(predicted)
    prediction = torch.cat(prediction).numpy()
    test_y = test_data.y.numpy()
    print(f"Testing RMSE is: {mean_squared_error(test_y, prediction, squared=False)}")
