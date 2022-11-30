import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    acc_base = "../dataset_ACC/"
    fd_base = "../dataset_FD/"

    cifar_f_path = "cifar10-f.npy"
    acc_cifar_f_path = acc_base + cifar_f_path
    fd_cifar_d_path = fd_base + cifar_f_path

    acc_cifar_f = np.load(acc_cifar_f_path)
    fd_cifar_d = np.load(fd_cifar_d_path)

    # x-axis is the FD, y-axis is the Accuracy
    plt.scatter(x=fd_cifar_d, y=acc_cifar_f, label="CIFAR-10-F")
    plt.xlabel("FD")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
