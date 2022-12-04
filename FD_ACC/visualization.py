import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    acc_base = "dataset_ACC/"
    fd_base = "dataset_FD/"

    cifar_f_path = "cifar10-f.npy"
    acc_cifar_f_path = acc_base + cifar_f_path
    fd_cifar_d_path = fd_base + cifar_f_path

    custom_cifar_path = "custom_cifar.npy"
    acc_custom_cifar_path = acc_base + custom_cifar_path
    fd_custom_cifar_path = fd_base + custom_cifar_path

    correct_wrong_path = "correct_wrong.npy"
    acc_correct_wrong_path = acc_base + correct_wrong_path
    fd_correct_wrong_path = fd_base + correct_wrong_path

    acc_cifar_f = np.load(acc_cifar_f_path)
    fd_cifar_f = np.load(fd_cifar_d_path)
    acc_custom_cifar = np.load(acc_custom_cifar_path)
    fd_custom_cifar = np.load(fd_custom_cifar_path)
    acc_correct_wrong = np.load(acc_correct_wrong_path)
    fd_correct_wrong = np.load(fd_correct_wrong_path)

    # x-axis is the FD, y-axis is the Accuracy
    plt.scatter(x=fd_cifar_f, y=acc_cifar_f, label="CIFAR-10-F")
    plt.scatter(x=fd_custom_cifar, y=acc_custom_cifar, label="Custom CIFAR")
    plt.scatter(x=fd_correct_wrong, y=acc_correct_wrong, label="Correct/Wrong CIFAR-10-F")
    plt.xlabel("FD")
    plt.ylabel("Accuracy")
    plt.title("Accuracy against FD for datasets")
    plt.legend()
    plt.show()
