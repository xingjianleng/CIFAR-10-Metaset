import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    model = "lenet"
    acc_base = f"dataset_{model}_ACC/"
    fd_base = f"dataset_{model}_FD/"

    # paths
    cifar_f_path = "cifar10-f.npy"
    acc_cifar_f_path = acc_base + cifar_f_path
    fd_cifar_d_path = fd_base + cifar_f_path

    custom_cifar_path = "custom_cifar.npy"
    acc_custom_cifar_path = acc_base + custom_cifar_path
    fd_custom_cifar_path = fd_base + custom_cifar_path

    correct_wrong_path = "correct_wrong.npy"
    acc_correct_wrong_path = acc_base + correct_wrong_path
    fd_correct_wrong_path = fd_base + correct_wrong_path

    cifar_c_path = "cifar10-c.npy"
    acc_cifar_c_path = acc_base + cifar_c_path
    fd_cifar_c_path = fd_base + cifar_c_path

    cifar_transformed_path = "cifar10-transformed.npy"
    acc_cifar_transformed_path = acc_base + cifar_transformed_path
    fd_cifar_transformed_path = fd_base + cifar_transformed_path

    # data
    acc_cifar_f = np.load(acc_cifar_f_path)
    fd_cifar_f = np.load(fd_cifar_d_path)
    acc_custom_cifar = np.load(acc_custom_cifar_path)
    fd_custom_cifar = np.load(fd_custom_cifar_path)
    acc_correct_wrong = np.load(acc_correct_wrong_path)
    fd_correct_wrong = np.load(fd_correct_wrong_path)
    acc_cifar_c = np.load(acc_cifar_c_path)
    fd_cifar_c = np.load(fd_cifar_c_path)
    acc_cifar_transformed = np.load(acc_cifar_transformed_path)
    fd_cifar_transformed = np.load(fd_cifar_transformed_path)

    # plot
    # x-axis is the FD, y-axis is the Accuracy
    plt.scatter(x=fd_cifar_transformed, y=acc_cifar_transformed, label="CIFAR-10-Transformed")
    plt.scatter(x=fd_cifar_c, y=acc_cifar_c, label="CIFAR-10-C")
    plt.scatter(x=fd_cifar_f, y=acc_cifar_f, label="CIFAR-10-F")
    plt.scatter(x=fd_custom_cifar, y=acc_custom_cifar, label="Custom CIFAR")
    plt.scatter(x=fd_correct_wrong, y=acc_correct_wrong, label="Sampled target accuracy")

    plt.xlabel("FD")
    plt.ylabel("Accuracy")
    plt.title("Accuracy against FD for datasets")
    plt.legend()
    plt.savefig(f"{model}_acc_fd.png")
