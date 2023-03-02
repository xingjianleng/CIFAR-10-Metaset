import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


if __name__ == "__main__":
    # model = "resnet"
    model = "repvgg"
    acc_base = f"dataset_{model}_ACC/"
    fd_base = f"dataset_{model}_FD/"

    # paths
    cifar_f_path = "cifar10-f-32.npy"
    acc_cifar_f_path = acc_base + cifar_f_path
    fd_cifar_d_path = fd_base + cifar_f_path

    cifar_c_path = "cifar10-c.npy"
    acc_cifar_c_path = acc_base + cifar_c_path
    fd_cifar_c_path = fd_base + cifar_c_path

    cifar_101_c_path = "cifar-10.1-c.npy"
    acc_cifar_101_c_path = acc_base + cifar_101_c_path
    fd_cifar_101_c_path = fd_base + cifar_101_c_path

    train_data_path = "train_data.npy"
    acc_train_data_path = acc_base + train_data_path
    fd_train_data_path = fd_base + train_data_path

    cifar_101_path = "cifar-10.1.npy"
    acc_cifar_101_path = acc_base + cifar_101_path
    fd_cifar_101_path = fd_base + cifar_101_path

    test_data_path = "test_data_processed_ver2.0.npy"
    acc_test_data_path = acc_base + test_data_path
    fd_test_data_path = fd_base + test_data_path

    # data
    acc_cifar_f = np.load(acc_cifar_f_path)
    fd_cifar_f = np.load(fd_cifar_d_path)
    acc_cifar_c = np.load(acc_cifar_c_path)
    fd_cifar_c = np.load(fd_cifar_c_path)
    acc_cifar_101_c = np.load(acc_cifar_101_c_path)
    fd_cifar_101_c = np.load(fd_cifar_101_c_path)
    acc_train_data = np.load(acc_train_data_path)
    fd_train_data = np.load(fd_train_data_path)
    acc_cifar_101 = np.load(acc_cifar_101_path)
    fd_cifar_101 = np.load(fd_cifar_101_path)
    acc_test_data = np.load(acc_test_data_path)
    fd_test_data = np.load(fd_test_data_path)

    x_concat = np.concatenate((
        fd_cifar_f,
        fd_cifar_c,
        fd_train_data,
        fd_cifar_101.reshape(1),
        fd_cifar_101_c,
        fd_test_data,
    ))
    y_concat = np.concatenate((
        acc_cifar_f,
        acc_cifar_c,
        acc_train_data,
        acc_cifar_101.reshape(1),
        acc_cifar_101_c,
        acc_test_data,
    ))

    # plot
    # x-axis is the FD, y-axis is the Accuracy
    plt.figure(figsize=(10, 8))
    plt.scatter(x=fd_train_data, y=acc_train_data, s=10, label="Train Datasets")
    plt.scatter(x=fd_cifar_c, y=acc_cifar_c, label="CIFAR-10-C")
    plt.scatter(x=fd_cifar_101_c, y=acc_cifar_101_c, label="CIFAR-10.1-C")
    plt.scatter(x=fd_cifar_f, y=acc_cifar_f, label="CIFAR-10-F")
    plt.scatter(x=fd_cifar_101, y=acc_cifar_101, label="CIFAR-10.1")
    plt.scatter(x=fd_test_data, y=acc_test_data, label="Test Datasets")

    plt.xlabel("FD")
    plt.ylabel("Accuracy")
    plt.title("Accuracy against FD for datasets")
    plt.legend()
    plt.savefig(f"generated_files/{model}_acc_fd.png")

    # statistical analysis
    print(f"Pearson correlation coefficient is: {scipy.stats.pearsonr(x_concat, y_concat)[0]}")
