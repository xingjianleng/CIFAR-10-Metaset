import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


if __name__ == "__main__":
    model = "resnet"
    acc_base = f"dataset_{model}_ACC/"
    ps_base = f"dataset_{model}_PS/"

    # paths
    cifar_f_path = "cifar10-f.npy"
    acc_cifar_f_path = acc_base + cifar_f_path
    ps_cifar_d_path = ps_base + cifar_f_path

    cifar_c_path = "cifar10-c.npy"
    acc_cifar_c_path = acc_base + cifar_c_path
    ps_cifar_c_path = ps_base + cifar_c_path

    cifar_transformed_path = "cifar10-transformed.npy"
    acc_cifar_transformed_path = acc_base + cifar_transformed_path
    ps_cifar_transformed_path = ps_base + cifar_transformed_path

    cifar_clean = "custom_cifar_clean.npy"
    acc_cifar_clean_path = acc_base + cifar_clean
    ps_cifar_clean_path = ps_base + cifar_clean

    cifar_101 = "cifar-10.1.npy"
    acc_cifar_101_path = acc_base + cifar_101
    ps_cifar_101_path = ps_base + cifar_101

    # data
    acc_cifar_f = np.load(acc_cifar_f_path)
    ps_cifar_f = np.load(ps_cifar_d_path)
    acc_cifar_c = np.load(acc_cifar_c_path)
    ps_cifar_c = np.load(ps_cifar_c_path)
    acc_cifar_transformed = np.load(acc_cifar_transformed_path)
    ps_cifar_transformed = np.load(ps_cifar_transformed_path)
    acc_cifar_clean = np.load(acc_cifar_clean_path)
    ps_cifar_clean = np.load(ps_cifar_clean_path)
    acc_cifar_101 = np.load(acc_cifar_101_path)
    ps_cifar_101 = np.load(ps_cifar_101_path)

    x_concat = np.concatenate((
        ps_cifar_f,
        ps_cifar_c,
        ps_cifar_transformed,
        ps_cifar_clean,
        np.expand_dims(ps_cifar_101, 0),
    ))
    y_concat = np.concatenate((
        acc_cifar_f,
        acc_cifar_c,
        acc_cifar_transformed,
        acc_cifar_clean,
        np.expand_dims(acc_cifar_101, 0),
    ))

    # plot
    # x-axis is the ps, y-axis is the Accuracy
    plt.figure(figsize=(10, 8))
    plt.scatter(x=ps_cifar_transformed, y=acc_cifar_transformed, s=10, label="CIFAR-10-Transformed")
    plt.scatter(x=ps_cifar_c, y=acc_cifar_c, label="CIFAR-10-C")
    plt.scatter(x=ps_cifar_f, y=acc_cifar_f, label="CIFAR-10-F")
    plt.scatter(x=ps_cifar_clean, y=acc_cifar_clean, label="Custom CIFAR-10 clean")
    plt.scatter(x=ps_cifar_101, y=acc_cifar_101, label="CIFAR-10.1")

    plt.xlabel("Predicted Score")
    plt.ylabel("Accuracy")
    plt.title("Accuracy against Predicted Score for datasets")
    plt.legend()
    plt.savefig(f"generated_files/{model}_acc_ps.png")

    # statistical analysis
    print(f"Pearson correlation coefficient is: {scipy.stats.pearsonr(x_concat, y_concat)[0]}")
