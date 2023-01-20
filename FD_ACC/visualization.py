import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


if __name__ == "__main__":
    model = "resnet"
    # model = "repvgg"
    acc_base = f"dataset_{model}_ACC/"
    fd_base = f"dataset_{model}_FD/"

    # paths
    cifar_f_path = "cifar10-f-32.npy"
    acc_cifar_f_path = acc_base + cifar_f_path
    fd_cifar_d_path = fd_base + cifar_f_path

    # custom_cifar_path = "custom_cifar.npy"
    # acc_custom_cifar_path = acc_base + custom_cifar_path
    # fd_custom_cifar_path = fd_base + custom_cifar_path

    # correct_wrong_path = "correct_wrong.npy"
    # acc_correct_wrong_path = acc_base + correct_wrong_path
    # fd_correct_wrong_path = fd_base + correct_wrong_path

    cifar_c_path = "cifar-10.1-c.npy"
    acc_cifar_c_path = acc_base + cifar_c_path
    fd_cifar_c_path = fd_base + cifar_c_path

    cifar_transformed_path = "cifar10-transformed.npy"
    acc_cifar_transformed_path = acc_base + cifar_transformed_path
    fd_cifar_transformed_path = fd_base + cifar_transformed_path

    cifar_clean = "custom_cifar_clean.npy"
    acc_cifar_clean_path = acc_base + cifar_clean
    fd_cifar_clean_path = fd_base + cifar_clean

    cifar_101 = "cifar-10.1.npy"
    acc_cifar_101_path = acc_base + cifar_101
    fd_cifar_101_path = fd_base + cifar_101

    diffusion = "diffusion_processed.npy"
    acc_diffusion_path = acc_base + diffusion
    fd_diffusion_path = fd_base + diffusion

    # data
    acc_cifar_f = np.load(acc_cifar_f_path)
    fd_cifar_f = np.load(fd_cifar_d_path)
    # acc_custom_cifar = np.load(acc_custom_cifar_path)
    # fd_custom_cifar = np.load(fd_custom_cifar_path)
    # acc_correct_wrong = np.load(acc_correct_wrong_path)
    # fd_correct_wrong = np.load(fd_correct_wrong_path)
    acc_cifar_c = np.load(acc_cifar_c_path)
    fd_cifar_c = np.load(fd_cifar_c_path)
    acc_cifar_transformed = np.load(acc_cifar_transformed_path)
    fd_cifar_transformed = np.load(fd_cifar_transformed_path)
    acc_cifar_clean = np.load(acc_cifar_clean_path)
    fd_cifar_clean = np.load(fd_cifar_clean_path)
    acc_cifar_101 = np.load(acc_cifar_101_path)
    fd_cifar_101 = np.load(fd_cifar_101_path)
    acc_diffusion = np.load(acc_diffusion_path)
    fd_diffusion = np.load(fd_diffusion_path)

    x_concat = np.concatenate((
        fd_cifar_f,
        fd_cifar_c,
        fd_cifar_transformed,
        fd_cifar_clean,
        fd_cifar_101.reshape(1),
        fd_diffusion,
    ))
    y_concat = np.concatenate((
        acc_cifar_f,
        acc_cifar_c,
        acc_cifar_transformed,
        acc_cifar_clean,
        acc_cifar_101.reshape(1),
        acc_diffusion,
    ))

    # plot
    # x-axis is the FD, y-axis is the Accuracy
    plt.figure(figsize=(10, 8))
    plt.scatter(x=fd_cifar_transformed, y=acc_cifar_transformed, s=10, label="CIFAR-10-Transformed")
    plt.scatter(x=fd_cifar_c, y=acc_cifar_c, label="CIFAR-10.1-C")
    plt.scatter(x=fd_cifar_f, y=acc_cifar_f, label="CIFAR-10-F")
    # plt.scatter(x=fd_custom_cifar, y=acc_custom_cifar, label="Custom CIFAR")
    # plt.scatter(x=fd_correct_wrong, y=acc_correct_wrong, label="Sampled target accuracy")
    plt.scatter(x=fd_cifar_clean, y=acc_cifar_clean, label="Custom CIFAR-10 clean")
    plt.scatter(x=fd_cifar_101, y=acc_cifar_101, label="CIFAR-10.1")
    plt.scatter(x=fd_diffusion, y=acc_diffusion, label="Diffusion model")

    plt.xlabel("FD")
    plt.ylabel("Accuracy")
    plt.title("Accuracy against FD for datasets")
    plt.legend()
    plt.savefig(f"generated_files/{model}_acc_fd.png")

    # statistical analysis
    print(f"Pearson correlation coefficient is: {scipy.stats.pearsonr(x_concat, y_concat)[0]}")
