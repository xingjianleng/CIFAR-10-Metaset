import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


if __name__ == "__main__":
    model = "resnet"
    # model = "repvgg"
    acc_base = f"dataset_{model}_ACC/"
    gi_base = f"dataset_{model}_GI/"

    # paths
    cifar_f_path = "cifar10-f-32.npy"
    acc_cifar_f_path = acc_base + cifar_f_path
    gi_cifar_d_path = gi_base + cifar_f_path

    cifar_c_path = "cifar-10.1-c.npy"
    acc_cifar_c_path = acc_base + cifar_c_path
    gi_cifar_c_path = gi_base + cifar_c_path

    cifar_transformed_path = "cifar10-transformed.npy"
    acc_cifar_transformed_path = acc_base + cifar_transformed_path
    gi_cifar_transformed_path = gi_base + cifar_transformed_path

    cifar_clean = "custom_cifar_clean.npy"
    acc_cifar_clean_path = acc_base + cifar_clean
    gi_cifar_clean_path = gi_base + cifar_clean

    cifar_101 = "cifar-10.1.npy"
    acc_cifar_101_path = acc_base + cifar_101
    gi_cifar_101_path = gi_base + cifar_101

    diffusion = "diffusion_processed.npy"
    acc_diffusion_path = acc_base + diffusion
    gi_diffusion_path = gi_base + diffusion

    # data
    acc_cifar_f = np.load(acc_cifar_f_path)
    gi_cifar_f = np.load(gi_cifar_d_path)
    acc_cifar_c = np.load(acc_cifar_c_path)
    gi_cifar_c = np.load(gi_cifar_c_path)
    acc_cifar_transformed = np.load(acc_cifar_transformed_path)
    gi_cifar_transformed = np.load(gi_cifar_transformed_path)
    acc_cifar_clean = np.load(acc_cifar_clean_path)
    gi_cifar_clean = np.load(gi_cifar_clean_path)
    acc_cifar_101 = np.load(acc_cifar_101_path)
    gi_cifar_101 = np.load(gi_cifar_101_path)
    acc_diffusion = np.load(acc_diffusion_path)
    gi_diffusion = np.load(gi_diffusion_path)

    x_concat = np.concatenate((
        gi_cifar_f,
        gi_cifar_c,
        gi_cifar_transformed,
        gi_cifar_clean,
        gi_cifar_101.reshape(1),
        gi_diffusion,
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
    # x-axis is the gi, y-axis is the Accuracy
    plt.figure(figsize=(10, 8))
    plt.scatter(x=gi_cifar_transformed, y=acc_cifar_transformed, s=10, label="CIFAR-10-Transformed")
    plt.scatter(x=gi_cifar_c, y=acc_cifar_c, label="CIFAR-10-C")
    plt.scatter(x=gi_cifar_f, y=acc_cifar_f, label="CIFAR-10-F")
    plt.scatter(x=gi_cifar_clean, y=acc_cifar_clean, label="Custom CIFAR-10 clean")
    plt.scatter(x=gi_cifar_101, y=acc_cifar_101, label="CIFAR-10.1")
    plt.scatter(x=gi_diffusion, y=acc_diffusion, label="Diffusion model")

    plt.xlabel("Grayscale invariance")
    plt.ylabel("Accuracy")
    plt.title("Accuracy against Grayscale invariance for datasets")
    plt.legend()
    plt.savefig(f"generated_files/{model}_acc_gi.png")

    # statistical analysis
    print(f"Pearson correlation coefficient is: {scipy.stats.pearsonr(x_concat, y_concat)[0]}")