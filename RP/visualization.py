import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


if __name__ == "__main__":
    model = "resnet"
    acc_base = f"dataset_{model}_ACC/"
    rp_base = f"dataset_{model}_RP/"

    # paths
    cifar_f_path = "cifar10-f.npy"
    acc_cifar_f_path = acc_base + cifar_f_path
    rp_cifar_d_path = rp_base + cifar_f_path

    cifar_c_path = "cifar10-c.npy"
    acc_cifar_c_path = acc_base + cifar_c_path
    rp_cifar_c_path = rp_base + cifar_c_path

    cifar_transformed_path = "cifar10-transformed.npy"
    acc_cifar_transformed_path = acc_base + cifar_transformed_path
    rp_cifar_transformed_path = rp_base + cifar_transformed_path

    cifar_clean = "custom_cifar_clean.npy"
    acc_cifar_clean_path = acc_base + cifar_clean
    rp_cifar_clean_path = rp_base + cifar_clean

    cifar_101 = "cifar-10.1.npy"
    acc_cifar_101_path = acc_base + cifar_101
    rp_cifar_101_path = rp_base + cifar_101

    diffusion = "diffusion_processed.npy"
    acc_diffusion_path = acc_base + diffusion
    rp_diffusion_path = rp_base + diffusion

    # data
    acc_cifar_f = np.load(acc_cifar_f_path)
    rp_cifar_f = np.load(rp_cifar_d_path)
    acc_cifar_c = np.load(acc_cifar_c_path)
    rp_cifar_c = np.load(rp_cifar_c_path)
    acc_cifar_transformed = np.load(acc_cifar_transformed_path)
    rp_cifar_transformed = np.load(rp_cifar_transformed_path)
    acc_cifar_clean = np.load(acc_cifar_clean_path)
    rp_cifar_clean = np.load(rp_cifar_clean_path)
    acc_cifar_101 = np.load(acc_cifar_101_path)
    rp_cifar_101 = np.load(rp_cifar_101_path)
    acc_diffusion = np.load(acc_diffusion_path)
    rp_diffusion = np.load(rp_diffusion_path)

    x_concat = np.concatenate((
        rp_cifar_f,
        rp_cifar_c,
        rp_cifar_transformed,
        rp_cifar_clean,
        np.expand_dims(rp_cifar_101, 0),
        rp_diffusion,
    ))
    y_concat = np.concatenate((
        acc_cifar_f,
        acc_cifar_c,
        acc_cifar_transformed,
        acc_cifar_clean,
        np.expand_dims(acc_cifar_101, 0),
        acc_diffusion,
    ))

    # plot
    # x-axis is the rp, y-axis is the Accuracy
    plt.figure(figsize=(10, 8))
    plt.scatter(x=rp_cifar_transformed, y=acc_cifar_transformed, s=10, label="CIFAR-10-Transformed")
    plt.scatter(x=rp_cifar_c, y=acc_cifar_c, label="CIFAR-10-C")
    plt.scatter(x=rp_cifar_f, y=acc_cifar_f, label="CIFAR-10-F")
    plt.scatter(x=rp_cifar_clean, y=acc_cifar_clean, label="Custom CIFAR-10 clean")
    plt.scatter(x=rp_cifar_101, y=acc_cifar_101, label="CIFAR-10.1")
    plt.scatter(x=rp_diffusion, y=acc_diffusion, label="Diffusion model")

    plt.xlabel("Rotation prediction")
    plt.ylabel("Accuracy")
    plt.title("Accuracy against rp for datasets")
    plt.legend()
    plt.savefig(f"generated_files/{model}_acc_rp.png")

    # statistical analysis
    print(f"Pearson correlation coefficient is: {scipy.stats.pearsonr(x_concat, y_concat)[0]}")
