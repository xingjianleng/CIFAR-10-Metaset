import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


if __name__ == "__main__":
    model = "resnet"
    # model = "repvgg"
    acc_base = f"dataset_{model}_ACC/"
    ri_base = f"dataset_{model}_RI/"

    # paths
    cifar_f_path = "cifar10-f-32.npy"
    acc_cifar_f_path = acc_base + cifar_f_path
    ri_cifar_d_path = ri_base + cifar_f_path

    cifar_c_path = "cifar-10.1-c.npy"
    acc_cifar_c_path = acc_base + cifar_c_path
    ri_cifar_c_path = ri_base + cifar_c_path

    cifar_transformed_path = "cifar10-transformed.npy"
    acc_cifar_transformed_path = acc_base + cifar_transformed_path
    ri_cifar_transformed_path = ri_base + cifar_transformed_path

    cifar_clean = "custom_cifar_clean.npy"
    acc_cifar_clean_path = acc_base + cifar_clean
    ri_cifar_clean_path = ri_base + cifar_clean

    cifar_101 = "cifar-10.1.npy"
    acc_cifar_101_path = acc_base + cifar_101
    ri_cifar_101_path = ri_base + cifar_101

    diffusion = "diffusion_processed.npy"
    acc_diffusion_path = acc_base + diffusion
    ri_diffusion_path = ri_base + diffusion

    cartoon = "google_cartoon.npy"
    acc_cartoon_path = acc_base + cartoon
    ri_cartoon_path = ri_base + cartoon

    # data
    acc_cifar_f = np.load(acc_cifar_f_path)
    ri_cifar_f = np.load(ri_cifar_d_path)
    acc_cifar_c = np.load(acc_cifar_c_path)
    ri_cifar_c = np.load(ri_cifar_c_path)
    acc_cifar_transformed = np.load(acc_cifar_transformed_path)
    ri_cifar_transformed = np.load(ri_cifar_transformed_path)
    acc_cifar_clean = np.load(acc_cifar_clean_path)
    ri_cifar_clean = np.load(ri_cifar_clean_path)
    acc_cifar_101 = np.load(acc_cifar_101_path)
    ri_cifar_101 = np.load(ri_cifar_101_path)
    acc_diffusion = np.load(acc_diffusion_path)
    ri_diffusion = np.load(ri_diffusion_path)
    acc_cartoon = np.load(acc_cartoon_path)
    ri_cartoon = np.load(ri_cartoon_path)

    x_concat = np.concatenate((
        ri_cifar_f,
        ri_cifar_c,
        ri_cifar_transformed,
        ri_cifar_clean,
        ri_cifar_101.reshape(1),
        ri_diffusion,
        ri_cartoon,
    ))
    y_concat = np.concatenate((
        acc_cifar_f,
        acc_cifar_c,
        acc_cifar_transformed,
        acc_cifar_clean,
        acc_cifar_101.reshape(1),
        acc_diffusion,
        acc_cartoon,
    ))

    # plot
    # x-axis is the ri, y-axis is the Accuracy
    plt.figure(figsize=(10, 8))
    plt.scatter(x=ri_cifar_transformed, y=acc_cifar_transformed, s=10, label="CIFAR-10-Transformed")
    plt.scatter(x=ri_cifar_c, y=acc_cifar_c, label="CIFAR-10-C")
    plt.scatter(x=ri_cifar_f, y=acc_cifar_f, label="CIFAR-10-F")
    plt.scatter(x=ri_cifar_clean, y=acc_cifar_clean, label="Custom CIFAR-10 clean")
    plt.scatter(x=ri_cifar_101, y=acc_cifar_101, label="CIFAR-10.1")
    plt.scatter(x=ri_diffusion, y=acc_diffusion, label="Diffusion model")
    plt.scatter(x=ri_cartoon, y=acc_cartoon, label="Google Cartoon")

    plt.xlabel("Rotation invariance")
    plt.ylabel("Accuracy")
    plt.title("Accuracy against Rotation invariance for datasets")
    plt.legend()
    plt.savefig(f"generated_files/{model}_acc_ri.png")

    # statistical analysis
    print(f"Pearson correlation coefficient is: {scipy.stats.pearsonr(x_concat, y_concat)[0]}")
