import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


if __name__ == "__main__":
    model = "resnet"
    # model = "repvgg"
    threshold = 0.8
    acc_base = f"dataset_{model}_ACC/"
    ps_base = f"dataset_{model}_PS_{threshold}/"

    # paths
    cifar_f_path = "cifar10-f-32.npy"
    acc_cifar_f_path = acc_base + cifar_f_path
    ps_cifar_d_path = ps_base + cifar_f_path

    cifar_c_path = "cifar-10.1-c.npy"
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

    diffusion = "diffusion_processed.npy"
    acc_diffusion_path = acc_base + diffusion
    ps_diffusion_path = ps_base + diffusion

    cartoon = "google_cartoon.npy"
    acc_cartoon_path = acc_base + cartoon
    ps_cartoon_path = ps_base + cartoon

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
    acc_diffusion = np.load(acc_diffusion_path)
    ps_diffusion = np.load(ps_diffusion_path)
    acc_cartoon = np.load(acc_cartoon_path)
    ps_cartoon = np.load(ps_cartoon_path)

    x_concat = np.concatenate((
        ps_cifar_f,
        ps_cifar_c,
        ps_cifar_transformed,
        ps_cifar_clean,
        ps_cifar_101.reshape(1),
        ps_diffusion,
        ps_cartoon,

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
    # x-axis is the ps, y-axis is the Accuracy
    plt.figure(figsize=(10, 8))
    plt.scatter(x=ps_cifar_transformed, y=acc_cifar_transformed, s=10, label="CIFAR-10-Transformed")
    plt.scatter(x=ps_cifar_c, y=acc_cifar_c, label="CIFAR-10-C")
    plt.scatter(x=ps_cifar_f, y=acc_cifar_f, label="CIFAR-10-F")
    plt.scatter(x=ps_cifar_clean, y=acc_cifar_clean, label="Custom CIFAR-10 clean")
    plt.scatter(x=ps_cifar_101, y=acc_cifar_101, label="CIFAR-10.1")
    plt.scatter(x=ps_diffusion, y=acc_diffusion, label="Diffusion model")
    plt.scatter(x=ps_cartoon, y=acc_cartoon, label="Google Cartoon")

    plt.xlabel("Predicted Score")
    plt.ylabel("Accuracy")
    plt.title("Accuracy against Predicted Score for datasets")
    plt.legend()
    plt.savefig(f"generated_files/{model}_acc_ps.png")

    # statistical analysis
    print(f"Pearson correlation coefficient is: {scipy.stats.pearsonr(x_concat, y_concat)[0]}")
