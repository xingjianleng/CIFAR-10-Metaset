import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error


methods = ["FD", "PS", "ES", "RP", "GI", "RI"]
ps_thresh = [0.8]
es_thresh = [0.3]
method_names = {
    "FD": "Frechet Distance",
    "PS": "Prediction score",
    "ES": "Entropy score",
    "RP": "Rotation prediction",
    "GI": "Grayscale invariance",
    "RI": "Rotation invariance",
}
used_model = "resnet"
regressor = "LinearRegression"
# regressor = "HuberRegressor"


def get_train_data(method, **kwargs):
    assert method in methods
    if method == "PS" or method == "ES":
        threshold = kwargs.get("threshold", None)
        assert threshold is not None
        base_dir_method = f"dataset_{used_model}_{method}_{threshold}"
    else:
        base_dir_method = f"dataset_{used_model}_{method}"
    base_dir_acc = f"dataset_{used_model}_ACC"
    x = np.load(f"{base_dir_method}/cifar10-transformed.npy")
    y = np.load(f"{base_dir_acc}/cifar10-transformed.npy")
    # convert accuracy to percentage
    return x.reshape(-1, 1), y * 100


def get_test_data(method, **kwargs):
    # NOTE: For test data, currently we include CIFAR-10.1, CIFAR-10.1-C, CIFAR-10-F
    assert method in methods
    if method == "PS" or method == "ES":
        threshold = kwargs.get("threshold", None)
        assert threshold is not None
        base_dir_method = f"dataset_{used_model}_{method}_{threshold}"
    else:
        base_dir_method = f"dataset_{used_model}_{method}"
    base_dir_acc = f"dataset_{used_model}_ACC"
    x1 = np.load(f"{base_dir_method}/cifar-10.1-c.npy")
    y1 = np.load(f"{base_dir_acc}/cifar-10.1-c.npy")
    x2 = np.load(f"{base_dir_method}/cifar10-f-32.npy")
    y2 = np.load(f"{base_dir_acc}/cifar10-f-32.npy")
    x3 = np.load(f"{base_dir_method}/cifar-10.1.npy")
    y3 = np.load(f"{base_dir_acc}/cifar-10.1.npy")
    x = np.concatenate((x1, x2, x3.reshape(1)))
    y = np.concatenate((y1, y2, y3.reshape(1)))
    # convert accuracy to percentage
    return x.reshape(-1, 1), y * 100


def main():
    print(f"Using the {regressor} regressor for accuracy prediction\n")
    for method in methods:
        # initialize regressors
        if regressor == "LinearRegression":
            model = LinearRegression()
        elif regressor == "HuberRegressor":
            model = HuberRegressor()
        else:
            raise ValueError("Invalid regressor")

        if method == "PS":
            for threshold in ps_thresh:
                x_train, y_train = get_train_data(method, threshold=threshold)
                x_test, y_test = get_test_data(method, threshold=threshold)
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                error = mean_squared_error(y_test, prediction, squared=False)
                print(
                    f"Prediction score method with threshold {threshold} has test RMSE: {error}\n"
                )

        elif method == "ES":
            for threshold in es_thresh:
                x_train, y_train = get_train_data(method, threshold=threshold)
                x_test, y_test = get_test_data(method, threshold=threshold)
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                error = mean_squared_error(y_test, prediction, squared=False)
                print(
                    f"Entropy score method with threshold {threshold} has test RMSE: {error}\n"
                )
        else:
            # NOTE: two other methods
            x_train, y_train = get_train_data(method)
            x_test, y_test = get_test_data(method)
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            error = mean_squared_error(y_test, prediction, squared=False)
            method_spec = method_names[method]
            print(
                f"{method_spec} method has test RMSE: {error}\n"
            )


if __name__ == "__main__":
    main()
