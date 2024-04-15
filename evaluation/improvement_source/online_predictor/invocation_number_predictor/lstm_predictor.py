import pandas as pd
import numpy as np
import warnings
import os
import tqdm

# disable the warning output
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_input_data(func, datapath, all_function=False):
    data = pd.read_csv(datapath)
    data = data.drop(["invocation_number"], axis=1)
    # change the column to row
    if not all_function:
        data = data.T
        # use the first row as the column name and drop the first row
        data.columns = data.iloc[0]
        data = data.drop(data.index[0]).reset_index(drop=True)
        return data[func]
    else:
        return data


def binning(original, interval):
    if interval == 0:
        return original, interval
    else:
        discretizated_data = np.ceil(original / interval) * interval
        return discretizated_data, interval


def inverse_bining(data, interval):
    if interval == 0:
        return data
    else:
        inverse_data = np.array([max(round(x / interval), 1) * interval for x in data])
        return inverse_data


def scale_data(data):
    from sklearn.preprocessing import StandardScaler

    # from sklearn.preprocessing import MinMaxScaler
    # scaler = StandardScaler()
    # data = data.reshape(1,-1)
    scaler = StandardScaler()
    # scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    # scaled_train_y = scaler.transform(train_y)

    # scaled_train_x1 = MinMaxScaler(feature_range=[ , ])
    # scaled_train_x1 = scaled_train_x1.fit_transform(train_x)
    return scaler, scaled_data


def inverse_scale_data(scaler, data):
    if len(data.shape) == 1:
        data = data.reshape(1, -1)
    data = scaler.inverse_transform(data)
    data = [int(x) for x in data[0]]
    return data


def calculate_error(actual, predicted, error_type, interval=1) -> float:
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)
    if error_type == "MAPE":
        percentage = round(
            np.mean(
                [
                    np.abs(x - y) / ((np.abs(x) + np.abs(y)) / 2)
                    for x, y in zip(actual, predicted)
                    if x != 0 or y != 0
                ]
                + [0 for x, y in zip(actual, predicted) if x == 0 and y == 0]
            )
            * 100,
            2,
        )
    # # calculate the percentage that the predicted value is less than the actual
    elif error_type == "under_estimation":
        percentage = np.sum(predicted < actual) / len(actual) * 100
    elif error_type == "container_number_estimation_error":
        container_number_errors = []
        for i in range(len(actual)):
            actual_container = actual[i] / interval
            predicted_container = predicted[i] / interval
            container_number_errors.append((predicted_container - actual_container))
        percentage = np.median(container_number_errors)

    elif error_type == "error_classification":
        error_count = 0
        for i in range(len(actual)):
            if predicted[i] != actual[i]:
                error_count += 1
        percentage = error_count / len(actual) * 100
    return percentage


import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as datautil


class AirModel(nn.Module):
    def __init__(self, n_in, n_out):
        super(AirModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_in, hidden_size=n_in, num_layers=1, batch_first=True
        )
        self.linear = nn.Linear(n_in, n_out, bias=False)

    def forward(self, x):
        x, _ = self.lstm(x)
        out = self.linear(x)
        return out


def create_dataset(input_time_series, n_in=1, n_out=1, return_tensor=True):
    X, y = [], []
    for i in range(len(input_time_series) - n_in - n_out):
        X.append(input_time_series[i : i + n_in])
        y.append(input_time_series[i + n_in : i + n_in + n_out])
    if return_tensor:
        return torch.tensor(np.array(X)).to(torch.float32), torch.tensor(
            np.array(y)
        ).to(torch.float32)
    else:
        return np.array(X), np.array(y)


def split_data(data, n_test):
    train_data = data[:-n_test][:]
    test_data = data[-n_test:][:]
    return train_data, test_data


def get_learning_rate(data):
    data = np.array(data[:-1])
    data = data[np.nonzero(data)]
    logs = np.ceil(np.max(np.log10(list(data))))
    if logs == 0 or logs == 1:
        return 0.001
    return 10 ** (-logs)


losses = []


def train_model(
    n_in, n_out, learning_rate, train_x, train_y, device, model_type="LSTM"
):
    if model_type == "LSTM":
        model = AirModel(n_in, n_out)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.L1Loss()
        loader = datautil.DataLoader(
            datautil.TensorDataset(train_x, train_y), batch_size=8
        )
        patience = 4
        best_val_loss = float("inf")
        early_stopping_counter = 0
        for epoch in range(500):
            val_loss = []

            for x, y in loader:
                model.train()
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                val_loss.append(loss.item())

            # 计算平均验证集损失
            avg_val_loss = np.mean(val_loss)

            # 判断是否早期停止
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # 判断是否触发早期停止
            if early_stopping_counter >= patience:
                break

        return model
    elif model_type == "random_forest_classifier":
        from sklearn.ensemble import RandomForestClassifier

        # fit model
        if isinstance(train_x, torch.Tensor):
            train_x = train_x.numpy()
        if isinstance(train_y, torch.Tensor):
            train_y = train_y.numpy()
        model = RandomForestClassifier(n_estimators=50)
        model.fit(train_x, train_y)
        return model


def handling_data(
    func,
    n_test,
    n_in,
    n_out,
    interval,
    datapath,
    trace_data=None,
    use_discretization=True,
    non_zero=True,
    return_tensor=True,
    scale=False,
    device="cpu",
):
    scaler = None
    if trace_data is None:
        trace_data = get_input_data(func, datapath)
    if non_zero:
        trace_data = trace_data[trace_data != 0]
        trace_data = trace_data.dropna().reset_index(drop=True)
    if n_test > len(trace_data):
        # print("n_test should be less than the length of trace data, default to 20%")
        n_test = int(len(trace_data) * 0.2)
    _, original_trace_test_data = split_data(trace_data, n_test)
    if use_discretization:
        binned_trace_data, train_interval = binning(trace_data, interval)
    if len(binned_trace_data) < n_in:
        return None, None, None, None, None, None, None, None, None
    learning_rate = get_learning_rate(binned_trace_data)
    if scale:
        scaler, scaled_trace_data = scale_data(binned_trace_data)
    train_data, test_data = split_data(scaled_trace_data, n_test)
    original_train_data = np.array(train_data).reshape(-1)
    original_test_data = np.array(test_data)[:-1].reshape(-1)
    if len(original_test_data) <= n_in or len(original_train_data) <= 16 * n_in:
        return None, None, None, None, None, None, None, None, None

    func_train_trace = list(original_train_data)
    func_test_trace = list(original_test_data)
    train_x, train_y = create_dataset(
        func_train_trace, n_in, n_out, return_tensor=return_tensor
    )
    test_x, test_y = create_dataset(
        func_test_trace, n_in, n_out, return_tensor=return_tensor
    )
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    return (
        train_x,
        train_y,
        test_x,
        test_y,
        train_interval,
        train_interval,
        learning_rate,
        scaler,
        original_trace_test_data.T.values,
    )


def get_prediction_result(
    model,
    X,
    y,
    original_test_data,
    scale=False,
    use_discretization=True,
    interval=4,
    return_result=True,
    scaler=None,
):
    with torch.no_grad():
        pred = model(X)
    if pred.device != "cpu":
        pred = torch.max(pred, dim=1).values.cpu().detach().numpy().flatten()
        y = y[:, 0].cpu().detach().numpy().flatten()
    else:
        pred = torch.max(pred, dim=1).values.detach().numpy().flatten()
        y = y[:, 0].detach().numpy().flatten()
    if scale:
        pred = inverse_scale_data(scaler, pred)
        y = inverse_scale_data(scaler, y)
    if use_discretization:
        pred = inverse_bining(pred, interval)
        inverse_y = inverse_bining(y, interval)
    result = str(
        {
            "y": list(inverse_y),
            "pred": list(pred),
            "original": list(original_test_data),
        }
    )
    error_type = "MAPE"
    mape = calculate_error(inverse_y, pred, error_type, interval=interval)
    error_type = "under_estimation"
    under_estimation_error = calculate_error(inverse_y, pred, error_type)
    if not return_result:
        result = None
    return mape, under_estimation_error, result


def save_result(data, data_path):
    import os

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = df.append(data, ignore_index=True)
        data = df.to_csv(data_path, index=False)
    else:
        data.to_csv(data_path, index=False)


def get_predict_result(datapath):
    data = get_input_data("", datapath, all_function=True)
    function_names = data["HashFunction"].values
    n_out = 2
    n_test = 150
    model_type = "LSTM"
    all_function_result = []
    all_function = []
    intervals = [1, 2, 4, 8]
    n_ins = [i for i in range(10, 15)]
    for n_in in tqdm(n_ins):
        for interval in tqdm(intervals):
            for func_name in tqdm(function_names):
                func_data = data[data["HashFunction"] == func_name]
                func_data = func_data.T
                func_data = func_data.drop(func_data.index[0]).reset_index(drop=True)
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

                (
                    train_x,
                    train_y,
                    test_x,
                    test_y,
                    train_interval,
                    test_interval,
                    learning_rate,
                    scaler,
                    original_test_data,
                ) = handling_data(
                    "",
                    n_test,
                    n_in,
                    n_out,
                    interval,
                    datapath,
                    trace_data=func_data,
                    return_tensor=True,
                    non_zero=True,
                    scale=True,
                    device=device,
                )
                if train_x is None:
                    continue
                all_function.append(func_name)
                model = train_model(
                    n_in,
                    n_out,
                    learning_rate,
                    train_x,
                    train_y,
                    device,
                    model_type=model_type,
                )

                (mape, under_estimation_error, result) = get_prediction_result(
                    model,
                    test_x,
                    test_y,
                    original_test_data=original_test_data,
                    scale=True,
                    use_discretization=True,
                    interval=interval,
                    return_result=True,
                    scaler=scaler,
                )
                if result is not None:
                    all_function_result = [
                        [
                            func_name,
                            n_in,
                            interval,
                            mape,
                            under_estimation_error,
                            result,
                        ]
                    ]
                all_function_result_df = pd.DataFrame(
                    all_function_result,
                    columns=[
                        "function_name",
                        "n_in",
                        "interval",
                        "mape",
                        "under_estimation_error",
                        "result",
                    ],
                )
                save_result(
                    all_function_result_df,
                    f"{BASE_DIR}/data/invocation_number_lstm_result.csv",
                )


if __name__ == "__main__":
    datapath = f"{BASE_DIR}/data/source_file.csv"
    get_predict_result(datapath)
