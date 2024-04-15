import pandas as pd
import numpy as np
import warnings
import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as datautil
import argparse
import os
from tqdm import tqdm

warnings.filterwarnings("ignore")
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_input_data(func, datapath, all_function=False):
    data = pd.read_csv(datapath)
    # change the column to row
    if not all_function:
        data = data.T
        # use the first row as the column name and drop the first row
        data.columns = data.iloc[0]
        data = data.drop(data.index[0]).reset_index(drop=True)
        data = data[func].values
        return data
    else:
        return data


def data_scaler(data):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler1 = MinMaxScaler()

    if input_type == "single":
        scaler.fit(np.reshape(data.values, (-1, 1)))
        scaled_data = scaler.transform(np.reshape(data.values, (-1, 1)))
        scaled_data = pd.DataFrame(scaled_data, columns=["inter_arrival_time"])
    elif input_type == "multi":
        scaling_inter_arrival_time = np.array(data["inter_arrival_time"]).reshape(-1, 1)
        scaler.fit(scaling_inter_arrival_time)
        scaling_invocation_number = np.array(data["invocation_number"]).reshape(-1, 1)
        scaler1.fit(scaling_invocation_number)
        scaled_inter_arrival_time = scaler.transform(scaling_inter_arrival_time)
        scaled_invocation_number = scaler1.transform(scaling_invocation_number)
        scaled_data = np.concatenate(
            (scaled_inter_arrival_time, scaled_invocation_number), axis=1
        )
        scaled_data = pd.DataFrame(
            scaled_data, columns=["inter_arrival_time", "invocation_number"]
        )

    return scaler, scaled_data


def inverse_scale_data(scaler, data):
    data = scaler.inverse_transform([data])
    data = [round(x) for x in data[0]]
    return data


def parse_inter_arrival_time_series(data, func_name, input_type):
    """
    Parse the inter-arrival time data.
    Example:
        First item is always 0, left items calculates the distince of the current point to next non-zero point
    [1,0,0,1,0,0,1] => [3,2,1,3,2,1]
    [0,1,1,0,0,0,1] => [1,1,4,3,2,1]
    """
    global inter_arrival_time_series, invocation_number_series
    if func_name in inter_arrival_time_series:
        return inter_arrival_time_series[func_name], invocation_number_series[func_name]
    inter_arrival_time = []
    invocation_number = []
    non_zero_loc = np.where(data > 0)
    inter_arrival_time = [non_zero_loc[0][0]] + list(np.diff(non_zero_loc)[0])
    invocation_number = list(data[non_zero_loc])
    if input_type == "single":
        data = pd.DataFrame(inter_arrival_time, columns=["inter_arrival_time"])
        invocation_number = None
    elif input_type == "multi":
        updated_data = []
        for i in range(len(inter_arrival_time)):
            updated_data.append([invocation_number[i], inter_arrival_time[i]])
        data = pd.DataFrame(
            updated_data, columns=["invocation_number", "inter_arrival_time"]
        )
    if max(data["inter_arrival_time"]) > 30:
        return None, None
    return data, invocation_number


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
    elif error_type == "over_estimate_mape":
        percentage = round(
            np.mean(
                [
                    np.abs(x - y) / ((np.abs(x) + np.abs(y)) / 2)
                    for x, y in zip(actual, predicted)
                    if x < y
                ]
                + [0 for x, y in zip(actual, predicted) if x >= y]
            )
            * 100,
            2,
        )
    elif error_type == "over_estimation_error":
        percentage = np.sum(predicted > actual) / len(actual) * 100
    return percentage


class AirModel(nn.Module):
    def __init__(self, n_in, n_out):
        super(AirModel, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=n_in, hidden_size=128, num_layers=1, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=n_in, hidden_size=128, num_layers=1, batch_first=True
        )
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(128, 128, bias=False)
        self.linear2 = nn.Linear(128, n_out, bias=False)

    def forward(self, x):
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        x1, _ = self.lstm1(x1)
        x2, _ = self.lstm2(x2)
        x2 = x1 + x2
        x2 = self.linear1(x2)
        x2 = self.linear2(x2)
        out = self.tanh(x2)

        return out


class AirModelOriginal(nn.Module):
    def __init__(self, n_in, n_out):
        super(AirModelOriginal, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=n_in, hidden_size=n_in, num_layers=1, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=n_in, hidden_size=n_in, num_layers=1, batch_first=True
        )
        self.linear = nn.Linear(n_in, n_out, bias=False)

    def forward(self, x):
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x[-1:])
        out = self.linear(x1 + x2)
        return out


def create_dataset(input_time_series, n_in=1, n_out=1, return_tensor=True):
    X, y = [], []
    for i in range(len(input_time_series) - n_in - n_out):
        if input_type == "single":
            x = np.array(input_time_series[i : i + n_in]).reshape(-1)
        else:
            x = input_time_series[i : i + n_in]
        X.append(x)
        y.append(input_time_series[i + n_in : i + n_in + n_out]["inter_arrival_time"])

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
    if logs < 2:
        return 0.01
    return 10 ** (-logs)


def train_model(n_in, n_out, learning_rate, train_x, train_y, device):
    if input_type == "single":
        model = AirModelOriginal(n_in, n_out)
    elif input_type == "multi":
        model = AirModel(n_in, n_out)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()
    loader = datautil.DataLoader(
        datautil.TensorDataset(train_x, train_y), batch_size=16
    )
    patience = 8
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
        losses.append(avg_val_loss)
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


def get_original_y(data, n_test, n_in, n_out, return_tensor=True):
    train_data, test_data = split_data(data, n_test)
    test_x, test_y = create_dataset(
        test_data[:-1], n_in, n_out, return_tensor=return_tensor
    )
    train_x, train_y = create_dataset(
        train_data[:], n_in, n_out, return_tensor=return_tensor
    )
    train_y = np.array(train_y).reshape(-1)
    test_y = np.array(test_y).reshape(-1)
    return train_y, test_y


def handle_data(
    func,
    n_in,
    n_out,
    datapath,
    trace_data=None,
    return_tensor=True,
    scale=False,
    device="cpu",
):
    scaler = None
    if trace_data is None:
        trace_data = get_input_data(func, datapath)
    trace_data = trace_data.T.values[0]
    transfered_trace_data, invocation_number = parse_inter_arrival_time_series(
        trace_data, func, input_type
    )
    if transfered_trace_data is None:
        return None, None, None, None, None, None

    n_test = int(len(transfered_trace_data) * 0.2)
    learning_rate = get_learning_rate(transfered_trace_data)
    global original_train_y, original_test_y
    original_train_y, original_test_y = get_original_y(
        transfered_trace_data, n_test, n_in, n_out, return_tensor=return_tensor
    )

    if scale:
        scaler, transfered_trace_data = data_scaler(transfered_trace_data)
    train_data, test_data = split_data(transfered_trace_data, n_test)
    if len(test_data) <= n_in or len(train_data) <= 16 * n_in:
        return None, None, None, None, None, None

    train_x, train_y = create_dataset(
        train_data, n_in, n_out, return_tensor=return_tensor
    )
    test_x, test_y = create_dataset(
        test_data[:-1], n_in, n_out, return_tensor=return_tensor
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
        learning_rate,
        scaler,
    )


def get_prediction_result(
    model,
    X,
    y,
    train_stage=True,
    train_error=0.0,
    scale=False,
    return_result=False,
    scaler=None,
):
    global inter_arrival_time_series
    with torch.no_grad():
        pred = model(X)
    if pred.device != "cpu":
        pred = torch.max(pred, dim=1).values.cpu().detach().numpy().flatten()
        if scale:
            pred = inverse_scale_data(scaler, pred)
        if not train_stage:
            pred = [x * (1 - train_error / 100) for x in pred]
        pred = [round(x) if x > 1 else 1 for x in pred]
        if train_stage:
            y = original_train_y
        else:
            y = original_test_y
    else:
        pred = torch.max(pred, dim=1).values.detach().numpy().flatten()
        y = y[:, 1].detach().numpy().flatten()
        if scale:
            pred = inverse_scale_data(scaler, pred)
            y = inverse_scale_data(scaler, y)

        pred = [round(x) if x > 0 else 0 for x in pred]
    error_type = "MAPE"
    mape = calculate_error(y[-100:], pred[-100:], error_type)
    if train_stage:
        error_type = "over_estimate_mape"
        over_error = calculate_error(y[-100:], pred[-100:], error_type)
    else:
        error_type = "over_estimate"
        over_error = calculate_error(
            y[-100:],
            pred[-100:],
            error_type,
        )
    if not return_result:
        pred = None
    return mape, over_error, y, pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_type",
        type=str,
        default="multi",
        help="input type: multi or single",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_type = args.input_type
    datapath = os.path.join(SOURCE_DIR, "data", "source_file.csv")
    data = get_input_data("", datapath, all_function=True)
    losses = []
    function_names = data["HashFunction"].values
    n_in, n_out = 1, 1
    function_best_n_in = {}
    function_mapes = {}
    all_function_result = []
    all_function = []
    all_function_predict_values = []
    inter_arrival_time_series = {}
    invocation_number_series = {}
    start_n_in = 4
    end_n_in = 19
    scale_data = True
    for n_in in tqdm([i for i in range(start_n_in, end_n_in)], position=1):
        for func_name in tqdm(function_names, position=0):
            func_data = data[data["HashFunction"] == func_name]
            func_data = func_data.T
            func_data = func_data.drop(func_data.index[0]).reset_index(drop=True)
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            original_train_y, original_test_y = [], []
            (train_x, train_y, test_x, test_y, learning_rate, scaler) = handle_data(
                func_name,
                n_in,
                n_out,
                datapath,
                trace_data=func_data,
                return_tensor=True,
                scale=scale_data,
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
            )
            train_mape, train_over_error, y, train_pred = get_prediction_result(
                model,
                train_x,
                train_y,
                train_stage=True,
                train_error=0,
                scale=scale_data,
                return_result=True,
                scaler=scaler,
            )
            test_mape, test_over_error, test_under_error, y, test_pred = (
                get_prediction_result(
                    model,
                    test_x,
                    test_y,
                    train_stage=False,
                    train_error=train_mape,
                    scale=scale_data,
                    return_result=True,
                    scaler=scaler,
                )
            )
            if test_mape is not None:
                all_function_result.append(
                    [func_name, n_in, test_mape, test_over_error, test_under_error]
                )
                all_function_predict_values.append([func_name, n_in, y, test_pred])

            all_function_result_df = pd.DataFrame(
                all_function_result,
                columns=[
                    "function_name",
                    "n_in",
                    "mape",
                    "over_estimation_error",
                ],
            )
            all_function_result_df.to_csv(
                os.path.join(
                    SOURCE_DIR,
                    "data",
                    f"inter_arrival_time_lstm_{input_type}_result.csv",
                ),
                index=False,
            )
