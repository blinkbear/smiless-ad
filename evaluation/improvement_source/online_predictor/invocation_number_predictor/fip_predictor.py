import pandas as pd
import numpy as np
from numpy import fft
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
        inverse_data = np.array([round(x / interval) * interval for x in data])
        return inverse_data


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


def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = 10  # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)  # find linear trend in x
    x_notrend = x - p[0] * t  # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)  # type: ignore # frequencies
    indexes = list(range(n))
    # sort indexes by frequency, lower -> higher
    indexes.sort(key=lambda i: np.absolute(f[i]))

    t = np.arange(1, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[: 1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n  # amplitude
        phase = np.angle(x_freqdom[i])  # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t


def IceBreakerPredictor(data, interval=1):
    data = data[data != 0].dropna()
    if len(data) < 100:
        return None, None, None
    data = data.T.values[0]
    original_test = data[6:-2]
    binned_data, interval = binning(data, interval)
    pred = []
    y = []
    for i in range(6, len(binned_data) - 1):
        data = fourierExtrapolation(np.array(binned_data[:i]), 1)
        pred.append(data[-1])
        y.append(round(binned_data[i]))
    pred = [max(x, 1) for x in pred]
    pred = inverse_bining(pred, interval)
    inverse_y = inverse_bining(y, interval)
    result = str(
        {"y": list(inverse_y), "pred": list(pred), "original_test": list(original_test)}
    )
    error_type = "MAPE"
    mape = calculate_error(y, pred, error_type, interval=interval)
    error_type = "under_estimation"
    under_estimation_error = calculate_error(y, pred, error_type)
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

    intervals = [1, 2, 4, 8]
    function_names = data["HashFunction"].values

    all_function_result = []
    for interval in tqdm(intervals):
        for func_name in tqdm(function_names):
            func_data = data[data["HashFunction"] == func_name].drop(
                columns=["HashFunction"]
            )
            func_data = func_data.T
            func_data = (
                func_data.drop(func_data.index[0]).reset_index(drop=True).astype(int)
            )
            mape, under_estimation, result = IceBreakerPredictor(func_data, interval)
            if result is None:
                continue
            all_function_result = [
                [func_name, interval, mape, under_estimation, result]
            ]
            all_function_result_df = pd.DataFrame(
                all_function_result,
                columns=[
                    "function_name",
                    "interval",
                    "mape",
                    "under_estimation",
                    "result",
                ],
            )
            save_result(
                all_function_result_df,
                f"{BASE_DIR}/data/invocation_number_fip_result.csv",
            )


if __name__ == "__main__":
    datapath = f"{BASE_DIR}/data/source_file.csv"
    get_predict_result(datapath)
