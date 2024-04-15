import pandas as pd
import numpy as np
from numpy import fft
import os
from tqdm import tqdm

# disable the warning output
import warnings

warnings.filterwarnings("ignore")
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    elif error_type == "over_estimation_error":
        percentage = np.sum(predicted > actual) / len(actual) * 100

    return percentage


def fourierExtrapolation(x, n_predict):
    n = x.size
    n_harm = 10  # number of harmonics in model
    t = np.arange(0, n)
    # print(t, x)
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


def parse_inter_arrival_time_series(data, func_name):
    """
    Parse the inter-arrival time data.
    Example:
        First item is always 0, left items calculates the distince of the current point to next non-zero point
    [1,0,0,1,0,0,1] => [0,2,1,0,2,1,0]
    [0,1,1,0,0,0,1] => [0,1,0,3,2,1,0]
    """
    if func_name in inter_arrival_time_series:
        return inter_arrival_time_series[func_name]
    inter_arrival_time = [0]
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[j] > 0:
                inter_arrival_time.append(j - i)
                break
    if max(inter_arrival_time) > 30:
        return []
    data = pd.Series(inter_arrival_time, name=func_name)
    inter_arrival_time_series[func_name] = data
    return data


def IceBreakerPredictor(data, interval=1):
    data = data.T.values[0]
    data = parse_inter_arrival_time_series(data, func_name)
    if len(data) < 20:
        return None, None
    data = data.T.values
    predict_values = []
    real_values = []
    for i in range(6, len(data) - 1):
        predict_value = fourierExtrapolation(np.array(data[:i]), 1)
        predict_values.append(predict_value[-1])
        real_values.append(data[i])

    mape = calculate_error(real_values, predict_values, "MAPE")
    over_estimation_error = calculate_error(
        real_values, predict_values, "over_estimation_error"
    )
    return mape, over_estimation_error


if __name__ == "__main__":
    data_path = os.path.join(SOURCE_DIR, "data", "source_file.csv")
    data = get_input_data("", data_path, all_function=True)
    inter_arrival_time_series = {}
    function_names = data["HashFunction"].values

    all_function_result = []

    for func_name in tqdm(function_names):
        func_data = data[data["HashFunction"] == func_name].drop(
            columns=["HashFunction"]
        )
        func_data = func_data.T
        func_data = (
            func_data.drop(func_data.index[0]).reset_index(drop=True).astype(int)
        )
        mape, over_estimation_error = IceBreakerPredictor(func_data)
        if mape is None:
            continue
        all_function_result.append([func_name, mape, over_estimation_error])
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
            os.path.join(SOURCE_DIR, "data", "inter_arrival_time_fip_result.csv"),
            index=False,
        )
