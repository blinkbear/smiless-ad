import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.model_selection import train_test_split
import os
from tqdm import tqdm
# disable the warning output
import warnings

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


def calculate_error(actual, predicted, error_type, interval=1) -> float:
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)
    if error_type == "SMAPE":
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
    elif error_type == "over_estimation_error":
        percentage = np.sum(predicted > actual) / len(actual) * 100
    return percentage


def ARIMA_predict(data, func_name):
    data = data.T.values[0]
    data = parse_inter_arrival_time_series(data, func_name)
    if len(data) < 20:
        return None, None
    data = data.T.values

    train, test = train_test_split(data, train_size=200)
    try:
        model = pm.auto_arima(train, seasonal=True, m=12)
        # fit_model = model.fit()
        predicted_values = []
        real_values = []
        forcasts = model.predict(test.shape[0])
        predicted_values = forcasts
        real_values = test
        mape = calculate_error(real_values, predicted_values, "MAPE")
        over_estimation_error = calculate_error(
            real_values, predicted_values, "over_estimation_error"
        )
        return mape, over_estimation_error
    except Exception as e:
        print(e)
        return None, None


if __name__ == "__main__":
    data_path = os.path.join(SOURCE_DIR, "data", "source_file.csv")

    data = get_input_data("", data_path, all_function=True)
    function_names = data["HashFunction"].values
    all_function_result = []
    all_function = []

    inter_arrival_time_series = {}

    all_function_result = []
    total_count = len(data)
    count = 0
    for func_name in tqdm(function_names):
        func_data = data[data["HashFunction"] == func_name].drop(
            columns=["HashFunction"]
        )
        func_data = func_data.T
        func_data = (
            func_data.drop(func_data.index[0]).reset_index(drop=True).astype(int)
        )
        mape, over_estimation_error = ARIMA_predict(func_data, func_name)
        if mape is None:
            continue
        all_function_result.append([func_name, mape, over_estimation_error])
        all_function_result_df = pd.DataFrame(
            all_function_result,
            columns=["function_name", "mape", "over_estimation_error"],
        )
        all_function_result_df.to_csv(
            os.path.join(SOURCE_DIR, "data", "inter_arrival_time_arima_result.csv"),
            index=False,
        )
