import numpy as np
import pandas as pd
import pmdarima as pm
from pmdarima.model_selection import train_test_split
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
    elif error_type == "under_estimation_error":
        percentage = np.sum(predicted < actual) / len(actual) * 100

    return percentage


def calculate_bound_error(actual, predicted):
    errors = 0
    for i in range(len(actual)):
        if predicted[i] < actual[i]:
            errors += 1
    return errors / len(actual)


def ARIMA_predict(data, interval=1):
    data = data[data != 0].dropna()
    if len(data) < 150:
        return None, None, None

    data = data.T.values[0]
    original_test = data[200:]
    binned_data, interval = binning(data, interval)

    train, test = train_test_split(binned_data, train_size=200)
    try:
        model = pm.auto_arima(train, seasonal=True, m=12)
        predicted_values = []
        y = []
        forcasts = model.predict(test.shape[0])
        predicted_values = [inverse_bining(x, interval) for x in forcasts]
        y = [inverse_bining(x, interval) for x in test]
    except Exception as e:
        print(e)
        return None, None
    # # 拟合ARIMA模型
    result = str(
        {
            "y": list(y),
            "pred": list(predicted_values),
            "original_test": list(original_test),
        }
    )
    error_type = "MAPE"
    mape = calculate_error(y, predicted_values, error_type, interval=interval)
    error_type = "under_estimation_error"
    under_estimation_error = calculate_error(y, predicted_values, error_type)
    return mape, under_estimation_error, result


def save_result(result, datapath):
    import os

    if os.path.exists(datapath):
        df = pd.read_csv(datapath)
        df = df.append(result, ignore_index=True)
        result = df.to_csv(datapath, index=False)
    else:
        result.to_csv(datapath, index=False)


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
            try:
                mape, under_estimation, result = ARIMA_predict(func_data, interval)
            except:  # noqa: E722
                print(f"error: {func_name} {interval}, continue")
                continue
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
                f"{BASE_DIR}/data/invocation_number_arima_result.csv",
            )


if __name__ == "__main__":
    datapath = f"{BASE_DIR}/data/source_file.csv"
    get_predict_result(datapath)
