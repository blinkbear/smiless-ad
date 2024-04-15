from xgboost import XGBRegressor
import numpy as np
import pandas as pd
import os
import xgboost
from tqdm import tqdm
import warnings

xgboost.config.set_config()


warnings.filterwarnings("ignore")
SOURCE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_inter_arrival_time_series(data, func_name):
    """
    Parse the inter-arrival time data.
    Example:
        First item is always 0, left items calculates the distince of the current point to next non-zero point
    [1,0,0,1,0,0,1] => [0,2,1,0,2,1,0]
    [0,1,1,0,0,0,1] => [0,1,0,3,2,1,0]
    """
    inter_arrival_time = [0]
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[j] > 0:
                inter_arrival_time.append(j - i)
                break
    if max(inter_arrival_time) > 30:
        return []
    data = pd.Series(inter_arrival_time, name=func_name)
    return data


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


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Generate a supervised learning dataset from a given time series data.

    Parameters:
        data (array-like): The input time series data.
        n_in (int, optional): The number of lag observations as input (X). Defaults to 1.
        n_out (int, optional): The number of future observations as output (y). Defaults to 1.
        dropnan (bool, optional): Whether to drop rows with NaN values. Defaults to True.

    Returns:
        numpy.ndarray: A supervised learning dataset with input and output sequences.
    """
    df = pd.DataFrame(data)
    cols = []
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = pd.concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


def xgboost_train(train):
    """
    Trains an XGBoost regression model using the given training data.

    Parameters:
        train (list): The training data in the form of a list.

    Returns:
        XGBRegressor: The trained XGBoost regression model.
    """
    # transform list into array
    train = np.asarray(train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    ## OD: absoluteerror, mape, IR squaredlogerror,mape
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mape",
        n_estimators=20,
    )
    model.fit(trainX, trainy)
    return model


def xgboost_predict(model, testX):
    """
    Perform a one-step prediction using an XGBoost model.

    Args:
        model (xgboost.core.Booster): The trained XGBoost model.
        testX (numpy.ndarray): The input data for prediction.

    Returns:
        float: The predicted value.
    """
    # make a one-step prediction
    yhat = model.predict([testX])
    return yhat[0]


def train_test_split(data, n_test):
    """
    Splits the given data into training and testing sets.

    Parameters:
        data (numpy.ndarray): The input data to be split.
        n_test (int): The number of samples to be included in the testing set.

    Returns:
        numpy.ndarray: The training set consisting of the samples before the testing set.
        numpy.ndarray: The testing set consisting of the last n_test samples.
    """
    train_data = data[:-n_test, :]
    test_data = data[-n_test:, :]
    return train_data, test_data


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
    elif error_type == "over_estimation_error":
        percentage = np.sum(predicted > actual) / len(actual) * 100
    return percentage


def walk_forward_validation(data, n_test):
    """
    Perform walk-forward validation on a dataset.

    Parameters:
    - data: The dataset to perform validation on.
    - n_test: The number of test samples to use.
    - acc_type: The type of accuracy calculation to use.

    Returns:
    - error: The prediction error.
    - actual: The actual test values.
    - predictions: The predicted test values.
    """
    predictions = []
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = list(train)
    # print(len(history))
    # step over each time-step in the test set
    model = xgboost_train(history)
    for i in range(len(test)):
        # split test row into input and output columns
        testX, _ = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = xgboost_predict(model, testX)
        # store forecast in list of predictions
        predictions.append(max(round(yhat), 0))
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
    # estimate prediction error
    mape = calculate_error(test[:, -1], predictions, "MAPE")
    over_estimation_error = calculate_error(
        test[:, -1], predictions, "over_estimation_error"
    )
    # error = mean_absolute_percentage_error(test[:, -1], predictions)
    return mape, over_estimation_error


def XGboostPredictor(data, n_in):
    """
    Fetches the best values for 'n_in' and 'n_out' based on a given 'data_df' and evaluation metric 'acc_type'.

    Parameters:
        data_df (DataFrame): The input data frame.
        acc_type (str, optional): The evaluation metric to use. Defaults to "smape".

    Returns:
        Tuple[int, int]: A tuple containing the best values for 'n_in' and 'n_out'.
    """
    data = data.T.values[0]
    data = parse_inter_arrival_time_series(data, func_name)
    if len(data) < 160:
        return None, None
    data = data.T.values
    # Transform data
    data_trans = series_to_supervised(data, n_in=n_in, n_out=1)
    # Evaluate
    mape, over_estimation_error = walk_forward_validation(data_trans, 150)
    # Return best values
    return mape, over_estimation_error


def save_result(data, data_path):
    import os

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df = df.append(data, ignore_index=True)
        data = df.to_csv(data_path, index=False)
    else:
        data.to_csv(data_path, index=False)


if __name__ == "__main__":
    data_path = os.path.join(SOURCE_DIR, "data", "source_file.csv")
    data = get_input_data("", data_path, all_function=True)

    n_ins = [8]
    function_names = data["HashFunction"].values

    all_function_result = []
    for n_in in n_ins:
        for func_name in tqdm(function_names):
            func_data = data[data["HashFunction"] == func_name].drop(
                columns=["HashFunction"]
            )
            func_data = func_data.T
            func_data = (
                func_data.drop(func_data.index[0]).reset_index(drop=True).astype(int)
            )
            mape, over_estimation_error = XGboostPredictor(func_data, n_in)
            if mape is None or over_estimation_error is None:
                continue
            all_function_result = [[func_name, n_in, mape, over_estimation_error]]
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
                    SOURCE_DIR, "data", "inter_arrival_time_xgboost_result.csv"
                ),
                index=False,
            )
