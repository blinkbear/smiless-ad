import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from resources.auto_scaler import AutoScaler
from resources.function_profiler import FunctionProfiler

import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.dirname(BASE_DIR)


def calculate_smape(actual, predicted) -> float:
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual), np.array(predicted)

    return round(
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


def get_container_number_error(function_name, true_value, pred_value, interval):
    true_value = eval(true_value)
    pred_value = eval(pred_value)
    cpu_true_container_number = []
    cpu_pred_container_number = []
    gpu_true_container_number = []
    gpu_pred_container_number = []
    if len(true_value) != len(pred_value):
        if len(true_value) > len(pred_value):
            true_value = true_value[len(true_value) - len(pred_value) :]
        else:
            return pd.Series(
                {
                    "cpu_container_number_error": -1,
                    "gpu_container_number_error": -1,
                }
            )
    pred_value = [max(x, 1) * interval for x in pred_value]
    true_value = [max(x, 1) * interval for x in true_value]
    for i in range(len(true_value)):
        cpu_true_container_number.append(
            autoscaler.get_container_number(true_value[i], function_name, "cpu")
        )
        cpu_pred_container_number.append(
            autoscaler.get_container_number(pred_value[i], function_name, "cpu")
        )
        gpu_true_container_number.append(
            autoscaler.get_container_number(true_value[i], function_name, "gpu")
        )
        gpu_pred_container_number.append(
            autoscaler.get_container_number(pred_value[i], function_name, "gpu")
        )
    cpu_container_number_error = calculate_smape(
        cpu_true_container_number, cpu_pred_container_number
    )
    gpu_container_number_error = calculate_smape(
        gpu_true_container_number, gpu_pred_container_number
    )
    return pd.Series(
        {
            "cpu_container_number_error": cpu_container_number_error,
            "gpu_container_number_error": gpu_container_number_error,
        }
    )


def get_under_estimation_error(true_value, pred_value):
    true_value = eval(true_value)
    if len(true_value) == 1:
        true_value = true_value[0]
    pred_value = eval(pred_value)
    if len(true_value) != len(pred_value):
        if len(true_value) > len(pred_value):
            true_value = true_value[len(true_value) - len(pred_value) :]
        else:
            pred_value = pred_value[len(pred_value) - len(true_value) :]
    count = 0
    for i in range(len(true_value)):
        if true_value[i] > pred_value[i]:
            count += 1
    under_estimation_error = count / len(true_value) * 100
    return pd.Series(
        {
            "under_estimation_error": under_estimation_error,
        }
    )


def get_original_and_result(result):
    result = (
        result.replace("array(", "")
        .replace(")", "")
        .replace(", dtype=object", "")
        .replace("dtype=object,", "")
        .replace("dtype=object", "")
    )
    original = eval(result)["y"]
    pred = eval(result)["pred"]
    return pd.Series({"original_data": str(original), "result": str(pred)})


def filter_functions(df):
    functions = set()
    for interval in df["interval"].unique():
        function_name = df[df["interval"] == interval]["function_name"].unique()
        if len(functions) == 0:
            functions = set(function_name)
        else:
            functions = functions & set(function_name)
    df = df[df["function_name"].isin(functions)]
    return df


def get_container_number_data(df, model_name):
    df = df[~df["result"].str.contains("...,", regex=False)]
    df[["original_data", "result"]] = df.apply(
        lambda x: get_original_and_result(x["result"]), axis=1
    )
    df[["cpu_container_number_error", "gpu_container_number_error"]] = df.apply(
        lambda x: get_container_number_error(
            x["function_name"], x["original_data"], x["result"], x["interval"]
        ),
        axis=1,
    )
    df["under_estimation_error"] = df["under_estimation_error"]
    df = df[df["cpu_container_number_error"] != -1]
    df = df.groupby(["function_name", "interval"]).agg(
        {
            "cpu_container_number_error": "min",
            "gpu_container_number_error": "min",
            "under_estimation_error": "min",
        }
    )
    if "n_in" in df.columns:
        result = (
            df.groupby(["interval"])
            .agg(
                {
                    "cpu_container_number_error": "mean",
                    "gpu_container_number_error": "mean",
                    "under_estimation_error": "mean",
                }
            )
            .reset_index()
        )
    else:
        result = (
            df.groupby(["interval"])
            .agg(
                {
                    "cpu_container_number_error": "mean",
                    "gpu_container_number_error": "mean",
                    "under_estimation_error": "mean",
                }
            )
            .reset_index()
        )
    transform_horizon_to_vertical(result, model_name)


def transform_horizon_to_vertical(df: pd.DataFrame, model_name):
    df = df.melt(id_vars="interval", var_name="type", value_name="value")
    df["model"] = model_names[model_name]
    global total_result
    total_result = pd.concat([total_result, df], axis=0)


def plot_min_result(total_result):
    plt.figure(figsize=(3, 2), dpi=120)

    # sns.set(font_scale=1.02)
    sns.set_style("white")
    sns.set_palette("deep")
    df_for_plot = total_result[total_result["interval"] == 8]
    df_for_plot["model"].replace({"Icebreaker": "FIP"}, inplace=True)
    df_for_plot["model"].replace({"LSTM": "SMIless"}, inplace=True)
    df_for_plot["type"].replace(
        {
            "cpu_container_number_error": "CPU",
            "gpu_container_number_error": "GPU",
            "under_estimation_error": "Under Est.",
        },
        inplace=True,
    )
    ax = sns.barplot(
        df_for_plot,
        x="type",
        y="value",
        hue="model",
        hue_order=["XGBoost", "ARIMA", "FIP", "SMIless"],
    )
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.left.set_linewidth(1.3)
    ax.spines.bottom.set_linewidth(1.3)
    for p in plt.gca().patches:
        if p.get_height() == 0:
            continue
        plt.gca().annotate(
            "{:.2f}".format(p.get_height()),
            (p.get_x() + p.get_width() / 2.0, p.get_height() * 0.95),
            ha="center",
            va="center",
            xytext=(0, 10),
            fontsize=8,
            rotation=75,
            textcoords="offset points",
        )
    plt.legend(
        loc=(0.01, 0.75),
        ncol=2,
        fontsize=9,
        handlelength=0.6,
        handletextpad=0.1,
        columnspacing=0.5,
        labelspacing=0.3,
        frameon=False,
    )
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    # plt.xticks(rotation=10)
    plt.xlabel("")
    plt.ylabel("Error (%)")
    plt.ylim(0, 14)
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURE_DIR, "figure", "invocation_number_predict_result.pdf")
    )
    plt.show()
    plt.cla()


def get_filter_function(df):
    global function_names
    func_names = list(df["function_name"].unique())
    if len(function_names) == 0:
        function_names = set(func_names)
    else:
        function_names = function_names.intersection(set(func_names))


def get_total_result():
    function_profile_path = "/root/openfaas/benchmarks/invoke/inference/opt/result.csv"
    function_profiler = FunctionProfiler(
        function_profile_result_path=function_profile_path
    )
    global autoscaler
    autoscaler = AutoScaler(function_profiler)
    global function_names
    function_names = set()
    total_result = pd.DataFrame()
    for model in ["icebreaker", "arima", "xgboost", "lstm"]:
        data_path = os.path.join(BASE_DIR, f"data/invocation_number{model}_result.csv")
        df = pd.read_csv(data_path)
        get_filter_function(df)
        get_container_number_data(df, model)
    return total_result


if __name__ == "__main__":
    model_names = {
        "lstm": "LSTM",
        "icebreaker": "IceBreaker",
        "arima": "ARIMA",
        "xgboost": "XGBoost",
    }
    total_result = get_total_result()
    plot_min_result(total_result)
