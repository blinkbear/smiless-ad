import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.dirname(BASE_DIR)


def get_model_result(model_name):
    data_path = os.path.join(
        BASE_DIR, f"data/inter_arrival_time_{model_name}_result.csv"
    )
    df = pd.read_csv(data_path)
    if "lstm" in model_name:
        df = parse_lstm_result(df)
    return df


def parse_lstm_result(df):
    filter_df = df.groupby("function_name").count().reset_index()
    filter_function_name = filter_df[filter_df["n_in"] > 7]["function_name"]
    df = df[df["function_name"].isin(filter_function_name)]
    min_over_estimation_error_idx = df.groupby("function_name")[
        "over_estimation_error"
    ].transform("idxmin")
    result = df.loc[min_over_estimation_error_idx]
    result = result[["function_name", "n_in", "over_estimation_error", "mape"]]
    return result


def compares(data):
    model_names = []
    values_types = []
    accuracy = []

    for model_name in data.keys():
        model_values = data[model_name]
        if model_name == "lstm_multi":
            model_names.append("SMIless")
            values_types.append("Over Est.")
            accuracy.append(model_values["over_estimation_error"].mean())
            model_names.append("SMIless")
            values_types.append("MAPE")
            accuracy.append(model_values["mape"].mean())
        elif model_name == "lstm_single":
            model_names.append("SMIless-S")
            values_types.append("Over Est.")
            accuracy.append(model_values["over_estimation_error"].mean())
            model_names.append("SMIless-S")
            values_types.append("MAPE")
            accuracy.append(model_values["mape"].mean())

        elif model_name == "icebreaker":
            model_names.append("FIP")
            values_types.append("Over Est.")
            accuracy.append(model_values["over_estimation_error"].mean())
            model_names.append("FIP")
            values_types.append("MAPE")
            accuracy.append(model_values["mape"].mean())
        elif model_name == "arima":
            model_names.append("ARIMA")
            values_types.append("Over Est.")
            accuracy.append(model_values["over_estimation_error"].mean())
            model_names.append("ARIMA")
            values_types.append("MAPE")
            accuracy.append(model_values["mape"].mean())
        elif model_name == "xgboost":
            model_names.append("XGBoost")
            values_types.append("Over Est.")
            accuracy.append(model_values["over_estimation_error"].mean())
            model_names.append("XGBoost")
            values_types.append("MAPE")
            accuracy.append(model_values["mape"].mean())
    plt.figure(figsize=(3, 2), dpi=120)
    average_acc = pd.DataFrame(
        {"models": model_names, "accuracy": accuracy, "values_type": values_types}
    )
    sns.set_style("white")
    sns.set_palette("deep")

    sns.barplot(
        data=average_acc,
        hue="models",
        y="accuracy",
        x="values_type",
        width=0.9,
        zorder=1,
    )
    fig = plt.gca()
    fig.spines["right"].set_visible(False)
    fig.spines["top"].set_visible(False)
    fig.spines["left"].set_linewidth(1.3)
    fig.spines["bottom"].set_linewidth(1.3)
    count = 0
    for p in plt.gca().patches:
        count = count + 1
        if count > len(model_names):
            continue
        plt.gca().annotate(
            "{:.2f}".format(p.get_height(), 2),
            (p.get_x() + p.get_width() / 2.0, p.get_height() * 1.02),
            ha="center",
            va="center",
            xytext=(0, 10),
            rotation=75,
            fontsize=8,
            textcoords="offset points",
        )
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    plt.ylabel("Error (%)")
    plt.xlabel("")
    plt.xticks(rotation=0)
    plt.ylim(0, 170)
    plt.legend(
        title="",
        frameon=False,
        loc=(0.01, 0.75),
        fontsize=9,
        ncols=3,
        handlelength=0.6,
        handletextpad=0.1,
        columnspacing=0.5,
        labelspacing=0.3,
    )
    plt.tight_layout()

    plt.savefig(
        os.path.join(FIGURE_DIR, "figure", "inter_arrival_time_predict_result.pdf"),
        bbox_inches="tight",
    )


if __name__ == "__main__":
    total_result = {}
    for model_name in ["xgboost", "arima", "icebreaker", "lstm_single", "lstm_multi"]:
        data = get_model_result(model_name)
        total_result[model_name] = data
    compares(total_result)
