import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.pyplot import MultipleLocator
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_initialize_result():
    initialize_result = {}
    for coefficient in [1, 2, 3]:
        RESULT_DIR = os.path.join(
            BASE_DIR, "offline_profiling", "data", f"result_coefficient_{coefficient}"
        )
        MAIN_RESULT = "e2e_result.csv"
        result = pd.read_csv(os.path.join(RESULT_DIR, MAIN_RESULT))
        result = result[(result["round"] != -1)]
        workflow_running_time = result[
            ["workflow_running_time", "workflow_name", "round"]
        ].drop_duplicates()
        violation_ratio = np.sum(
            workflow_running_time["workflow_running_time"] > 2
        ) / len(workflow_running_time)
        initialize_result[r"$\mu+$" + str(coefficient) + r"$\sigma$"] = violation_ratio
    return initialize_result


def get_inference_time():
    inference_time_path = os.path.join(
        BASE_DIR, "offline_profiling", "data", "inference_time.csv"
    )
    df = pd.read_csv(inference_time_path)
    return df


def plot_initialize_result(initialize_reuslt):
    plt.figure(figsize=(3, 2), dpi=120)
    sns.set_palette("deep")
    ax = sns.barplot(data=initialize_reuslt, x="init", y="value", width=0.7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    for p in plt.gca().patches:
        plt.gca().annotate(
            r"{:.2f}".format(p.get_height()),
            (p.get_x() + p.get_width() / 2, p.get_height() + 0.01),
            ha="center",
            va="center",
            xytext=(0, 4),
            textcoords="offset points",
        )
    plt.ylabel("Violation Ratio")
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    plt.xlabel("Profling Range")
    plt.ylim(0, 0.39)
    plt.yticks([0, 0.2, 0.4])
    plt.savefig(
        os.path.join(
            BASE_DIR,
            "figure",
            "initial_violation_ratio.pdf",
        ),
        bbox_inches="tight",
    )
    plt.cla()


def plot_inference_time(inference_time):
    y_major_locator = MultipleLocator(0.25)
    plt.figure(figsize=(3, 2), dpi=120)
    sns.set_palette("deep")
    sns.set_style("white")
    sns.ecdfplot(inference_time["cpu"], label="CPU")
    sns.ecdfplot(inference_time["gpu"], label="GPU", linestyle="-.")
    fig = plt.gca()
    fig.spines["right"].set_visible(False)
    fig.spines["top"].set_visible(False)
    fig.spines["left"].set_linewidth(1.5)
    fig.spines["bottom"].set_linewidth(1.5)
    fig.yaxis.set_major_locator(y_major_locator)
    plt.xlabel("MAPE")
    plt.ylabel("CDF")
    plt.xlim(0, 0.21)

    plt.xticks([0, 0.05, 0.1, 0.15, 0.2])
    plt.grid(linestyle="--", alpha=0.5)
    # plt.ylim(0,1.05)
    plt.legend(
        ncol=1,
        loc=(0.02, 0.75),
        labelspacing=0.1,
        columnspacing=0.5,
        handletextpad=0.5,
        handlelength=1.5,
        frameon=False,
    )
    plt.savefig(
        os.path.join(
            BASE_DIR,
            "figure",
            "offline-profiling-inference-time-result.pdf",
        ),
        bbox_inches="tight",
    )
    plt.cla()


initialize_result = get_initialize_result()
inference_time = get_inference_time()


plot_initialize_result(initialize_result)
plot_inference_time(inference_time)
