import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import os

from matplotlib.pyplot import MultipleLocator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(BASE_DIR, "figure/")


def plot_invocation_number_latency():
    plt.figure(figsize=(3, 2), dpi=120)
    sns.set_style("white")
    sns.set_palette("deep")
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "invocation_number_latency.csv"))
    ax = sns.lineplot(x="invocation_number", y="latency", data=df)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    y_major_locator = MultipleLocator(0.01)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0.01, 0.05)
    plt.xlabel("Invocation Number")
    plt.ylabel("Time (ms)")
    plt.xlim(0, 1050)
    plt.grid(alpha=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "invocation_number_latency.pdf",
        ),
        bbox_inches="tight",
    )
    plt.cla()


def plot_scalability():
    plt.figure(figsize=(3, 2), dpi=120)
    sns.set_palette("deep")
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "smiless_scalability_result.csv"))
    ax = sns.lineplot(data=df, x="workflow_length", y="time")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    plt.ylim(0.6, 1)
    plt.xlabel("Workflow Length")
    plt.ylabel("Time (ms)")
    plt.grid(True, alpha=0.5, linestyle="--")
    plt.tight_layout()
    # plt.legend().remove()
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "smiless_scalability.pdf",
        ),
        bbox_inches="tight",
    )


plot_invocation_number_latency()
plot_scalability()
