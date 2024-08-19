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
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "co_optimizer_overhead.csv"))
    df['optimizer_names']=df['optimizer_names'].apply(lambda row: "SMIless" if len(row.split("-"))==1 else row.split("-")[1].upper())
    df['optimizer_names']=df['optimizer_names'].apply(lambda row: r"$A^\bigstar$" if row=='ASTAR' else row)
    df['optimizer_names']=df['optimizer_names'].apply(lambda row: "S-Top2" if row=='AUG' else row)
    
    ax = sns.lineplot(data=df,hue='optimizer_names', x="workflow_length", y="duration")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(1.3)
    ax.spines["bottom"].set_linewidth(1.3)
    plt.xlabel("Workflow Length")
    plt.ylabel("Time (s)")
    plt.yscale('log')
    plt.grid(True, alpha=0.5, linestyle="--")
    plt.tight_layout()
    plt.legend(
        loc=(0.01, 0.5),
        ncol=2,
        handlelength=1.1,
        handletextpad=0.1,
        columnspacing=0.6,
        title=None,
        frameon=False,
        fontsize=10, )
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "smiless_scalability.pdf",
        ),
        bbox_inches="tight",
    )


# plot_invocation_number_latency()
plot_scalability()
