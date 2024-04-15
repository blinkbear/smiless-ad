import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os
import copy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.join(os.path.dirname(BASE_DIR), "figure/")


def get_cost(result, SLAs):
    result = result[result["sla"].isin(SLAs)]
    running_performance = result[
        [
            "optimizer",
            "workflow_name",
            "workflow_running_time",
            "function_name",
            "running_cost",
            "keep_alive_cost",
            "model_load_cost",
            "sla",
            "round",
        ]
    ]
    running_performance.loc[:, "cost"] = (
        running_performance["running_cost"] + running_performance["keep_alive_cost"]
    )
    running_performance.loc[:, "violation"] = (
        running_performance["workflow_running_time"] > running_performance["sla"]
    )
    running_status = result[
        [
            "optimizer",
            "process_time",
            "workflow_name",
            "pod_name",
            "function_name",
            "running_cost",
            "keep_alive_cost",
            "total_cost",
            "sla",
            "round",
        ]
    ]
    function_load_costs = (
        result[
            [
                "optimizer",
                "workflow_name",
                "function_name",
                "pod_name",
                "model_load_cost",
                "sla",
            ]
        ]
        .drop_duplicates(
            subset=[
                "optimizer",
                "workflow_name",
                "function_name",
                "pod_name",
                "sla",
            ]
        )
        .groupby(["optimizer", "workflow_name", "function_name", "sla"])
        .sum()
        .reset_index()
    )

    function_running_costs = (
        running_status[
            [
                "optimizer",
                "workflow_name",
                "pod_name",
                "function_name",
                "running_cost",
                "sla",
                "round",
            ]
        ]
        .drop_duplicates(
            subset=[
                "optimizer",
                "pod_name",
                "function_name",
                "running_cost",
                "sla",
                "round",
            ]
        )
        .groupby(["optimizer", "workflow_name", "function_name", "sla"])
        .sum()
        .reset_index()[
            [
                "optimizer",
                "workflow_name",
                "function_name",
                "running_cost",
                "sla",
            ]
        ]
    )
    function_running_costs = pd.merge(
        function_running_costs,
        function_load_costs,
        on=["optimizer", "workflow_name", "function_name", "sla"],
    )
    function_keep_alive_cost = (
        running_status[
            [
                "optimizer",
                "workflow_name",
                "pod_name",
                "function_name",
                "keep_alive_cost",
                "sla",
            ]
        ]
        .drop_duplicates(
            subset=[
                "optimizer",
                "pod_name",
                "function_name",
                "keep_alive_cost",
                "sharing",
                "sla",
            ]
        )
        .groupby(["optimizer", "workflow_name", "pod_name", "sla"])
        .max()
        .reset_index()
        .groupby(["optimizer", "workflow_name", "function_name", "sla"])
        .sum()
        .reset_index()[
            [
                "optimizer",
                "workflow_name",
                "function_name",
                "keep_alive_cost",
                "sla",
            ]
        ]
    )
    function_total_costs = pd.merge(
        function_running_costs,
        function_keep_alive_cost,
        on=["optimizer", "workflow_name", "function_name", "sla"],
    )
    function_total_costs["total_cost"] = (
        function_total_costs["running_cost"]
        + function_total_costs["keep_alive_cost"]
        + function_total_costs["model_load_cost"]
    )
    total_cost = (
        function_total_costs.groupby(["optimizer", "workflow_name", "sla"])
        .sum()
        .reset_index()[["optimizer", "workflow_name", "keep_alive_cost", "sla"]]
    )
    overall_cost = (
        function_total_costs.groupby(["optimizer", "sla"])
        .sum()
        .reset_index()[["optimizer", "keep_alive_cost", "sla"]]
    )
    overall_cost["workflow_name"] = "Total"
    total_cost = pd.concat([total_cost, overall_cost])
    e2e_result = copy.copy(total_cost)
    e2e_result["type"] = "cost"
    e2e_result["value"] = e2e_result["keep_alive_cost"]
    e2e_result.drop(["keep_alive_cost"], axis=1, inplace=True)
    e2e_result_merged = pd.merge(
        e2e_result,
        e2e_result[e2e_result["optimizer"] == "OPT"][["workflow_name", "value"]],
        on=["workflow_name"],
    )
    e2e_result_merged["ratio"] = (
        e2e_result_merged["value_x"] / e2e_result_merged["value_y"]
    )
    cost = e2e_result_merged[
        e2e_result_merged["optimizer"].isin(
            ["Aquatope", "GrandSLAm", "IceBreaker", "Orion", "SMIless", "OPT"]
        )
    ]
    cost["index"] = [i for i in range(len(cost))]
    return cost


def get_violation_ratio(result, SLAs):
    result = result[result["sla"].isin(SLAs)]
    running_performance = result[
        [
            "optimizer",
            "workflow_name",
            "workflow_running_time",
            "function_name",
            "running_cost",
            "keep_alive_cost",
            "model_load_cost",
            "sla",
            "round",
        ]
    ]
    running_performance["violation"] = (
        running_performance["workflow_running_time"] > running_performance["sla"]
    )
    optimizer_violation = running_performance[
        ["optimizer", "workflow_name", "violation", "round", "sla"]
    ].drop_duplicates()[["optimizer", "violation", "sla"]]

    violation_ratio = (
        (
            optimizer_violation.groupby(["optimizer", "sla"]).sum()
            / (
                optimizer_violation.groupby(["optimizer", "sla"])
                .count()
                .reset_index()["violation"]
                .max()
            )
        )
        .reset_index()
        .rename({0: "violation"}, axis=1)
    )
    violation_ratio.loc[:, "violation"] = violation_ratio["violation"] * 100
    return violation_ratio


def get_workflow_running_time(result):
    workflow_running_time = result[
        ["optimizer", "workflow_running_time", "workflow_name", "round"]
    ].drop_duplicates()
    return workflow_running_time


def get_device_ratio(result, SLAs):
    result = result[result["sla"].isin(SLAs)]
    pod_number = (
        result[["optimizer", "sla", "function_name", "pod_name", "device"]]
        .drop_duplicates()
        .groupby(["optimizer", "sla", "device"])
        .count()
        .reset_index()
    )
    device_ratio = (
        pod_number[["optimizer", "device", "pod_name"]]
        .pivot(index="optimizer", columns="device", values="pod_name")
        .reset_index()
    )
    device_ratio = device_ratio.fillna(0)
    device_ratio["cpu_ratio"] = device_ratio["cpu"] / (
        device_ratio["cpu"] + device_ratio["cuda"]
    )
    device_ratio["gpu_ratio"] = device_ratio["cuda"] / (
        device_ratio["cpu"] + device_ratio["cuda"]
    )
    return device_ratio


def get_cold_start_number(result, SLAs):
    result = result[result["sla"].isin(SLAs)]
    cold_start_pod_number = (
        result[["optimizer", "sla", "function_name", "pod_name"]]
        .drop_duplicates()
        .groupby(["optimizer", "sla", "function_name"])
        .count()
        .reset_index()
    )
    total_round = max(result["round"])
    cold_start_pod_number["pod_name"] = (
        cold_start_pod_number["pod_name"] - 1
    ) / total_round
    cold_start_pod_number = (
        cold_start_pod_number[["optimizer", "pod_name"]]
        .groupby(["optimizer"])
        .mean()
        .reset_index()
    )
    return cold_start_pod_number


def get_multi_sla_result(result):
    cost = get_cost(result, [3, 4, 5, 6])
    violation_ratio = get_violation_ratio(result, [3, 4, 5, 6])
    return cost, violation_ratio


def parse_result():
    RESULT_DIR = os.path.join(BASE_DIR, "data/result/")
    MAIN_RESULT = "e2e_result.csv"
    result = pd.read_csv(os.path.join(RESULT_DIR, MAIN_RESULT))
    result = result[(result["round"] != -1)]
    result["optimizer"] = result["optimizer"].apply(lambda x: x.capitalize())
    result = result.replace("smiless", "SMIless")
    result = result.replace("grandslam", "GrandSLAm")
    result = result.replace("icebreaker", "IceBreaker")
    result = result.replace("aquatope", "Aquatope")
    result = result.replace("orion", "Orion")
    result = result.replace("opt", "OPT")
    return result


def plot_end_to_end_result(cost, workflow_running_time):
    sns.set_palette("deep")
    sns.set_style("white")
    fig = plt.figure(figsize=(6, 1.8))
    grid = plt.GridSpec(1, 5, wspace=0.5, hspace=0.5, figure=fig)
    axes = [
        plt.subplot(grid[0, 0:4]),
        plt.subplot(grid[0, 4]),
    ]
    colors = sns.color_palette("deep", n_colors=len(cost["optimizer"].unique()))
    ax1 = sns.barplot(
        ax=axes[0],
        x="optimizer",
        y="ratio",
        hue="workflow_name",
        data=cost,
        width=0.8,
        order=["Aquatope", "GrandSLAm", "IceBreaker", "Orion", "SMIless", "OPT"],
    )

    ax1.get_legend().remove()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_linewidth(1)
    ax1.spines["bottom"].set_linewidth(1)
    cmap = dict(
        zip(["Aquatope", "GrandSLAm", "IceBreaker", "Orion", "SMIless", "OPT"], colors)
    )
    patchs = [Patch(color=v, label=k) for k, v in cmap.items()]

    axes[0].set_xlabel("(a)")
    axes[0].set_ylabel("Norm. total cost", fontsize=10, labelpad=0)
    axes[0].set_xticks([])
    axes[0].tick_params(axis="y", which="major", pad=-8, size=8)

    hatch_style = [
        "///",
        "\\\\\\",
        "ooo",
        "+++",
    ]
    labels = ["AMBER Alert", "Image Query", "Voice Assistant", "Total"]
    hatchs = [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=h,
            label=labels[hatch_style.index(h)],
        )
        for h in hatch_style
    ]

    patch_indices = [i for i in range(len(cost["optimizer"].unique()))] * 4
    for p in ax1.patches:
        if p.get_height() == 0:
            continue
        ax1.annotate(
            r"{:.1f}".format(p.get_height(), 2),
            (p.get_x() + p.get_width() / 2.0, p.get_height() * 1.2),
            ha="center",
            va="center",
            xytext=(0, 10),
            rotation=85,
            fontsize=10,
            textcoords="offset points",
        )
        if ax1.patches.index(p) < 24:
            p.set_hatch(hatch_style[int(axes[0].patches.index(p) / 6)])
            p.set_edgecolor("black")
            p.set_facecolor(
                patchs[patch_indices[axes[0].patches.index(p)]]._hatch_color
            )

    axes[0].grid(linestyle="--", linewidth=0.5, axis="y")
    axes[0].set_ylim(0.15, 500)
    axes[0].set_yscale("log")
    first_legend = plt.legend(
        handles=hatchs,
        loc="upper center",
        bbox_to_anchor=(-3.2, 1.1),
        ncol=4,
        handlelength=2,
        handleheight=1,
        handletextpad=0.3,
        columnspacing=0.3,
        title=None,
        frameon=False,
        fontsize=8,
    )
    plt.gca().add_artist(first_legend)
    plt.legend(
        handles=patchs,
        loc="upper center",
        bbox_to_anchor=(-2.5, 1.26),
        ncol=6,
        handlelength=1.1,
        handletextpad=0.1,
        columnspacing=0.6,
        title=None,
        frameon=False,
        fontsize=10,
    )
    ax2 = sns.boxplot(
        ax=axes[1],
        x="optimizer",
        y="workflow_running_time",
        data=workflow_running_time,
        showfliers=False,
        linewidth=0.8,
        fliersize=1.5,
        order=["Aquatope", "GrandSLAm", "IceBreaker", "Orion", "SMIless", "OPT"],
    )
    for p in ax2.patches:
        p.set_facecolor(patchs[patch_indices[ax2.patches.index(p)]]._hatch_color)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(1)
    ax2.spines["bottom"].set_linewidth(1)
    axes[1].set_xlabel("(b)")
    axes[1].set_ylabel("Latency (s)", fontsize=10, labelpad=0)
    axes[1].set_xticks([])
    axes[1].tick_params(axis="y", which="major", pad=-5, size=8)
    axes[1].grid(linestyle="--", linewidth=0.5, axis="y")
    plt.subplots_adjust(wspace=0.1)
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "optimizer_cost_merge_all.pdf",
        ),
        bbox_inches="tight",
    )
    plt.cla()


def plot_cpu_gpu_ratio(device_ratio):
    plt.figure(figsize=(3, 2), dpi=120)
    sns.set_palette("deep")
    sns.set_style("white")
    sns.barplot(
        data=device_ratio,
        x="optimizer",
        y="cpu_ratio",
        color="#5975a4",
        label="CPU",
        width=0.6,
    )
    sns.barplot(
        data=device_ratio,
        x="optimizer",
        y="gpu_ratio",
        color="#cc8963",
        # color = "#4ca66c",
        bottom=device_ratio["cpu_ratio"],
        width=0.6,
        label="GPU",
    )
    fig = plt.gca()
    fig.spines["right"].set_visible(False)
    fig.spines["top"].set_visible(False)
    fig.spines["left"].set_linewidth(1.3)
    fig.spines["bottom"].set_linewidth(1.3)
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    plt.xticks(rotation=40)
    plt.legend(
        loc=(0.4, 0.8),
        ncol=2,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=0.5,
        frameon=False,
    )

    plt.ylim(0, 1.2)
    plt.ylabel("Device Ratio")
    plt.xlabel("")
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "device_ratio.pdf",
        ),
        bbox_inches="tight",
    )
    plt.cla()


def plot_reinit_ratio(pod_reinit_ratio):
    sns.set_palette("deep")
    sns.set_style("white")
    plt.figure(figsize=(3, 2), dpi=120)
    sns.barplot(data=pod_reinit_ratio, x="optimizer", y="pod_name", width=0.7)
    fig = plt.gca()
    for p in fig.patches:
        fig.annotate(
            "{:.2f}".format(p.get_height()),
            (p.get_x() + p.get_width() / 2.0, p.get_height() - p.get_height() * 0.04),
            ha="center",
            va="center",
            xytext=(0, 10),
            fontsize=10,
            textcoords="offset points",
        )
    fig.spines["right"].set_visible(False)
    fig.spines["top"].set_visible(False)
    fig.spines["left"].set_linewidth(1.4)
    fig.spines["bottom"].set_linewidth(1.4)
    plt.grid(linestyle="--", linewidth=0.5, axis="y")
    plt.xticks(rotation=40)
    plt.ylabel("Re-init Ratio")
    plt.ylim(0, 0.48)
    plt.xlabel("")
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "reinit_ratio.pdf",
        ),
        bbox_inches="tight",
    )
    plt.cla()


def plot_multi_sla(multi_sla_cost, multi_sla_violation_ratio):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 1.7), dpi=120)
    sns.set(font_scale=1.03)
    sns.set_palette("deep")
    sns.set_style("white")

    colors = sns.color_palette("deep", n_colors=5)
    ax1 = sns.lineplot(
        ax=axes[0],
        x="sla",
        y="ratio",
        hue="optimizer",
        hue_order=["Aquatope", "GrandSLAm", "IceBreaker", "Orion", "SMIless"],
        style="optimizer",
        data=multi_sla_cost,
        legend=False,
        markers=True,
        markersize=7,
    )
    # axes[0].set_xlabel("\n(a) The overall cost of all applications",fontsize=10)
    axes[0].set_xlabel("SLA (s)\n(a)")
    axes[0].set_ylabel("Norm. total cost", fontsize=12)
    axes[0].set_xticks([3, 4, 5, 6])
    axes[0].grid(linestyle="--", linewidth=0.5)
    # axes[0].set_ylim(10,200)
    # plt.grid(True,alpha=0.3)
    # plt.xticks(rotation=40)
    for p in axes[0].patches:
        print(p.get_height())
        axes[0].annotate(
            r"{:.2f}".format(p.get_height(), 2),
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 5),
            # rotation=30,
            fontsize=11,
            textcoords="offset points",
        )
    axes[0].grid(linestyle="--", linewidth=0.5, axis="y")
    plt.legend().remove()
    axes[0].set_ylim(0, 16)

    # plt.figure(figsize=(4, 2),dpi=200)
    ax2 = sns.lineplot(
        ax=axes[1],
        x="sla",
        y="violation",
        hue="optimizer",
        hue_order=["Aquatope", "GrandSLAm", "IceBreaker", "Orion", "SMIless"],
        style="optimizer",
        data=multi_sla_violation_ratio,
        legend=False,
        markers=True,
    )
    # axes[1].set_xlabel("\n(b) The distribution of the end-to-end \n latency of all applications",fontsize=10)
    marker_styles = [line.get_marker() for line in ax2.get_lines()]
    line_styles = [line.get_linestyle() for line in ax2.get_lines()]
    cmap = tuple(
        zip(
            ["Aquatope", "GrandSLAm", "IceBreaker", "Orion", "SMIless"],
            colors,
            marker_styles,
            line_styles,
        )
    )
    # markers =
    patchs = [
        Line2D([0], [0], color=v, label=k, marker=m, linestyle=l) for k, v, m, l in cmap
    ]

    plt.legend(
        # axes[0],
        handles=patchs,
        loc="upper center",
        bbox_to_anchor=(-0.3, 1.3),
        ncol=6,
        handlelength=2,
        handletextpad=0.1,
        columnspacing=0.5,
        title=None,
        frameon=False,
        # fontsize=10
    )
    axes[1].set_xlabel("SLA (s)\n(b)")
    axes[1].set_ylabel("Vio. Ratio")
    axes[1].set_xticks([3, 4, 5, 6])
    axes[1].grid(linestyle="--", linewidth=0.5)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax == axes[1]:
            ax.spines["left"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
    plt.subplots_adjust(wspace=0.44)
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "mutli_sla_optimizer_cost_merge.pdf",
        ),
        bbox_inches="tight",
    )
    plt.cla()


result = parse_result()
cost = get_cost(result, [2])
workflow_running_time = get_workflow_running_time(result, [2])
device_ratio = get_device_ratio(result, [2])
cold_start_pod_number = get_cold_start_number(result, [2])
multi_sla_cost, multi_sla_violation = get_multi_sla_result(result)

plot_end_to_end_result(cost, workflow_running_time)
plot_cpu_gpu_ratio(device_ratio)
plot_reinit_ratio(cold_start_pod_number)
plot_multi_sla(multi_sla_cost, multi_sla_violation)
