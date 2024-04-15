import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
import os
import copy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "data/")
FIGURE_DIR = os.path.join(BASE_DIR, "figure/")


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


def parse_result():
    RESULT_DIR = os.path.join(BASE_DIR, "data/")
    MAIN_RESULT = "result_co_optimizer.csv"
    result = pd.read_csv(os.path.join(RESULT_DIR, MAIN_RESULT))
    result = result[(result["round"] != -1)]
    result["optimizer"] = result["optimizer"].apply(lambda x: x.capitalize())
    result = result.replace("smiless", "SMIless")
    result = result.replace("smiless-homo", "SMIless-Homo")
    result = result.replace("smiless-no-dag", "SMIless-No-DAG")
    return result


def plot_result(component_result, component_violation_result):
    fig, axes = plt.subplots(1, 2, figsize=(6, 1.8), dpi=120)
    sns.set_palette("deep")
    sns.set_style("white")
    colors = sns.color_palette("deep", n_colors=3)
    cmap = dict(zip(component_result["optimizer"], colors))
    patchs = [Patch(color=v, label=k) for k, v in cmap.items()]
    ax1 = sns.barplot(
        ax=axes[0], x="optimizer", y="value", data=component_result, width=0.55
    )
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.3)
    ax1.spines["bottom"].set_linewidth(1.3)
    axes[0].set_xlabel("(a)")
    axes[0].set_ylabel("Norm. total cost")
    axes[0].set_xticks([])
    # plt.grid(True,alpha=0.3)
    # plt.xticks(rotation=40)
    for p in axes[0].patches:
        print(p.get_height())
        axes[0].annotate(
            r"{:.2f}".format(p.get_height()),
            (p.get_x() + p.get_width() / 2.0, p.get_height() - 0.05),
            ha="center",
            va="center",
            xytext=(0, 10),
            fontsize=10,
            textcoords="offset points",
        )
    axes[0].grid(linestyle="--", linewidth=0.5, axis="y")
    plt.legend().remove()
    axes[0].set_ylim(0, 1.6)
    plt.legend(
        handles=patchs,
        loc="upper center",
        bbox_to_anchor=(-0.2, 1.3),
        ncol=6,
        handlelength=1,
        handletextpad=0.5,
        columnspacing=0.5,
        title=None,
        frameon=False,
    )

    ax2 = sns.barplot(
        ax=axes[1],
        x="optimizer",
        y="violation",
        data=component_violation_result,
        width=0.55,
    )
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_linewidth(1.3)
    ax2.spines["bottom"].set_linewidth(1.3)
    for p in axes[1].patches:
        print(p.get_height())
        axes[1].annotate(
            r"{:.2f}".format(p.get_height()),
            (p.get_x() + p.get_width() / 2.0, p.get_height() + 0.005),
            ha="center",
            va="center",
            xytext=(0, 4),
            fontsize=10,
            textcoords="offset points",
        )
    # axes[1].set_ylim(0,1)
    axes[1].set_xlabel("(b)")
    axes[1].set_ylabel("Vio. Ratio")
    axes[1].set_xticks([])
    axes[1].grid(linestyle="--", linewidth=0.5, axis="y")
    # axes[1].set_tight_layout(box_pad=0.1)
    plt.subplots_adjust(wspace=0.45)
    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            "co-optimizer.pdf",
        ),
        bbox_inches="tight",
    )


result = parse_result()
component_result = get_cost(result, [2])
component_violation_result = get_violation_ratio(result, [2])


plot_result(component_result, component_violation_result)
