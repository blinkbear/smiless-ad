import numpy as np
import pandas as pd
import copy
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


class PathSearchOPT:
    def init(self, df, shared_nodes, SLA, IT):
        self.df = df
        self.SLA = SLA
        self.IT = IT
        self.network_time = 0.0
        self.shared_nodes = shared_nodes
        self.df["cpu_execution_time"] = (
            self.df["cpu_running_time"]
            + self.df["cpu_cs_extra_time"]
            + self.df["cold_start_time"]
            + self.network_time
        )
        self.df["gpu_execution_time"] = (
            self.df["gpu_running_time"]
            + self.df["gpu_cs_extra_time"]
            + self.df["cold_start_time"]
            + self.df["gpu_trans_time"]
            + self.network_time
        )
        self.df["cpu_inference_time"] = (
            self.df["cpu_running_time"]
            + self.df["cpu_cs_extra_time"]
            + self.network_time
        )
        self.df["gpu_inference_time"] = (
            self.df["gpu_running_time"]
            + self.df["gpu_cs_extra_time"]
            + self.network_time
        )
        self.df["cpu_execution_cost"] = (
            self.df["cpu_execution_time"] * self.df["cpu_cost"]
        )
        self.df["gpu_execution_cost"] = (
            self.df["gpu_execution_time"] * self.df["gpu_cost"]
        )
        self.df["cpu_it_cost"] = self.df["cpu_cost"] * self.IT
        self.df["gpu_it_cost"] = self.df["gpu_cost"] * self.IT
        self.df = self.calc_function_running_time(self.df, 0, 0)
        self.available_cost = []

    def get_running_plan_df(self):
        self.best_cost = float("inf")
        self.df["cpu_cs_extra_time"] = self.df["cpu_cs_extra_time"]
        self.df["gpu_cs_extra_time"] = self.df["gpu_cs_extra_time"]
        self.available_solution = None
        self.Path_search(self.df, self.SLA)
        if self.available_solution is None:
            print("No feasible solution")
            self.available_solution = self.df
            self.available_solution["device"] = "cuda"
        self.calc_prewarm_window()
        return self.available_solution

    def calc_function_running_time(self, df, index, prev_time):
        """
        Calculates the running time for each stage in a given DataFrame.

        Parameters:
        - df: A pandas DataFrame containing the data for the calculation.
        - index: The index of the current stage.
        - prev_time: The running time of the previous stage.

        Returns:
        None
        """

        last_running_time = np.zeros(len(df))
        current_running_time = np.zeros(len(df))
        # Loop through each row in the DataFrame
        for i, row in df.iterrows():
            if row["node"] in self.shared_nodes:
                df.at[i, "device"] = self.shared_nodes[row["node"]][1]
            else:
                df.at[i, "device"] = "cpu"
            if i < index:
                # For functions before the current one, just append the values from the DataFrame
                last_running_time[i] = row["last_running_time"]
                current_running_time[i] = row["current_running_time"]
                prev_time = row["current_running_time"]
                continue
            last_running_time[i] = prev_time
            if row["device"] == "cpu":
                prev_time = row["cpu_inference_time"] + prev_time
                current_running_time[i] = prev_time
            else:
                prev_time = row["gpu_inference_time"] + prev_time
                current_running_time[i] = prev_time

        df["last_running_time"] = last_running_time
        df["current_running_time"] = current_running_time
        return df

    def calc_current_cost(self, df, index):
        return df[:index]["cost"].sum()

    def calc_task_cost(self, row):
        """
        Calculates the cost of running a task based on the values in the input `row`.

        Args:
            row (pandas.Series): A pandas series containing the following columns:
                - device: The mode in which the task is running (either "cpu" or "cuda").
                - start_mode: The mode in which the task was started (either "cold" or "warm").
                - cpu_running_time: The amount of CPU time used by the task (in seconds).
                - cpu_cs_extra_time: The amount of extra CPU time used during context switching (in seconds).
                - gpu_running_time: The amount of GPU time used by the task (in seconds).
                - gpu_cs_extra_time: The amount of extra GPU time used during context switching (in seconds).
                - cpu_cost: The cost per second of using a CPU.
                - gpu_cost: The cost per second of using a GPU.
                - keep_alive_cost: The cost per second of keeping the task alive (i.e., not terminating it).
                - last_running_time: The amount of time the task spent running during its last execution (in seconds).

        Returns:
            float: The cost of running the task (in dollars).
        """
        cost = 0
        if row["device"] == "cpu":
            if (
                row["cpu_running_time"] < self.IT
                and self.IT < row["cpu_execution_time"]
            ):
                cost += row["cpu_it_cost"]
            else:
                cost += row["cpu_execution_cost"]
        else:
            if (
                row["gpu_running_time"] < self.IT
                and self.IT < row["gpu_execution_time"]
            ):
                cost += row["gpu_it_cost"]
            else:
                cost += row["gpu_execution_cost"]
        return cost

    def calc_prewarm_window(self):
        cpu_condition = self.available_solution.device.values == "cpu"

        cpu_prewarm = np.maximum(
            0,
            self.IT - self.available_solution.cpu_execution_time.values,
            dtype=np.float64,
        )
        gpu_prewarm = np.maximum(
            0,
            self.IT - self.available_solution.gpu_execution_time.values,
            dtype=np.float64,
        )

        self.available_solution["prewarm_window"] = np.where(
            cpu_condition, cpu_prewarm, gpu_prewarm
        )
        return

    def order_running_start_mode_by_cost(self, row, shared_nodes, SLA, index):
        device = []
        if row["node"] in shared_nodes:
            # If the task is running on a shared node, we only use cuda and warm start mode
            device = [
                self.shared_nodes[row["node"]][1],
                self.shared_nodes[row["node"]][1],
            ]
            return device
        if row["cpu_execution_cost"] < row["gpu_execution_cost"]:
            device = ["cpu", "cuda"]
        else:
            device = ["cuda", "cpu"]
        return device

    def OPT_search(self, df, SLA):
        # initialize varindex:index:
        if df.loc[df.index[-1], "current_running_time"] < SLA:
            self.available_solution = df
            return
        df_copy = copy.deepcopy(df)
        df_copy.loc[0, "last_running_time"] = 0
        df_copy["cost"] = 0
        current_index = 0
        current_time = 0
        open_list = PriorityQueue()
        open_list.put(PrioritizedItem(0, (df_copy, 0)))
        # while there are still nodes in the open list
        while not open_list.empty():
            # get the node with the lowest cost

            item = open_list.get()
            curr_df, current_index = item.item
            # calculate the cost of each task in the current DataFrame
            curr_df["cost"] = curr_df.apply(
                lambda row: self.calc_task_cost(row), axis=1
            )
            # if all tasks have been scheduled, check if the current solution is better than the current best solution
            if current_index == len(curr_df):
                completion_time = curr_df.loc[curr_df.index[-1], "current_running_time"]
                if completion_time > SLA:
                    continue
                self.available_solution = curr_df
                return
            # generate child nodes by assigning different running modes and start modes to the current task
            device = self.order_running_start_mode_by_cost(
                curr_df.loc[current_index], self.shared_nodes, SLA, current_index
            )
            for i in range(len(device)):
                child_df = curr_df.copy()
                child_df.loc[current_index, "device"] = device[i]

                # calculate the running time of the function with the new assignment
                child_df = self.calc_function_running_time(
                    child_df, current_index, current_time
                )
                # if the resulting DataFrame has already been explored, skip it
                child_df.loc[current_index, "cost"] = self.calc_task_cost(
                    curr_df.loc[current_index]
                )
                current_cost = self.calc_current_cost(child_df, current_index)
                # add the new node to the open list
                open_list.put(
                    PrioritizedItem(current_cost, (child_df, current_index + 1))
                )
        return
