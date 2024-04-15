import numpy as np
from queue import PriorityQueue
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class PrioritizedItem:
    priority: float
    item: Any = field(compare=False)


class PathSearch:
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
        self.optimal_device = np.full(len(self.df), "cuda", dtype="U")
        # self.SLA=SLA-self.df["cpu_inference_time"][0]

    def get_running_plan_df(self):
        self.df["device"] = np.where(
            self.df.cpu_execution_cost.values < self.df.gpu_execution_cost.values,
            "cpu",
            "cuda",
        )
        self.init_device()

        self.best_cost = float("inf")
        self.available_solution = None

        self.Path_search(self.df, self.SLA)

        if self.available_solution is None:
            # print("No feasible solution")
            self.available_solution = self.df
            self.available_solution["device"] = self.optimal_device

        self.calc_prewarm_window()
        return self.available_solution

    def init_device(self):
        self.current_running_time = np.where(
            self.df.device.values == "cpu",
            self.df.cpu_inference_time.values,
            self.df.gpu_execution_time.values,
        ).sum()

        device_sorted = np.where(
            self.df.cpu_execution_cost.values < self.df.gpu_execution_cost.values, 0, 1
        )

        additional_time = np.abs(
            self.df.cpu_running_time.values - self.df.gpu_running_time.values
        )
        self.df["device_sorted"] = device_sorted
        self.df["additional_time"] = additional_time
        return

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

    def Path_search(self, df, SLA):
        # initialize varindex:index:
        current_running_time = self.current_running_time
        if current_running_time < SLA:
            self.available_solution = df
            return
        curr_df = df
        current_index = 0
        devices = ("cpu", "cuda")
        open_list = PriorityQueue()
        open_list.put(PrioritizedItem(0, (0, current_running_time)))
        selected_devices = np.full(len(curr_df), "", dtype="U")
        # while there are still nodes in the open list
        while not open_list.empty():
            # get the node with the lowest cost
            current_index, current_running_time = open_list.get().item
            if current_index == len(curr_df) - 1:
                curr_df["device"] = selected_devices
                self.available_solution = curr_df
                return
            device = (
                devices[int(curr_df.at[current_index, "device_sorted"])],
                devices[int(curr_df.at[current_index, "device_sorted"]) ^ 1],
            )
            for i in range(len(device)):
                if curr_df.at[current_index, "device"] != device[i]:
                    # curr_df.at[current_index, "device"] = device[i]
                    selected_devices[current_index] = device[i]
                    open_list.put(
                        PrioritizedItem(
                            current_running_time > SLA,
                            (
                                current_index + 1,
                                current_running_time
                                - curr_df.at[current_index, "additional_time"],
                            ),
                        )
                    )

        return
