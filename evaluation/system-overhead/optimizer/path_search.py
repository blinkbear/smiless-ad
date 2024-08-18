import numpy as np
import pandas as pd
import copy
from queue import PriorityQueue, Queue
from collections import deque
from dataclasses import dataclass, field
from typing import Any
import time
import json


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
        self.df['device']='cpu'
        self.df["cpu_execution_time"] = (
            df["cpu_running_time"]
            + df["cpu_cs_extra_time"]
            + df["cold_start_time"]
            + self.network_time
        )
        self.df["gpu_execution_time"] = (
            df["gpu_running_time"]
            + df["gpu_cs_extra_time"]
            + df["cold_start_time"]
            + df["gpu_trans_time"]
            + self.network_time
        )
        self.df["cpu_inference_time"] = (
            df["cpu_running_time"]
            + df["cpu_cs_extra_time"]
            + self.network_time
        )
        self.df["gpu_inference_time"] = (
            df["gpu_running_time"]
            + df["gpu_cs_extra_time"]
            + self.network_time
        )
        self.df["cpu_execution_cost"] = (
            df["cpu_execution_time"] * df["cpu_cost"]
        )
        self.df["gpu_execution_cost"] = (
            df["gpu_execution_time"] * df["gpu_cost"]
        )
        self.df["cpu_it_cost"] = df["cpu_cost"] * IT
        self.df["gpu_it_cost"] = df["gpu_cost"] * IT
        self.df = self.calc_device_cost(df)
        self.df = self.calc_function_running_time(df, 0, 0)
        self.max_cost=self.df["max_cost"].max()
        self.available_cost = []

    def get_running_plan_df(self, search_method: str = "path_search"):
        self.best_cost = float("inf")
        self.df["cpu_cs_extra_time"] = self.df["cpu_cs_extra_time"]
        self.df["gpu_cs_extra_time"] = self.df["gpu_cs_extra_time"]
        self.available_solution = None

        if search_method == "path_search":
            self.Path_search(self.df, self.SLA)
        elif search_method == "bfs":
            self.bfs(self.df, self.SLA)
        elif search_method == "aug_smiless":
            self.bfs_with_top_k(self.df, self.SLA, k=2)
        elif search_method == "dfs":
            self.dfs(self.df, self.SLA)
        elif search_method == "Astar":
            self.A_star_search(self.df, self.SLA)
        if self.available_solution is None:
            print("No feasible solution")
            self.available_solution = self.df
            self.available_solution["device"] = "cuda"
            self.available_solution["cost"] = self.available_solution['cuda_cost']
        self.calc_prewarm_window()
        return self.available_solution
    
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
    def calc_total_cost(self, df):
        """
        Calculate the total cost of a DataFrame.

        :param df: A pandas DataFrame with a 'cost' column
        :return: The sum of the 'cost' column in the DataFrame
        """
        # return df["cost"].sum()
        return df.apply(lambda row: self.calc_task_cost(row), axis=1).sum()

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
        if index > 0:
            last_running_time[:index]=df[:index]['last_running_time'].tolist()
            current_running_time[:index]=df[:index]["current_running_time"].tolist()
        cpu_inference_time = df.loc[index:, 'cpu_inference_time'].values
        gpu_inference_time = df.loc[index:, 'gpu_inference_time'].values
        device = df.loc[index:, 'device'].values
        inference_time = np.where(device == 'cpu', cpu_inference_time, gpu_inference_time)
        # 计算累积和
        cumulative_time = np.cumsum(inference_time + prev_time)
        # 更新 last_running_time 和 current_running_time
        last_running_time[index+1:] = cumulative_time[:-1]
        current_running_time[index:] = cumulative_time
        # # Loop through each row in the DataFrame

        df["last_running_time"] = last_running_time
        df["current_running_time"] = current_running_time
        
        return df

    def calc_current_cost(self, df, index):
        return df[:index]["cost"].sum()

    def calc_device_cost(self,df):
        IT = self.IT
        cpu_cost = []
        cuda_cost = []
        max_cost = []
        total_max_cost=[]
        devices=[]
        max_cost=0
        for i, row in df.iterrows():
            if row["cpu_running_time"] < IT and IT < row["cpu_execution_time"]:
                cpu_cost.append(row['cpu_it_cost'])
            if row["gpu_running_time"] < IT and IT < row["gpu_execution_time"]:
                cuda_cost.append(row['gpu_it_cost'])
            if row["cpu_running_time"] >= IT or IT >= row["cpu_execution_time"]:
                cpu_cost.append(row['cpu_execution_cost'])
            if row["gpu_running_time"] >= IT or IT >= row["gpu_execution_time"]:
                cuda_cost.append(row['gpu_execution_cost'])
            if row["cpu_execution_cost"] <row["gpu_execution_cost"]:
                devices.append(('cpu','cuda'))
            if row["cpu_execution_cost"] >=row["gpu_execution_cost"]:
                devices.append(('cuda','cpu'))

            max_cost+=max([row['cpu_it_cost'], row['gpu_it_cost'],row['cpu_execution_cost'],row['gpu_execution_cost']])
            total_max_cost.append(max_cost)
        df['cpu_cost']=cpu_cost
        df['cuda_cost']=cuda_cost
        df['max_cost']=total_max_cost
        df['selected_devices']=devices
        return df

    def order_running_start_mode_by_cost(self, row, shared_nodes, SLA, index):
        device = []
        if row["cpu_execution_cost"] < row["gpu_execution_cost"]:
            device = ["cpu", "cuda"]
        else:
            device = ["cuda", "cpu"]
        return device

    def reverse_order_running_start_mode_by_cost(self, row, shared_nodes, SLA, index):
        device = []
        if row["cpu_execution_cost"] < row["gpu_execution_cost"]:
            device = ["cuda", "cpu"]
        else:
            device = ["cpu", "cuda"]
        return device              


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


    def calc_heuristic_cost(self, df, index, SLA):
        """
        Calculates the heuristic cost of executing a task in a given dataframe at a specific index.

        Args:
        - df: a pandas dataframe containing information about the tasks to be executed
        - index: the index of the task to be executed
        - SLA: the agreed upon Service Level Agreement for the task

        Returns:
        - cost: the heuristic cost of executing the task at the given index in the given dataframe
        """
        current_time = df.at[index - 1, "current_running_time"] if index > 0 else 0
        if current_time > SLA:
            return float("inf")
        estimation_minimum_execution_time=df['gpu_inference_time'].iloc[index:].sum()

        if estimation_minimum_execution_time + current_time > SLA:
            return float("inf")
        max_cost = self.max_cost-df.at[index-1,"max_cost"]
        return max_cost

    def Path_search(self, df, SLA):
        # initialize varindex:index:
        if df.at[df.index[-1], "current_running_time"] < SLA:
            self.available_solution = df
            return
        df_copy = copy.deepcopy(df)
        df_copy.at[0, "last_running_time"] = 0
        df_copy["cost"] = 0
        current_index = 0
        current_time = 0
        total_cost = 0
        df_length = len(df)
        open_list = PriorityQueue()
        open_list.put(PrioritizedItem(0, (df_copy, 0)))
        # while there are still nodes in the open list
        while not open_list.empty():
            # get the node with the lowest cost

            item = open_list.get()
            curr_df, current_index = item.item
            # calculate the cost of each task in the current DataFrame
            # if all tasks have been scheduled, check if the current solution is better than the current best solution
            if current_index == df_length:
                completion_time = curr_df.at[df_length-1, "current_running_time"]
                if completion_time > SLA:
                    continue
                self.available_solution = curr_df
                return
            devices = curr_df.at[current_index,"selected_devices"]
            for device in devices:
                child_df = curr_df
                child_df.at[current_index, "device"] = device

                child_df = self.calc_function_running_time(
                    child_df, current_index, current_time
                )
                # if the resulting DataFrame has already been explored, skip it
                child_df.at[current_index, "cost"]=child_df.at[current_index, f"{device}_cost"]
                # calculate the heuristic cost of the new node
                heuristic_cost = self.calc_heuristic_cost(
                    child_df, current_index + 1, SLA
                )
                if heuristic_cost == float("inf"):
                    continue
                total_cost = self.calc_current_cost(child_df, current_index)
                # add the new node to the open list
                open_list.put(
                    PrioritizedItem(
                        total_cost+ heuristic_cost, (child_df, current_index + 1)
                    )
                )
        return

    def A_star_search(self, df, SLA):
        # initialize varindex:index:
        if df.loc[df.index[-1], "current_running_time"] < SLA:
            self.available_solution = df
            return
        df_copy = copy.deepcopy(df)
        df_copy.loc[0, "last_running_time"] = 0
        df_copy["cost"] = 0
        current_index = 0
        current_time = 0
        # open_list = [(0, df_copy,current_index)]
        open_list = PriorityQueue()
        open_list.put(PrioritizedItem(0, (df_copy, 0)))
        # while there are still nodes in the open list
        while not open_list.empty():
            # get the node with the lowest cost

            item = open_list.get()
            curr_df, current_index = item.item
            # calculate the cost of each task in the current DataFrame
            if current_index == len(curr_df):
                completion_time = curr_df.loc[curr_df.index[-1], "current_running_time"]
                if completion_time > SLA:
                    continue
                self.available_solution = curr_df
                continue
            device = curr_df.at[current_index,"selected_devices"] 
            for i in range(len(device)):
                child_df = curr_df.copy()
                child_df.loc[current_index, "device"] = device[i]

                child_df = self.calc_function_running_time(
                    child_df, current_index, current_time
                )
                child_df.at[current_index, "cost"]=child_df.at[current_index, f"{device[i]}_cost"]
                heuristic_cost = self.calc_heuristic_cost(
                    child_df, current_index + 1, SLA
                )
                total_cost = self.calc_current_cost(child_df, current_index)

                open_list.put(
                    PrioritizedItem(
                        total_cost + heuristic_cost, (child_df, current_index + 1)
                    )
                )
        self.available_solution = df
        return
    def bfs(self, df, SLA):
        if df.loc[df.index[-1], "current_running_time"] < SLA:
            self.available_solution = df
            return

        df_copy = df.copy()
        df_copy.loc[0, "last_running_time"] = 0
        df_copy["cost"] = 0
        current_index = 0
        current_time = 0

        open_list = Queue()
        open_list.put((df_copy, 0))

        while not open_list.empty():
            curr_df, current_index = open_list.get()

            if current_index == len(curr_df):
                completion_time = curr_df.loc[curr_df.index[-1], "current_running_time"]
                if completion_time <= SLA:
                    self.available_solution = curr_df
                continue
            device = curr_df.at[current_index,"selected_devices"]
            for i in range(len(device)):
                child_df = curr_df.copy()
                child_df.loc[current_index, "device"] = device[i]
                child_df = self.calc_function_running_time(
                    child_df, current_index, current_time
                )
                child_df.at[current_index, "cost"]=child_df.at[current_index, f"{device[i]}_cost"]
                open_list.put((child_df, current_index + 1))
        self.available_solution = df
        return
    def bfs_with_top_k(self, df, SLA,k=2):
        if df.loc[df.index[-1], "current_running_time"] < SLA:
            self.available_solution = df
            return

        df_copy = df.copy()
        df_copy.loc[0, "last_running_time"] = 0
        df_copy["cost"] = 0
        current_index = 0
        current_time = 0

        open_list = deque()
        open_list.append((df_copy, 0))

        while not len(open_list) == 0:
            candidate_df = []

            selected_index = -1
            # print(open_list)
            while True:
                if len(open_list) == 0:
                    break
                curr_df, c_index = open_list.popleft()
                if selected_index == -1:
                    selected_index = c_index
                if c_index == selected_index:
                    candidate_df.append(curr_df)
                elif c_index > selected_index:
                    open_list.appendleft((curr_df, c_index))
                    break
            current_index = selected_index
            layer_save = {}
            for curr_df in candidate_df:
                child_df = curr_df.copy()
                if current_index == len(child_df):
                    min_cost_df = child_df
                    best_cost = self.calc_total_cost(curr_df) 
                    for child_df in candidate_df:
                        completion_time = child_df.loc[
                            child_df.index[-1], "current_running_time"
                        ]
                        if completion_time > SLA:
                            continue
                        total_cost = self.calc_total_cost(curr_df) 
                        if total_cost < best_cost:
                            min_cost_df = child_df

                    self.available_solution = min_cost_df
                    continue
                device = child_df.at[current_index,"selected_devices"]

                for i in range(len(device)):
                    child_df.loc[current_index, "device"] = device[i]

                    child_df = self.calc_function_running_time(
                        child_df, current_index, current_time
                    )
                    # if the resulting DataFrame has already been explored, skip it
                    child_df.loc[current_index, "cost"] =child_df.at[current_index, f"{device[i]}_cost"] 
                    total_cost = self.calc_total_cost(child_df)
                    heuristic_cost = self.calc_heuristic_cost(
                        child_df, current_index + 1, SLA
                    )
                    if heuristic_cost == float("inf"):
                        continue
                    tag = json.dumps(child_df.to_dict())
                    layer_save[(current_index, total_cost, tag)] = copy.deepcopy(
                        child_df
                    )
            sorted_layer_save = sorted(layer_save.items(), key=lambda x: x[0][1])
            for i in range(len(sorted_layer_save)):
                if i > k+1:
                    continue 
                df = sorted_layer_save[i][1]
                open_list.append((df, current_index + 1))
        self.available_solution = df
        return

    def dfs(self, df, SLA):
        if df.loc[df.index[-1], "current_running_time"] < SLA:
            self.available_solution = df
            return

        df_copy = df.copy()
        df_copy.loc[0, "last_running_time"] = 0
        df_copy["cost"] = 0
        current_index = 0
        current_time = 0

        # Use a stack for DFS
        stack = [(df_copy, 0)]
        best_cost = 1000
        while stack:
            curr_df, current_index = stack.pop()

            if current_index == len(curr_df):
                completion_time = curr_df.loc[curr_df.index[-1], "current_running_time"]
                if completion_time <= SLA:
                    if best_cost > self.calc_total_cost(curr_df):
                        best_cost = self.calc_total_cost(curr_df)
                        self.available_solution = curr_df
                continue
            device = self.reverse_order_running_start_mode_by_cost(
                curr_df.loc[current_index], self.shared_nodes, SLA, current_index
            )

            for i in range(len(device)):
                child_df = curr_df.copy()
                child_df.loc[current_index, "device"] = device[i]

                child_df = self.calc_function_running_time(
                    child_df, current_index, current_time
                )

                child_df.loc[current_index, "cost"] = child_df.at[current_index, f"{device[i]}_cost"] 
                stack.append((child_df, current_index + 1))
        self.available_solution = df
        return
