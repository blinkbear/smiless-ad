from offline_profiler.function_resource_profiler import FunctionProfiler
from .optimizer import Optimizer


class Orion(Optimizer):
    """
    Orion parse
    """

    def __init__(self, default_device, function_profiler: FunctionProfiler):
        self.performance_model = {}
        self.target_utilization = 0.7
        self.default_device = default_device
        self.function_profiler = function_profiler
        function_running_time_profilers = function_profiler.get_running_time_profilers()
        self.workflow_strategy = {}
        self.init_performance_model(function_running_time_profilers)

    def get_workflow_running_plan_df(
        self,
        workflow_name,
        graph_df,
        graph_dfs,
        IT,
        SLA,
        interval_time_unit,
    ):
        self.init_state(graph_df)
        all_states = self.optimize_cpu_cores(SLA)
        graph_df = self.update_resource_request(graph_df, all_states)
        graph_df = self.BFS(graph_df)
        # set all stage as cold start
        graph_df["cold_start_stage"] = "True"
        graph_df.loc[0, "cold_start_stage"] = "False"
        graph_df["device"] = "cpu"
        nodes = graph_df["node"].tolist()
        graph_df["keep_alive_resource"] = graph_df["resource_quantity"]
        graph_df[
            [
                "predicted_cpu_time",
                "cpu_running_cost",
                "gpu_running_cost",
                "cpu_keep_alive_cost",
                "gpu_keep_alive_cost",
            ]
        ] = graph_df.apply(
            lambda x: self.function_profiler.get_cpu_cost_running_time(
                x["types"], x["qps"], x["resource_quantity"], x["image_size"]
            ),
            axis=1,
            result_type="expand",
        )
        graph_df["keep_alive_time"] = graph_df["predicted_cpu_time"] * 20
        graph_df["knee_point"] = graph_df.apply(
            lambda x: 32 if x["device"] == "cuda" else x["resource_quantity"], axis=1
        )
        self.workflow_strategy[workflow_name] = graph_df
        return graph_df, nodes

    def __func(self, x, a, b):
        return a / x + b

    def update_resource_request(self, graph_df, all_states):
        """
        update resource request in graph_df
        """
        graph_df["resource_quantity"] = 0
        graph_df["cpu_running_time"] = 0
        for i in range(len(all_states)):
            state = all_states[i]
            for j, row in graph_df.iterrows():
                if row["node"] == state[3]:
                    if state[1] <= 4:
                        graph_df.loc[j, "resource_quantity"] = state[1]
                        graph_df.loc[j, "cpu_running_time"] = state[2]
                    else:
                        graph_df.loc[j, "device"] = "cuda"
                        graph_df.loc[j, "resource_quantity"] = state[1]
                        graph_df.loc[j, "gpu_running_time"] = state[2]
        return graph_df

    def init_performance_model(self, function_running_time_profilers):
        """
        divide the configuration into 4 parts:[1,4],[5,9],[10,16]
        """
        import numpy as np

        available_cpu_resource = [i for i in range(1, 17)]
        for function_name in function_running_time_profilers:
            opt = function_running_time_profilers[function_name]
            latencies = self.__func(available_cpu_resource, opt[0], opt[1])
            self.performance_model[function_name] = {
                i + 1: np.max(latencies[i] + 0.5) for i in range(0, 16)
            }

    def get_function_latency(self, function_name, CPU_cores):
        function_name = function_name.split("-")[0]
        """
        get the latency of a function with a given CPU cores from the performance model
        """
        # if CPU_cores < 5:
        #     return self.performance_model[function_name][(1, 4)]
        # elif CPU_cores < 10:
        #     return self.performance_model[function_name][(5, 9)]
        # else:
        #     return self.performance_model[function_name][(10, 16)]
        if CPU_cores <= 8:
            return self.performance_model[function_name][CPU_cores]
        else:
            return self.performance_model[function_name][8]

    def getCost(self, cpu_cores, latency):
        """
        get the cost of a state
        """
        return cpu_cores * latency

    def optimize_cpu_cores(self, objective_latency):
        from queue import PriorityQueue

        step_size = 1
        pq = PriorityQueue()
        for s0 in self.init_states:
            pq.put(s0)
        while not pq.empty():
            all_states = []
            total_latency = []
            while not pq.empty():
                snext = pq.get()
                all_states.append(snext)
            for i in range(len(all_states)):
                snext = all_states[i]
                snew_priority = snext[0]
                snew_cores = snext[1]
                snew_latency = snext[2]
                snew_name = snext[3]
                snew_cores = snew_cores + step_size
                snew_latency = self.get_function_latency(snew_name, snew_cores)
                total_latency.append(snew_latency)
                snew_priority = (
                    -1 * snew_latency * self.getCost(snew_cores, snew_latency)
                )
                all_states[i] = (snew_priority, snew_cores, snew_latency, snew_name)

                pq.put(all_states[i])
            if sum(total_latency) <= objective_latency:
                return all_states
            total_max_cores = False
            for i in range(len(all_states)):
                if all_states[i][1] <= 8:
                    break
                if i == len(all_states) - 1:
                    total_max_cores = True
            if total_max_cores:
                return all_states
        return None

    def init_state(self, graph_df):
        self.init_states = []
        for i, row in graph_df.iterrows():
            cores = 1
            latency = self.get_function_latency(row["node"], cores)
            priority = -1 * latency * self.getCost(cores, latency)
            s = (priority, cores, latency, row["node"])
            self.init_states.append(s)

    def update_running_time(self, graph_df):
        last_stage_running_time = []
        current_stage_running_time = []
        stage_utilization = []
        current_running_time = 0
        for i, row in graph_df.iterrows():
            if i == 0:
                last_stage_running_time.append(0)
                current_stage_running_time.append(row["cpu_running_time"])
                current_running_time = row["cpu_running_time"]
                stage_utilization.append(1)
            else:
                last_stage_time = current_running_time
                last_stage_running_time.append(last_stage_time)
                current_running_time = (
                    max(
                        last_stage_time,
                        row["cold_start_time"] + row["delay"],
                    )
                    + row["cpu_running_time"]
                    + row["cpu_cs_extra_time"]
                )
                current_stage_running_time.append(current_running_time)
                stage_utilization.append(
                    (
                        row["cold_start_time"]
                        + row["cpu_running_time"]
                        + row["cpu_cs_extra_time"]
                    )
                    / (current_running_time - row["delay"])
                )
        graph_df["last_running_time"] = last_stage_running_time
        graph_df["current_running_time"] = current_stage_running_time
        graph_df["stage_utilization"] = stage_utilization
        return graph_df

    def calculate_utilization_e2e_latency(self, graph_df, d):
        graph_df["delay"] = d
        graph_df = self.update_running_time(graph_df)
        return (
            graph_df["stage_utilization"].min(),
            graph_df["current_running_time"].max(),
        )

    def BFS(self, graph_df):
        import numpy as np

        nodes = graph_df["node"].tolist()
        d = np.zeros(len(nodes))
        delta = 0.1

        # Define the best utilization.
        utilization, min_latency = self.calculate_utilization_e2e_latency(graph_df, d)

        while utilization <= self.target_utilization:
            stage_utilization = graph_df["stage_utilization"].tolist()
            # find the stage with the lowest utilization
            new_d = np.copy(d)
            min_index = stage_utilization.index(min(stage_utilization))
            new_d[min_index] += delta
            utilization, latency = self.calculate_utilization_e2e_latency(
                graph_df, new_d
            )
            d = new_d

        graph_df["function_prewarm_time"] = graph_df["delay"]
        graph_df = graph_df.drop(["delay"], axis=1)
        return graph_df

    def update_workflow_running_plan_df(
        self, workflow_name, IT, SLA, interval_time_unit
    ):
        graph_df = self.workflow_strategy[workflow_name]
        graph_df, nodes = self.get_workflow_running_plan_df(
            workflow_name=workflow_name,
            graph_df=graph_df,
            graph_dfs=self.graph_dfs,
            IT=IT,
            SLA=SLA,
            interval_time_unit=interval_time_unit,
        )
        return graph_df, nodes

    def remove_workflow(self, workflow_name):
        del self.workflow_strategy[workflow_name]
