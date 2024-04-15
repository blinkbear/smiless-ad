from offline_profiler.function_resource_profiler import FunctionProfiler
from .optimizer import Optimizer


class GrandSLAm(Optimizer):
    def __init__(self, default_device, function_profiler: FunctionProfiler):
        self.default_device = default_device
        self.function_profiler = function_profiler
        self.available_cpu_resource = 1
        self.available_gpu_resource = 100
        self.workflow_strategy = {}

    def get_workflow_running_plan_df(
        self,
        workflow_name,
        graph_df,
        graph_dfs,
        IT,
        SLA,
        interval_time_unit,
    ):
        graph_df = self.get_cpu_time(graph_df)
        graph_df["device"] = self.get_device(graph_df, SLA)
        graph_df = self.get_slack(graph_df, SLA)
        graph_df = self.get_resources(graph_df)
        nodes = graph_df["node"].tolist()
        graph_df["keep_alive_time"] = 0
        graph_df["cold_start_stage"] = "False"
        graph_df["function_prewarm_time"] = -1000
        graph_df["knee_point"] = 1
        for i, row in graph_df.iterrows():
            if row["device"] == "cuda":
                graph_df.loc[i, "resource_quantity"] = self.available_cpu_resource
                graph_df["knee_point"] = 32
            else:
                graph_df.loc[i, "resource_quantity"] = self.available_gpu_resource
                graph_df["knee_point"] = 1
        self.workflow_strategy[workflow_name] = graph_df
        return graph_df, nodes

    def get_slack(self, graph_df, SLA):
        if self.default_device == "cuda":
            total_latency = graph_df["gpu_running_time"].sum()
            graph_df["slack"] = graph_df["gpu_running_time"] / total_latency * SLA
        elif self.default_device == "cpu":
            total_latency = graph_df["cpu_running_time"].sum()
            graph_df["slack"] = graph_df["cpu_running_time"] / total_latency * SLA
        return graph_df

    def get_device(self, graph_df, SLA):
        total_cpu_running_time = graph_df["cpu_running_time"].sum()
        if total_cpu_running_time > SLA:
            return "cuda"
        else:
            return "cpu"

    def get_cpu_time(self, df):
        df[
            [
                "cpu_running_time",
                "cpu_running_cost",
                "gpu_running_cost",
                "cpu_keep_alive_cost",
                "gpu_keep_alive_cost",
            ]
        ] = df.apply(
            lambda x: self.function_profiler.get_cpu_cost_running_time(
                x["types"], x["qps"], self.available_cpu_resource, x["image_size"]
            ),
            axis=1,
            result_type="expand",
        )
        return df

    def get_resources(self, graph_df):
        graph_df["keep_alive_resource"] = graph_df.apply(
            lambda x: self.function_profiler.get_resource_by_latency(
                x["node"], self.default_device, x["slack"]
            ),
            axis=1,
        )
        return graph_df

    def update_workflow_running_plan_df(
        self, workflow_name, IT, SLA, interval_time_unit
    ):
        graph_df = self.workflow_strategy[workflow_name]
        graph_df, nodes = self.get_workflow_running_plan_df(
            workflow_name=workflow_name,
            graph_df=graph_df,
            graph_dfs=None,
            IT=IT,
            SLA=SLA,
            interval_time_unit=interval_time_unit,
        )
        return graph_df, nodes

    def remove_workflow(self, workflow_name):
        del self.workflow_strategy[workflow_name]
