import pandas as pd
from offline_profiler.function_resource_profiler import FunctionProfiler
from .path_search import PathSearch
from .optimizer import Optimizer
from online_predictor.online_predictor import OnlinePredictor
from cache.cache import Cache
import logging

log = logging.getLogger("rich")


class SMIlessNoDag(Optimizer):
    def __init__(
        self,
        cache: Cache,
        function_profilers: FunctionProfiler,
        online_predictor: OnlinePredictor,
    ):
        self.function_profilers = function_profilers

        # Set the number of available CPU resources.
        self.available_cpu_resource = 1

        self.cache = cache

        self.delay = 0.0

        self.path_search = PathSearch()
        self.workflow_strategy = {}
        self.online_predictor = online_predictor

    def update_workflow_running_plan_df(
        self,
        workflow_name,
        IT,
        SLA,
        interval_time_unit,
    ):
        graph_df, graph_dfs = self.workflow_strategy[workflow_name]
        graph_df, nodes = self.get_workflow_running_plan_df(
            workflow_name,
            graph_df,
            graph_dfs,
            IT,
            SLA,
            interval_time_unit,
        )
        return graph_df, nodes

    def get_workflow_running_plan_df(
        self,
        workflow_name,
        graph_df,
        graph_dfs,
        IT,
        SLA,
        interval_time_unit,
    ):
        # Get list of nodes in the call graph
        updated = workflow_name in self.workflow_strategy
        self.interval_time_unit = interval_time_unit
        self.workflow_strategy[workflow_name] = (graph_df, graph_dfs)
        self.IT = IT
        import datetime

        st = datetime.datetime.now()
        nodes = graph_df["node"].tolist()
        if IT == 0 and updated:
            logging.info(f"workflow {workflow_name} no need to update running plan")
            return None, None
        knee_point = 1
        shared_nodes = self.cache.get_shared_nodes(nodes)
        if updated:
            shared_nodes = {}
        shared_nodes_resource_quantity = {}
        for node in shared_nodes:
            qps = shared_nodes[node][0] + 1
            device = shared_nodes[node][1]
            (
                device,
                resource_quantity,
            ) = self.function_profilers.get_shared_function_resource_with_qps(
                node, qps, device
            )
            shared_nodes[node] = (qps, device, knee_point)
            shared_nodes_resource_quantity[node] = resource_quantity
        # Initialize an empty list to store DataFrames for each function in the call graph
        dfs = []

        for df in graph_dfs:
            # Set the resource quantity to the available CPU resource
            df["resource_quantity"] = self.available_cpu_resource
            # df["knee_point"] = 4
            for node in shared_nodes_resource_quantity:
                df.loc[df["node"] == node, "resource_quantity"] = (
                    shared_nodes_resource_quantity[node]
                )
            # Calculate the running time and cost for the CPU and GPU for each function in the DataFrame
            df[
                [
                    "cpu_running_time",
                    "cpu_cost",
                    "gpu_cost",
                    "cpu_keep_alive_cost_unit",
                    "gpu_keep_alive_cost_unit",
                ]
            ] = df.apply(
                lambda x: self.function_profilers.get_cpu_cost_running_time(
                    x["types"], x["qps"], x["resource_quantity"], x["image_size"]
                ),
                axis=1,
                result_type="expand",
            )
            self.path_search.init(df, shared_nodes, SLA, self.IT)
            df = self.path_search.get_running_plan_df()
            df["prewarm_window"] = df.loc[0, "prewarm_window"]

            logging.debug(f"running plan is {df}")
            df = self.get_keep_alive_time_for_each_function(df, shared_nodes)
            dfs.append(df)
        df = pd.concat(dfs).drop_duplicates()
        result_df = pd.DataFrame(columns=df.columns)
        for node in nodes:
            tmp_df = df[df["node"] == node]
            tmp = tmp_df.max()
            result_df = pd.concat([result_df, tmp.to_frame().T])
        result_df = result_df.reset_index(drop=True)
        result_df["prewarm_window"] = abs(result_df["prewarm_window"])
        et = datetime.datetime.now()
        log.info(f"time is  {et- st}, SLA is {SLA}")
        return result_df, nodes

    def get_keep_alive_time_for_each_function(self, df):
        entry_node = df[df["depth"] == 1]["node"].values[0]
        keep_alive_time = (
            self.function_profilers.calc_keep_alive_time(entry_node)
            * self.interval_time_unit
        )
        df["keep_alive_time"] = keep_alive_time

        def __get_keep_alive_time(row):
            if row["prewarm_window"] == 0:
                return row["keep_alive_time"]

            else:
                return -1

        df["keep_alive_time"] = df.apply(lambda x: __get_keep_alive_time(x), axis=1)
        df["keep_alive_resource"] = df["resource_quantity"]
        return df

    def remove_workflow(self, workflow_name):
        del self.workflow_strategy[workflow_name]
