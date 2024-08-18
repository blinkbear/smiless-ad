import pandas as pd
from offline_profiler.function_resource_profiler import FunctionProfiler
from .path_search import PathSearch
from .optimizer import Optimizer
from online_predictor.online_predictor import OnlinePredictor
from cache.cache import Cache
import logging
import numpy as np
from numpy import fft

log = logging.getLogger("rich")


class SMIless(Optimizer):
    def __init__(
        self,
        cache: Cache,
        function_profilers: FunctionProfiler,
        online_predictor: OnlinePredictor,
    ):
        """
        Args:
            cache (Cache): An instance of the Cache class.
            function_profilers (FunctionProfiler): An instance of the FunctionProfiler class.
        """

        self.function_profilers = function_profilers

        # Set the number of available CPU resources.
        self.available_cpu_resource = 1

        self.cache = cache

        self.delay = 0.0

        self.path_search = PathSearch()

        self.workflow_strategy = {}
        self.SLAs = {}
        self.online_predictor = online_predictor

    def set_shared_node_unique(self, graph_df, workflow_name):
        for i, row in graph_df:
            graph_df.loc[
                (graph_df["depth"] == i & graph_df["node"] == row["node"]), "node"
            ] = row["node"] + workflow_name + str(i)
        return graph_df

    def update_workflow_running_plan_df(self, workflow_name):
        graph_df, graph_dfs = self.workflow_strategy[workflow_name]
        SLA = self.SLAs[workflow_name]
        interval_time_unit = self.interval_time_unit
        graph_df, nodes = self.get_workflow_running_plan_df(
            workflow_name,
            graph_df,
            graph_dfs,
            SLA,
            interval_time_unit,
        )
        return graph_df, nodes

    def get_inter_arrival_time(self, updated, entry_node, interval_time_unit):
        # if not updated:
        #     IT = 0
        # else:
        IT = self.online_predictor.get_predicted_inter_arrival_time(entry_node)
        return IT

    def _get_running_plan(self, df, shared_nodes, SLA):
        self.path_search.init(df, shared_nodes, SLA, self.IT)
        # Get the running plan DataFrame using the A* search algorithm
        df = self.path_search.get_running_plan_df()
        return df

    def _get_shared_nodes(self, nodes, updated):
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
        return shared_nodes, shared_nodes_resource_quantity

    def get_workflow_running_plan_df(
        self,
        workflow_name,
        graph_df,
        graph_dfs,
        SLA,
        interval_time_unit,
    ):
        # Get list of nodes in the call graph
        updated = workflow_name in self.workflow_strategy
        self.IT = self.get_inter_arrival_time(
            updated, graph_df["node"].tolist()[0], interval_time_unit
        )
        print(f"inter arrival time is {self.IT}")
        self.SLAs[workflow_name] = SLA
        self.interval_time_unit = interval_time_unit
        self.workflow_strategy[workflow_name] = (graph_df, graph_dfs)
        import datetime

        st = datetime.datetime.now()
        nodes = graph_df["node"].tolist()
        if self.IT == 0 and updated:
            logging.info(f"workflow {workflow_name} no need to update running plan")
            return None, None
        shared_nodes, shared_nodes_resource_quantity = self._get_shared_nodes(
            nodes, updated
        )
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
            # Calculate the keep-alive time for each function in the DataFrame
            df = self.get_keep_alive_time(df)
            # Initialize the A* search algorithm with the DataFrame and SLA
            df = self._get_running_plan(df, shared_nodes, SLA)
            logging.debug(f"running plan is {df}")
            if df is None:
                logging.warning("No feasible solution")
                return None, None
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

    def get_keep_alive_time(self, df):
        """
        Calculates the keep alive time for the entry node in the given DataFrame.

        Args:
            df (pandas.DataFrame): A DataFrame containing the nodes and their depths.

        Returns:
            pandas.DataFrame: A copy of the input DataFrame with a new 'keep_alive_time' column.
        """
        entry_node = df[df["depth"] == 1]["node"].values[0]
        keep_alive_time = (
            self.function_profilers.calc_keep_alive_time(entry_node)
            * self.interval_time_unit
        )
        df["keep_alive_time"] = keep_alive_time
        return df

    def get_shared_node_keep_alive_time(self, node, current_entry_node):
        workflow_names = self.cache.get_shared_nodes_to_workflow_name(node)
        entry_nodes = [current_entry_node]
        for workflow_name in workflow_names:
            entry_node = self.cache.get_workflow_entries(workflow_name)
            entry_nodes.append(entry_node)
        keep_alive_time = (
            self.function_profilers.calc_shared_keep_alive_time(entry_nodes)
            * self.interval_time_unit
        )
        return keep_alive_time

    def get_keep_alive_time_for_each_function(self, df, shared_nodes):
        """
        Calculates the keep alive time for each row in the data frame based on the device type.

        Args:
            self: object
            df (pandas.DataFrame): The data frame containing columns 'device', 'cpu_running_time',
            'gpu_running_time', 'cold_start_time' and 'resource_quantity'.

        Returns:
            pandas.DataFrame: The input data frame with an additional column 'keep_alive_time' and an
            existing column 'keep_alive_resource' that is the same as the 'resource_quantity' column.
        """

        def __get_keep_alive_time(row):
            if row["prewarm_window"] == 0:
                return row["keep_alive_time"]
                # return 100000

            else:
                return -1
                # return 10000

        df["keep_alive_time"] = df.apply(lambda x: __get_keep_alive_time(x), axis=1)
        df["keep_alive_resource"] = df["resource_quantity"]
        return df

    def remove_workflow(self, workflow_name):
        del self.workflow_strategy[workflow_name]


class SMIlessFIP(SMIless):
    def __init__(
        self,
        cache: Cache,
        function_profilers: FunctionProfiler,
        online_predictor: OnlinePredictor,
    ):
        super().__init__(cache, function_profilers, online_predictor)

    def fourier_extrapolation(self, x, n_predict):
        n = x.size
        n_harm = self.harmonics  # number of harmonics in model
        t = np.arange(0, n)
        p = np.polyfit(t, x, 1)  # find linear trend in x
        x_notrend = x - p[0] * t  # detrended x
        x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
        f = fft.fftfreq(n)  # type: ignore # frequencies
        indexes = list(range(n))
        # sort indexes by frequency, lower -> higher
        indexes.sort(key=lambda i: np.absolute(f[i]))

        t = np.arange(1, n + n_predict)
        restored_sig = np.zeros(t.size)
        for i in indexes[: 1 + n_harm * 2]:
            ampli = np.absolute(x_freqdom[i]) / n  # amplitude
            phase = np.angle(x_freqdom[i])  # phase
            restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
        return restored_sig + p[0] * t

    def get_predict_values(self, entry_node, interval_time_unit):
        n_predict = 1
        training_trace = self.online_predictor.get_history_inter_arrival_time(
            entry_node
        )
        extrapolation = self.fourier_extrapolation(training_trace, n_predict)
        pred_value = extrapolation[len(extrapolation) - 1] * interval_time_unit
        IT, keep_alive_time = (
            self.online_predictor.update_invocation_number_inter_arrival_time_from_gateway(
                entry_node
            )
        )
        return pred_value, keep_alive_time

    def get_workflow_running_plan_df(
        self,
        workflow_name,
        graph_df,
        graph_dfs,
        SLA,
        interval_time_unit,
    ):
        # Get list of nodes in the call graph
        updated = workflow_name in self.workflow_strategy
        self.IT, self.keep_alive_time = self.get_predict_values(
            graph_df["node"].tolist()[0], interval_time_unit
        )

        self.interval_time_unit = interval_time_unit
        self.workflow_strategy[workflow_name] = (graph_df, graph_dfs)
        import datetime

        st = datetime.datetime.now()
        nodes = graph_df["node"].tolist()
        if self.IT == 0 and updated:
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
            # Calculate the keep-alive time for each function in the DataFrame
            df = self.get_keep_alive_time(df)
            # Initialize the A* search algorithm with the DataFrame and SLA
            df = self._get_running_plan(df, shared_nodes, SLA)
            logging.debug(f"running plan is {df}")
            if df is None:
                logging.warning("No feasible solution")
                return None, None
            df = self.get_keep_alive_time_for_each_function(df, shared_nodes)
            df["prewarm_window"] = self.IT
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

    def get_keep_alive_time(self, df):
        """
        Calculates the keep alive time for the entry node in the given DataFrame.

        Args:
            df (pandas.DataFrame): A DataFrame containing the nodes and their depths.

        Returns:
            pandas.DataFrame: A copy of the input DataFrame with a new 'keep_alive_time' column.
        """
        df["keep_alive_time"] = self.keep_alive_time
        return df

    def get_keep_alive_time_for_each_function(self, df, shared_nodes):
        """
        Calculates the keep alive time for each row in the data frame based on the device type.

        Args:
            self: object
            df (pandas.DataFrame): The data frame containing columns 'device', 'cpu_running_time',
            'gpu_running_time', 'cold_start_time' and 'resource_quantity'.

        Returns:
            pandas.DataFrame: The input data frame with an additional column 'keep_alive_time' and an
            existing column 'keep_alive_resource' that is the same as the 'resource_quantity' column.
        """

        df["keep_alive_time"] = self.keep_alive_time
        df["keep_alive_resource"] = df["resource_quantity"]
        return df


class SMIlessAzure(SMIless):
    def __init__(
        self,
        cache: Cache,
        function_profilers: FunctionProfiler,
        online_predictor: OnlinePredictor,
    ):
        super().__init__(cache, function_profilers, online_predictor)

    def get_inter_arrival_time_azure(self, updated, entry_node, interval_time_unit):
        IT, keep_alive_time = (
            self.online_predictor.predict_inter_arrival_time_keep_alive_time_with_distribution(
                entry_node, interval_time_unit
            )
        )
        return IT, keep_alive_time

    def get_workflow_running_plan_df(
        self,
        workflow_name,
        graph_df,
        graph_dfs,
        SLA,
        interval_time_unit,
    ):
        # Get list of nodes in the call graph
        updated = workflow_name in self.workflow_strategy
        self.IT, self.keep_alive_time = self.get_inter_arrival_time_azure(
            updated, graph_df["node"].tolist()[0], interval_time_unit
        )

        self.interval_time_unit = interval_time_unit
        self.workflow_strategy[workflow_name] = (graph_df, graph_dfs)
        import datetime

        st = datetime.datetime.now()
        nodes = graph_df["node"].tolist()
        if self.IT == 0 and updated:
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
            # Calculate the keep-alive time for each function in the DataFrame
            df = self.get_keep_alive_time(df)
            # Initialize the A* search algorithm with the DataFrame and SLA
            df = self._get_running_plan(df, shared_nodes, SLA)
            logging.debug(f"running plan is {df}")
            if df is None:
                logging.warning("No feasible solution")
                return None, None
            df = self.get_keep_alive_time_for_each_function(df, shared_nodes)
            df["prewarm_window"] = self.IT
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

    def get_keep_alive_time(self, df):
        """
        Calculates the keep alive time for the entry node in the given DataFrame.

        Args:
            df (pandas.DataFrame): A DataFrame containing the nodes and their depths.

        Returns:
            pandas.DataFrame: A copy of the input DataFrame with a new 'keep_alive_time' column.
        """
        df["keep_alive_time"] = self.keep_alive_time
        return df

    def get_keep_alive_time_for_each_function(self, df, shared_nodes):
        """
        Calculates the keep alive time for each row in the data frame based on the device type.

        Args:
            self: object
            df (pandas.DataFrame): The data frame containing columns 'device', 'cpu_running_time',
            'gpu_running_time', 'cold_start_time' and 'resource_quantity'.

        Returns:
            pandas.DataFrame: The input data frame with an additional column 'keep_alive_time' and an
            existing column 'keep_alive_resource' that is the same as the 'resource_quantity' column.
        """

        df["keep_alive_time"] = self.keep_alive_time
        df["keep_alive_resource"] = df["resource_quantity"]
        return df


class SMIlessDFS(SMIless):
    def __init__(
        self,
        cache: Cache,
        function_profilers: FunctionProfiler,
        online_predictor: OnlinePredictor,
    ):
        super().__init__(cache, function_profilers, online_predictor)

    def _get_running_plan(self, df, shared_nodes, SLA):
        self.path_search.init(df, shared_nodes, SLA, self.IT)
        df = self.path_search.get_running_plan_df(search_method="dfs")
        return df


class SMIlessBFS(SMIless):
    def __init__(
        self,
        cache: Cache,
        function_profilers: FunctionProfiler,
        online_predictor: OnlinePredictor,
    ):
        super().__init__(cache, function_profilers, online_predictor)

    def _get_running_plan(self, df, shared_nodes, SLA):
        self.path_search.init(df, shared_nodes, SLA, self.IT)
        df = self.path_search.get_running_plan_df(search_method="bfs")
        return df


class SMIlessAstar(SMIless):
    def __init__(
        self,
        cache: Cache,
        function_profilers: FunctionProfiler,
        online_predictor: OnlinePredictor,
    ):
        super().__init__(cache, function_profilers, online_predictor)

    def _get_running_plan(self, df, shared_nodes, SLA):
        self.path_search.init(df, shared_nodes, SLA, self.IT)
        df = self.path_search.get_running_plan_df(search_method="Astar")
        return df


class SMIlessAug(SMIless):
    def __init__(
        self,
        cache: Cache,
        function_profilers: FunctionProfiler,
        online_predictor: OnlinePredictor,
    ):
        super().__init__(cache, function_profilers, online_predictor)

    def _get_running_plan(self, df, shared_nodes, SLA):
        self.path_search.init(df, shared_nodes, SLA, self.IT)
        df = self.path_search.get_running_plan_df(search_method="aug_smiless")
        return df
