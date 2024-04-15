import pandas as pd
from .path_search import PathSearch
from .optimizer import Optimizer
import logging
from .function_profiler import FunctionProfiler
import time

log = logging.getLogger("rich")


class SMIless(Optimizer):
    def __init__(
        self,
    ):
        """
        Args:
            cache (Cache): An instance of the Cache class.
            function_profilers (FunctionProfiler): An instance of the FunctionProfiler class.
        """

        self.function_profiler = FunctionProfiler()
        # Set the number of available CPU resources.
        self.available_cpu_resource = 1

        self.delay = 0.0

        self.path_search = PathSearch()

        self.workflow_strategy = {}

    def set_shared_node_unique(self, graph_df, workflow_name):
        for i, row in graph_df:
            graph_df.loc[
                (graph_df["depth"] == i & graph_df["node"] == row["node"]), "node"
            ] = row["node"] + workflow_name + str(i)
        return graph_df

    def update_workflow_running_plan_df(self, workflow_name, SLA, interval_time_unit):
        graph_df, graph_dfs = self.workflow_strategy[workflow_name]
        graph_df, nodes = self.get_workflow_running_plan_df(
            workflow_name,
            graph_df,
            graph_dfs,
            SLA,
            interval_time_unit,
        )
        return graph_df, nodes

    def get_workflow_running_plan_df(
        self,
        workflow_name,
        graph_df,
        graph_dfs,
        SLA,
    ):
        """
        This function calculates the running plan of the call graph.

        Args:
            graph_df (pandas.DataFrame): DataFrame containing information about the call graph.
            graph_dfs (List[pandas.DataFrame]): List of DataFrames containing information about each function in the call graph.
            SLA (float): Service Level Agreement (SLA) for the workflow.

        Returns:
            Tuple[pandas.DataFrame, List[str]]: A tuple containing a DataFrame with information about the running plan and a list of nodes in the call graph.
        """
        # Get list of nodes in the call graph
        updated = workflow_name in self.workflow_strategy
        IT = 10
        self.interval_time_unit = 10
        self.workflow_strategy[workflow_name] = (graph_df, graph_dfs)
        self.IT = IT

        nodes = graph_df["node"].tolist()
        if IT == 0 and updated:
            logging.info(f"workflow {workflow_name} no need to update running plan")
            return None, None
        shared_nodes = {}

        # Initialize an empty list to store DataFrames for each function in the call graph
        dfs = []
        execution_times = 0
        for df in graph_dfs:
            df["resource_quantity"] = self.available_cpu_resource
            # Calculate the keep-alive time for each function in the DataFrame
            df["keep_alive_time"] = 100
            # Initialize the A* search algorithm with the DataFrame and SLA
            self.path_search.init(df, shared_nodes, SLA, self.IT)
            # Get the running plan DataFrame using the A* search algorithm
            st = time.time()
            self.path_search.get_running_plan_df()
            et = time.time()
            df = self.path_search.available_solution
            if df is None:
                logging.warning("No feasible solution")
                return None, None, None
            df = self.get_keep_alive_time_for_each_function(df)
            dfs.append(df)
            execution_times += et - st
        df = pd.concat(dfs).drop_duplicates()
        result_df = pd.DataFrame(columns=df.columns)
        for node in nodes:
            tmp_df = df[df["node"] == node]
            tmp = tmp_df.max()
            result_df = pd.concat([result_df, tmp.to_frame().T])
        result_df = result_df.reset_index(drop=True)
        result_df["prewarm_window"] = abs(result_df["prewarm_window"])
        return result_df, nodes, execution_times

    def get_keep_alive_time_for_each_function(self, df):
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
            else:
                return -1

        df["keep_alive_time"] = df.apply(lambda x: __get_keep_alive_time(x), axis=1)
        df["keep_alive_resource"] = df["resource_quantity"]
        return df
