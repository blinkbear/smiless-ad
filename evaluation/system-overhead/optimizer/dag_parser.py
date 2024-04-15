import networkx as nx
import pandas as pd
from .function_profiler import FunctionProfiler
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class DAGParser:
    def __init__(self):
        self.workflow_dags = {}
        self.workflow_sla = {}
        self.function_info = {}
        self.function_profiler = FunctionProfiler()
        self.set_function_info()

    def set_function_info(self):
        with open(os.path.join(SCRIPT_DIR, "config.json"), "r") as f:
            function_info = eval(f.read())
        for function_name in function_info:
            (
                function_cold_start_time,
                function_gpu_trans_time,
            ) = self.function_profiler.get_cold_start_time(function_name)
            function_gpu_execution_time = self.function_profiler.get_gpu_execution_time(
                function_name
            )
            function_gpu_cs_extra_time = self.function_profiler.get_gpu_cs_extra_time(
                function_name
            )
            function_cpu_cs_extra_time = self.function_profiler.get_cpu_cs_extra_time(
                function_name
            )
            function_info[function_name]["cold_start_time"] = function_cold_start_time
            function_info[function_name]["gpu_trans_time"] = function_gpu_trans_time
            function_info[function_name]["gpu_cs_extra_time"] = (
                function_gpu_cs_extra_time
            )
            function_info[function_name]["cpu_cs_extra_time"] = (
                function_cpu_cs_extra_time
            )
            function_info[function_name]["gpu_running_time"] = (
                function_gpu_execution_time
            )
        self.function_info = function_info

    def generate_workflow_dag(self, workflow_name, workflow_dag_json, workflow_length):
        workflow_dag = nx.DiGraph()
        self.workflow_sla[workflow_name] = 10
        count = 0
        for node in workflow_dag_json["node"]:
            if node in workflow_dag.nodes:
                node = node + "-1"
            types = node.split("-")[0]
            function_info = self.function_info[types]
            (
                cpu_running_time,
                cpu_cost,
                gpu_cost,
                cpu_keep_alive_cost_unit,
                gpu_keep_alive_cost_unit,
            ) = self.function_profiler.get_cpu_cost_running_time(
                function_info["types"],
                1,
                1,
                function_info["image_size"],
            )

            workflow_dag.add_node(
                node,
                image_size=function_info["image_size"],
                types=function_info["types"],
                gpu_running_time=function_info["gpu_running_time"],
                cold_start_time=function_info["cold_start_time"],
                gpu_trans_time=function_info["gpu_trans_time"],
                gpu_cs_extra_time=function_info["gpu_cs_extra_time"],
                cpu_cs_extra_time=function_info["cpu_cs_extra_time"],
                cpu_running_time=cpu_running_time,
                cpu_cost=cpu_cost,
                gpu_cost=gpu_cost,
                cpu_keep_alive_cost_unit=cpu_keep_alive_cost_unit,
                gpu_keep_alive_cost_unit=gpu_keep_alive_cost_unit,
            )
            count += 1
            if count == workflow_length:
                break
        for edge in workflow_dag_json["edges"]:
            if (
                edge["source"] in workflow_dag.nodes
                and edge["target"] in workflow_dag.nodes
            ):
                workflow_dag.add_edge(edge["source"], edge["target"])
        self.workflow_dags[workflow_name] = workflow_dag

    def get_workflow_SLA(self, workflow_name):
        return self.workflow_sla[workflow_name]

    def _get_node_depth(self, node, graph):
        depth = 0
        for parent in graph.predecessors(node):
            depth = max(depth, self._get_node_depth(parent, graph))
        return depth + 1

    def longest_path(self, graph, start, end):
        from collections import defaultdict

        # 创建一个字典来保存每个节点的最长路径
        longest_paths = defaultdict(lambda: float("-inf"))
        # 设置起始节点的最长路径为0
        longest_paths[start] = 0
        # 创建一个字典来保存每个节点的前驱节点
        predecessors = {}

        # 按照拓扑排序的顺序遍历节点
        for node in nx.topological_sort(graph):
            # 遍历节点的所有邻居节点
            for neighbor in graph[node]:
                # 计算最长路径
                if longest_paths[neighbor] < longest_paths[node] + 1:
                    longest_paths[neighbor] = longest_paths[node] + 1
                    predecessors[neighbor] = node

        # 构建最长路径
        path = []
        current_node = end
        while current_node != start:
            path.append(current_node)
            current_node = predecessors[current_node]
        path.append(start)
        path.reverse()

        # 返回最长路径和路径长度
        return path, longest_paths[end]

    def _parse_simple_path(self, graph, root, leaf):
        tmp_graph = graph.copy()
        while True:
            path, path_length = self.longest_path(tmp_graph, root, leaf)
            yield path
            remove_node = []
            for i in list(path):
                if i == root or i == leaf:
                    continue
                G2 = tmp_graph.copy()
                G2.remove_node(i)
                if nx.has_path(G2, root, leaf):
                    remove_node.append(i)
                    tmp_graph.remove_node(i)
            if len(remove_node) == 0:
                break

    def __decompose_parallel_graph(self, graph: nx.DiGraph, graph_df: pd.DataFrame):
        """
        Decompose the parallel call graph to several call chain.
        """
        sub_graphs = []
        root = [n for n, d in graph.in_degree() if d == 0]
        leaves = [n for n, d in graph.out_degree() if d == 0]
        for r in root:
            for leaf in leaves:  # noqa: E741
                # sub_graphs += list(all_simple_paths(graph, r, l))

                sub_graphs = self._parse_simple_path(graph, r, leaf)
        graph_dfs = []
        for sub_graph in sub_graphs:
            sub_graph_df = (
                graph_df[graph_df["node"].isin(sub_graph)].copy().reset_index(drop=True)
            )
            graph_dfs.append(sub_graph_df)
        # print(graph_dfs)
        return graph_dfs

    def _get_node_qps(self, node, graph: nx.DiGraph):
        return max(graph.in_degree(node), 1)

    def _graph_to_df(self, graph: nx.DiGraph):
        """
        Reconstruct the graph to a dataframe, and use the max depth of each node as the depth of the node.
        """
        for k, v in graph.nodes.items():
            v["depth"] = self._get_node_depth(k, graph)
            v["qps"] = self._get_node_qps(k, graph)

        graph_list = []
        for k, v in graph.nodes.items():
            graph_list.append(
                (
                    k,
                    v["image_size"],
                    v["depth"],
                    v["qps"],
                    v["gpu_running_time"],
                    v["cold_start_time"],
                    v["gpu_trans_time"],
                    v["gpu_cs_extra_time"],
                    v["cpu_cs_extra_time"],
                    v["types"],
                    v["cpu_running_time"],
                    v["cpu_cost"],
                    v["gpu_cost"],
                    v["cpu_keep_alive_cost_unit"],
                    v["gpu_keep_alive_cost_unit"],
                )
            )
        graph_df = pd.DataFrame(
            graph_list,
            columns=[
                "node",
                "image_size",
                "depth",
                "qps",
                "gpu_running_time",
                "cold_start_time",
                "gpu_trans_time",
                "gpu_cs_extra_time",
                "cpu_cs_extra_time",
                "types",
                "cpu_running_time",
                "cpu_cost",
                "gpu_cost",
                "cpu_keep_alive_cost_unit",
                "gpu_keep_alive_cost_unit",
            ],
        )
        graph_df = graph_df.sort_values(by=["depth"])
        entry_node = graph_df.loc[0, "node"]
        return graph_df, entry_node

    def parse_graph_to_df(self, workflow_name):
        graph = self.workflow_dags[workflow_name]
        graph_df, entry_node = self._graph_to_df(graph)

        graph_dfs = self.__decompose_parallel_graph(graph, graph_df)

        return graph_df, graph_dfs, entry_node

    def get_workflow_functions(self, workflow_name):
        """
        Get the functions in the workflow.
        """
        return self.workflow_dags[workflow_name].nodes

    def delete_workflow(self, workflow_name):
        """
        Delete the workflow.
        """
        del self.workflow_dags[workflow_name]
        del self.workflow_sla[workflow_name]
