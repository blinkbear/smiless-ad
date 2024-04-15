import base64
import datetime
from time import sleep
import uuid
from invoker.invoke import INVOKER_FACTORY
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, ALL_COMPLETED
import os
import pandas as pd
from evaluation.end_to_end.config_parser import ConfigParser
from utils.kube_operator import KubeOperator
from utils.prom_operator import PodKeepAliveInfo
import warnings
from multiprocessing import Manager
import time
import logging
import networkx as nx
import argparse


warnings.filterwarnings("ignore")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


# set lock for invocation
MAX_CONCURRENCY = 8
MAX_FUNCTION_CONCURRENT = 10
SOURCEDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = "config.json"
RESULT_DIR = "data/result"
OLD_RESULT_DIR = "data/old_result"

monitor_running = True

CPU_COST = {
    # aws Ohio region
    1: 0.034,  # c6g.medium 1c2G10Gbps
    2: 0.068,  # c6g.large 2c4G10Gbps
    4: 0.136,  # c6g.xlarge 4c8G10Gbps
    8: 0.272,  # c6g.2xlarge 8c16G10Gbps
    16: 0.544,  # c6g.4xlarge 16c32G10Gbps
    32: 1.088,  # c6g.8xlarge 32c64G25Gbps
}
GPU_COST = {
    1: 0.306  # p3.2xlarge 8c61G10Gbps+V100
}


function_names = [
    "distilbert",
    "imagerecognition",
    "textgeneration",
    "translation",
    "facerecognition",
    "humanactivitypose",
    "nameentityrecognition",
    "objectdetection",
    "questionanswering",
    "speechrecognition",
    "texttospeech",
    "topicmodeling",
    "wordchunking",
]


class Result:
    def __init__(self, function_name, result, invoke_num):
        function_name = function_name.split("-")[0]
        self.function_name = function_name
        self.pod_name = result["runtime"]["pod_name"]
        self.start_time = result["runtime"]["start_time"]
        self.progress_time = self.__ms_to_s(result["runtime"]["progress_time"])
        self.device = result["runtime"]["device"]
        self.resource_quantity = float(result["runtime"]["resource_quantity"])
        self.model_predict_time = self.__ms_to_s(result["runtime"]["runtime"])
        self.model_load_time = self.__ms_to_s(result["runtime"]["model_load_time"])
        self.model_trans_time = self.__ms_to_s(result["runtime"]["model_trans_time"])
        self.total_time = self.__ms_to_s(result["query_time"])
        self.invoke_num = invoke_num
        self.warm_cold = result["runtime"]["warm_cold"]
        self.running_cost = 0
        if self.device == "cpu":
            self.running_cost = (
                CPU_COST[self.resource_quantity]
                * self.resource_quantity
                * self.progress_time
            )
            self.model_load_cost = 0
        if self.device == "cuda":
            self.running_cost = GPU_COST[1] * self.model_predict_time
            self.model_load_cost = (
                CPU_COST[self.resource_quantity]
                * self.resource_quantity
                * (self.model_load_time + self.model_trans_time)
            )

    def __ms_to_s(self, millisecond):
        return round(millisecond / 1000 / 1000, 4)


def invoke(args):
    """
    invoke function, if there are multiple functions in the chain, invoke them one by one
    Args:
        function_name: the function name
        input_size: the input size of the function
    """
    import time

    (
        function_name,
        header,
        workflow_name,
        SLA,
        invoke_num,
        workflow_round,
        request_ids,
    ) = args
    header = eval(header)
    header["X-Workflow-Round"] = f"{workflow_round}"
    header = str(header)
    logging.info(
        f"function name is {function_name}, invoke num is {invoke_num}, invoke time is {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}"
    )
    function_type = function_name.split("-")[0]
    invoker = INVOKER_FACTORY[function_type]()
    results = []
    while True:
        try:
            with ThreadPoolExecutor(max_workers=invoke_num * 2) as executor:
                futures = [
                    executor.submit(
                        invoker.invoke,
                        (
                            url,
                            namespace,
                            1,
                            header,
                            workflow_name,
                            SLA * 4,
                            request_ids[(workflow_round, i)],
                        ),
                    )
                    for i in range(invoke_num)
                ]
                for future in as_completed(futures):
                    res = future.result()
                    results.append(Result(function_name, res, invoke_num))
                    logging.info(
                        "ended function name is {}, invoke num is {}, invoke time is {}".format(
                            function_name,
                            invoke_num,
                            datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3],
                        )
                    )
            if "pod_name" not in res["runtime"]:
                logging.warning("no pod_name for function", function_name)
                continue
            break
        except Exception as e:
            logging.error(f"function {function_name} has error {e}")
            time.sleep(0.5)
            continue
    return results


def get_invoke_patterns(test_type=None):
    if test_type == "bursty":
        invoke_pattern_path = os.path.join(
            SOURCEDIR, "resources/selected_function_bursty.csv"
        )
    else:
        invoke_pattern_path = os.path.join(
            SOURCEDIR, "resources/selected_function_bursty.csv"
        )
    patterns = pd.read_csv(invoke_pattern_path)
    return patterns


def get_invoke_pattern(function_name, function_patterns):
    # get the pattern for the given function
    function_name = function_name.split("-")[0]
    function_pattern = function_patterns[
        function_patterns["HashFunction"] == function_name
    ]
    # convert the pattern row to a list
    invoke_pattern = function_pattern.iloc[0].tolist()
    return invoke_pattern


def save_old_result():
    if global_args.save_dir == "":
        result_file = os.path.join(RESULT_DIR, "result_total.csv")
        if os.path.exists(result_file):
            new_result_file = os.path.join(
                OLD_RESULT_DIR, "result_" + str(datetime.datetime.now()) + ".csv"
            )
            os.rename(result_file, new_result_file)
    else:
        return


def save_result(
    result,
):
    if global_args.save_dir == "":
        result_file = os.path.join(SOURCEDIR, global_args.test_type, "e2e_result.csv")
        result.to_csv(result_file, index=False, mode="w")
    else:
        result_file = os.path.join(global_args.save_dir, "e2e_result.csv")
        result.to_csv(result_file, index=False, mode="w")


def invocation_args_generator(workflow_graph):
    def _get_node_depth(node, graph):
        depth = 0
        for parent in graph.predecessors(node):
            depth = max(depth, _get_node_depth(parent, graph))
        return depth + 1

    graph = nx.DiGraph()
    graph.add_nodes_from(workflow_graph["node"])

    for edge in workflow_graph["edges"]:
        graph.add_edge(edge["source"], edge["target"])
    depth = {}
    for k, v in graph.nodes.items():
        function_depth = _get_node_depth(k, graph)
        if function_depth in depth:
            depth[function_depth].append(k)
        else:
            depth[function_depth] = [k]
    return depth


def invoke_workflow(args):
    (
        headers,
        workflow_name,
        invoke_round,
        workflow_running_result,
        invoke_num,
        depth_func_name,
        request_ids,
        SLA,
    ) = args
    pool = ThreadPoolExecutor(max_workers=MAX_FUNCTION_CONCURRENT)
    workflow_function_running_result = []
    st = time.time()
    for i in range(1, len(depth_func_name) + 1):
        request_args = []
        for function_name in depth_func_name[i]:
            request_args.append(
                (
                    function_name,
                    str(headers),
                    workflow_name,
                    SLA,
                    invoke_num,
                    invoke_round,
                    request_ids,
                )
            )
        for result in pool.map(invoke, request_args):
            for r in result:
                workflow_function_running_result.append(r)
    et = time.time()
    workflow_running_result.append(
        (
            workflow_function_running_result,
            [invoke_round] * len(workflow_function_running_result),
            [et - st] * len(workflow_function_running_result),
        )
    )


def start_invoke(workflow_name, workflow_graph, namespace, SLA):
    """
    start invoke function. The method will first parse the config file to get the function chains that start with the function_name (the function chains are defined in config.functions_chain). Then it will invoke the function chains parallel (the concurrency is defined in config.concurrency). Total invocation will repeat 10 times for each function chain. The result will be saved in df.
    Args:
        function_name: the function name
        df: the dataframe to save the result
        input_size: the input size of the function, default is 10
        cpu: the cpu limit of the function, default is 2
        memory: the memory limit of the function, default is 4096
        gpu: the gpu percentage of the function, default is 20
    Return:
        result_df: the dataframe to save the result
    """

    headers = {
        "Content-Type": "text/plain",
        "X-Workflow": "true",
        "X-Workflow-Names": workflow_name,
    }
    depth_func_name = invocation_args_generator(workflow_graph, namespace)
    workflow_invoke_round = 0
    workflow_running_result = {
        "workflow_name": workflow_name,
        "workflow_running_time": [],
        "workflow_function_running_result": [],
        "round": [],
        "workflow_time": [],
    }
    invoke_pattern = get_invoke_pattern(depth_func_name[1][0], invoke_patterns)
    time.sleep(15)
    tasks = []
    workflow_running_result = Manager().list()
    request_ids = Manager().dict()
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as executor:
        for invoke_num in invoke_pattern[1:]:
            logging.info(f"invoke num is {invoke_num}")

            if invoke_num == 0:
                logging.info(f"skip round {workflow_invoke_round}")
                workflow_invoke_round = workflow_invoke_round + 1
                sleep(interval_time_unit)
                continue
            for i in range(invoke_num):
                request_ids[(workflow_invoke_round, i)] = str(uuid.uuid4())
            args = (
                headers,
                workflow_name,
                workflow_invoke_round,
                workflow_running_result,
                invoke_num,
                depth_func_name,
                request_ids,
                SLA,
            )
            tasks.append(executor.submit(invoke_workflow, args))
            sleep(interval_time_unit)
            workflow_invoke_round = workflow_invoke_round + 1

    wait(tasks, return_when=ALL_COMPLETED)
    workflow_running_result = list(workflow_running_result)
    for r in workflow_running_result:
        workflow_running_result["workflow_running_time"].append(r[2])
        workflow_running_result["workflow_function_running_result"].append(r[0])
        workflow_running_result["round"].append(r[1])
    et = time.time()
    workflow_running_result["workflow_time"].append([et] * len(workflow_running_result))

    return workflow_running_result


def generate_workflow_configmap(
    workflow_name,
    workflow_graph_source,
    function_info,
    optimizer,
    SLA,
    kube_operator: KubeOperator,
):
    import copy

    workflow_graph = copy.deepcopy(workflow_graph_source)
    workflow_graph["SLA"] = SLA
    workflow_function_info = {}
    workflow_annotation = {
        "faas_workflow_name": workflow_name,
        "optimizer": optimizer,
        "test_type": global_args.test_type,
    }
    for node in workflow_graph["node"]:
        node = node.split("-")[0]
        workflow_function_info[node] = function_info[node]
    workflow_data = workflow_graph
    workflow_data["function_info"] = workflow_function_info
    workflow_data["interval_time_unit"] = interval_time_unit
    workflow_update_data = base64_encode(workflow_data)
    kube_operator.create_or_update_workflow_configmap(
        workflow_name, workflow_annotation, workflow_update_data
    )
    return workflow_data


def base64_encode(workflow_data):
    for key in workflow_data:
        workflow_data[key] = base64.b64encode(
            str(workflow_data[key]).encode("utf-8")
        ).decode("utf-8")
    return workflow_data


def workflow_invoker(args):
    import time

    (
        workflow_name,
        workflow_graph,
        namespace,
        kube_operator,
        function_info,
        SLA,
        optimizer,
    ) = args
    generate_workflow_configmap(
        workflow_name, workflow_graph, function_info, optimizer, SLA, kube_operator
    )
    result = start_invoke(workflow_name, workflow_graph, namespace, SLA)
    result["optimizer"] = optimizer
    result["sla"] = SLA
    logging.info(f"remove configmap {workflow_name}")
    time.sleep(5)
    kube_operator.delete_workflow_configmap(workflow_name)
    return result


def parse_result(pod_keep_alive_info: PodKeepAliveInfo, total_results, st, optimizer):
    result_df = pd.DataFrame()
    for k in range(len(total_results)):
        start_time = float("inf")
        tmp_result = total_results[k]
        tmp_df = pd.DataFrame(
            columns=[
                "workflow_name",
                "optimizer",
                "workflow_running_time",
                "function_name",
                "pod_name",
                "process_time",
                "device",
                "resource_quantity",
                "model_predict_time",
                "model_load_time",
                "model_trans_time",
                "running_cost",
                "keep_alive_time",
                "keep_alive_cost",
                "total_cost",
                "total_time",
                "sla",
                "round",
                "node",
            ]
        )
        if tmp_result["optimizer"] != optimizer:
            continue
        sla = tmp_result["sla"]
        workflow_name = tmp_result["workflow_name"]
        workflow_running_time = tmp_result["workflow_running_time"]
        workflow_time = tmp_result["workflow_time"][0][0]
        workflow_function_running_result = tmp_result[
            "workflow_function_running_result"
        ]
        result_round = tmp_result["round"]
        et = datetime.datetime.now()
        all_pod_result = pod_keep_alive_info.get_all_pod_from_ns(st, et, workflow_time)
        all_pod_result.to_csv("all_pod_result.csv", index=False)
        # get running pod cost
        for i in range(len(workflow_function_running_result)):
            for j in range(len(workflow_function_running_result[i])):
                if workflow_function_running_result[i][j].start_time < start_time:
                    start_time = workflow_function_running_result[i][j].start_time
                keep_alive_time = all_pod_result[
                    all_pod_result["pod_name"]
                    == workflow_function_running_result[i][j].pod_name
                ]["pod_exec_time"].values[0]
                node = all_pod_result[
                    all_pod_result["pod_name"]
                    == workflow_function_running_result[i][j].pod_name
                ]["node"].values[0]

                # delete the pod from all_pod_result
                if workflow_function_running_result[i][j].device == "cpu":
                    for k, v in CPU_COST.items():
                        if k < workflow_function_running_result[i][j].resource_quantity:
                            continue
                        keep_alive_cost = CPU_COST[k] * keep_alive_time
                        break

                else:
                    keep_alive_cost = GPU_COST[1] * keep_alive_time
                tmp = pd.DataFrame(
                    {
                        "workflow_name": [workflow_name],
                        "optimizer": [optimizer],
                        "sla": [sla],
                        "workflow_running_time": [workflow_running_time[i][j]],
                        "function_name": [
                            workflow_function_running_result[i][j].function_name
                        ],
                        "pod_name": [workflow_function_running_result[i][j].pod_name],
                        "warm_cold": [workflow_function_running_result[i][j].warm_cold],
                        "process_time": [
                            workflow_function_running_result[i][j].progress_time
                        ],
                        "device": [workflow_function_running_result[i][j].device],
                        "resource_quantity": [
                            workflow_function_running_result[i][j].resource_quantity
                        ],
                        "model_predict_time": [
                            workflow_function_running_result[i][j].model_predict_time
                        ],
                        "model_load_time": [
                            workflow_function_running_result[i][j].model_load_time
                        ],
                        "model_trans_time": [
                            workflow_function_running_result[i][j].model_trans_time
                        ],
                        "total_time": [
                            workflow_function_running_result[i][j].total_time
                        ],
                        "running_cost": [
                            workflow_function_running_result[i][j].running_cost
                        ],
                        "model_load_cost": [
                            workflow_function_running_result[i][j].model_load_cost
                        ],
                        "keep_alive_time": [keep_alive_time],
                        "keep_alive_cost": [keep_alive_cost],
                        "invoke_num": [
                            workflow_function_running_result[i][j].invoke_num
                        ],
                        "round": [result_round[i][j]],
                        "node": [node],
                    }
                )
                tmp["running_cost"] = round(tmp["running_cost"], 4)
                tmp["keep_alive_cost"] = round(tmp["keep_alive_cost"], 4)
                tmp["total_cost"] = tmp["running_cost"] + tmp["keep_alive_cost"]
                tmp_df = pd.concat([tmp_df, tmp], ignore_index=True)
        result_df = pd.concat([result_df, tmp_df], ignore_index=True)
        tmp_df = pd.DataFrame(
            columns=[
                "workflow_name",
                "optimizer",
                "workflow_running_time",
                "function_name",
                "pod_name",
                "process_time",
                "device",
                "resource_quantity",
                "model_predict_time",
                "model_load_time",
                "model_trans_time",
                "running_cost",
                "keep_alive_time",
                "keep_alive_cost",
                "total_cost",
                "total_time",
                "sla",
                "round",
                "node",
            ]
        )

        all_pod_result = all_pod_result[
            ~all_pod_result["pod_name"].isin(result_df["pod_name"].values)
        ]
        if all_pod_result.shape[0] != 0:
            for index, row in all_pod_result.iterrows():
                keep_alive_time = row["pod_exec_time"]
                # delete the pod from all_pod_result
                if row["container"] not in result_df["function_name"].values:
                    continue
                resource_quantity = result_df[
                    result_df["function_name"] == row["container"]
                ]["resource_quantity"].min()
                device = result_df[result_df["function_name"] == row["container"]][
                    "device"
                ].values[0]
                model_load_time = round(
                    result_df[result_df["function_name"] == row["container"]][
                        "model_load_time"
                    ].mean(),
                    4,
                )
                model_trans_time = round(
                    result_df[result_df["function_name"] == row["container"]][
                        "model_trans_time"
                    ].mean(),
                    4,
                )
                model_load_time = round(
                    result_df[result_df["function_name"] == row["container"]][
                        "model_load_time"
                    ].mean(),
                    4,
                )
                if device == "cpu":
                    for k, v in CPU_COST.items():
                        if k < resource_quantity:
                            continue
                        keep_alive_cost = v * keep_alive_time
                        break
                else:
                    keep_alive_cost = GPU_COST[1] * keep_alive_time
                tmp = pd.DataFrame(
                    {
                        "workflow_name": [workflow_name],
                        "optimizer": [optimizer],
                        "sla": [sla],
                        "workflow_running_time": [-1],
                        "function_name": [row["container"]],
                        "pod_name": [row["pod_name"]],
                        "warm_cold": ["cold"],
                        "process_time": [0],
                        "device": [device],
                        "resource_quantity": [resource_quantity],
                        "model_predict_time": [0],
                        "model_load_time": [model_load_time],
                        "model_trans_time": [model_trans_time],
                        "total_time": [keep_alive_time],
                        "running_cost": [0],
                        "model_load_cost": [0],
                        "keep_alive_time": [keep_alive_time],
                        "keep_alive_cost": [keep_alive_cost],
                        "invoke_num": [0],
                        "node": [row["node"]],
                        "round": [-1],
                    }
                )
                tmp["running_cost"] = round(tmp["running_cost"], 4)
                tmp["keep_alive_cost"] = round(tmp["keep_alive_cost"], 4)
                tmp["total_cost"] = tmp["running_cost"] + tmp["keep_alive_cost"]
                tmp_df = pd.concat([tmp_df, tmp], ignore_index=True)
        result_df = pd.concat([result_df, tmp_df], ignore_index=True)
    return result_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_ip", type=str)
    parser.add_argument("--slas", type=str, default="-1")
    parser.add_argument("--optimizer", type=str, default="all")
    parser.add_argument("--workflow", type=str, default="all")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--test_type", type=str, default="end_to_end")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    global_args = parse_args()
    start_time = datetime.datetime.now()
    namespace = "openfaas-fn"
    prom_url = f"http://{global_args.master_ip}:30091"
    save_old_result()
    config_path = os.path.join(SOURCEDIR, "end_to_end", "config.json")
    config = ConfigParser(config_path)
    (
        url,
        gateway,
        passwd,
    ) = config.get_openfaas_config()
    kube_operator = KubeOperator(namespace, SOURCEDIR)
    function_info = config.get_functions_info()
    pod_keep_alive_info = PodKeepAliveInfo(
        start_time=start_time, prom_url=prom_url, namespace=namespace
    )
    interval_time_unit = config.get_interval_time_unit()
    invoke_patterns = get_invoke_patterns(test_type=global_args.test_type)
    optimizers = config.get_optimizers()
    if global_args.optimizers != "all":
        optimizers = global_args.optimizers.split(",")
    results = pd.DataFrame()
    workflow_graph = config.get_workflows()
    if global_args.slas == "-1":
        slas = config.get_slas()
    else:
        slas = global_args.slas.split(",")
    total_results = []
    for SLA in slas:
        for optimizer in optimizers:
            workflow_invoker_args = []
            logging.info(f"optimizer is {optimizer}")
            workflow_names = []
            st = datetime.datetime.now()
            for workflow_name in workflow_graph:
                workflow_invoker_args.append(
                    (
                        workflow_name,
                        workflow_graph[workflow_name],
                        namespace,
                        kube_operator,
                        function_info,
                        SLA,
                        optimizer,
                    )
                )
                workflow_names.append(workflow_name)
            pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)
            for result in pool.map(workflow_invoker, workflow_invoker_args):
                total_results.append(result)
            sleep(10)
            result_df = parse_result(
                pod_keep_alive_info, function_info, total_results, st, optimizer
            )
            results = pd.concat([results, result_df], ignore_index=True)
            save_result(
                results,
            )
            logging.info(
                f"optimizer {optimizer} cost time {datetime.datetime.now() - st}"
            )
