import base64
from concurrent.futures import ThreadPoolExecutor
from utils.kube_operator import KubeOperator
from kubernetes import client, watch
from kubernetes import config as kconf
import time
from dag_parser import DAGParser
from offline_profiler.function_resource_profiler import FunctionProfiler
from online_predictor.online_predictor import OnlinePredictor
from cache.invocation_infos import InvocationInfos
from online_predictor.prometheus_operator import PrometheusOperator
from autoscaler.autoscaler import AutoScaler
from optimizer.optimizer import OptimizerFactory
from cache.cache import Cache
from faas_operator.faas_operator import FaasOperator
import os
import logging
import warnings
from rich.logging import RichHandler
from rich.traceback import install
import torch.nn as nn

install(show_locals=True)

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler(level="INFO")],
)
log = logging.getLogger("rich")
warnings.filterwarnings("ignore")


class InvocationNumberModel(nn.Module):
    def __init__(self, n_in, n_out):
        super(InvocationNumberModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=n_in, hidden_size=30, num_layers=1, batch_first=True
        )
        self.linear = nn.Linear(30, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        out = self.linear(x)
        return out


class InterArrivalTimeModel(nn.Module):
    def __init__(self, n_in, n_out):
        super(InterArrivalTimeModel, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=n_in, hidden_size=128, num_layers=1, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=n_in, hidden_size=128, num_layers=1, batch_first=True
        )
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(128, 128, bias=False)
        self.linear2 = nn.Linear(128, n_out, bias=False)

    def forward(self, x):
        x1 = x[:, :, 0]
        x2 = x[:, :, 1]
        x1, _ = self.lstm1(x1)
        x2, _ = self.lstm2(x2)
        x2 = x1 + x2
        x2 = self.linear1(x2)
        x2 = self.linear2(x2)
        out = self.tanh(x2)

        return out


def get_function_info(function_info, function_profiler: FunctionProfiler):
    """
    Retrieves information about a function using a FunctionProfiler object.

    Args:
        function_info (dict): A dictionary containing information about the function.
        function_profiler (FunctionProfiler): An instance of the FunctionProfiler class used to profile the function.

    Returns:
        dict: The updated function_info dictionary with additional information about the function.
    """
    for function_name in function_info:
        (
            function_cold_start_time,
            function_gpu_trans_time,
        ) = function_profiler.get_cold_start_time(function_name)
        function_gpu_execution_time = function_profiler.get_gpu_execution_time(
            function_name
        )
        function_gpu_cs_extra_time = function_profiler.get_gpu_cs_extra_time(
            function_name
        )
        function_cpu_cs_extra_time = function_profiler.get_cpu_cs_extra_time(
            function_name
        )
        function_info[function_name]["cold_start_time"] = function_cold_start_time
        function_info[function_name]["gpu_trans_time"] = function_gpu_trans_time
        function_info[function_name]["gpu_cs_extra_time"] = function_gpu_cs_extra_time
        function_info[function_name]["cpu_cs_extra_time"] = function_cpu_cs_extra_time
        function_info[function_name]["gpu_running_time"] = function_gpu_execution_time
    return function_info


def base64_decode(data):
    return base64.b64decode(data)


def update_workflow_functions(
    workflow_name, workflow_running_plan_df, function_list, SLA
):
    if workflow_running_plan_df is None:
        log.warning("workflow has no available schedule plan to meet SLA")
    updated_functions_list = cache.cache_function_result(
        workflow_running_plan_df, workflow_name, SLA=SLA
    )
    # return
    functions = cache.get_workflow_function_configs(workflow_name)
    for func in functions:
        online_predictor.start_monitor(func.function_name)
    if workflow_running_plan_df is None:
        log.info("no need to update function running plan")
        return
    faas_operator.update_function(namespace, updated_functions_list)
    faas_operator.deploy_function(namespace, function_list)
    updater[workflow_name] = True


def add_event(obj):
    """
    Parses the workflow DAG from the object metadata and data, caches and deploys the functions
    to their respective devices as per the workflow running plan.

    Args:
        obj (object): Object containing metadata and data for the workflow.

    Returns:
        None
    """
    global workflow_obj
    workflow_running_plan_df = None
    global interval_time_unit
    if (
        obj.metadata.annotations is not None
        and "faas_workflow_name" in obj.metadata.annotations
        and "dag_parse_finished" not in obj.metadata.annotations
    ):
        workflow_name = obj.metadata.annotations["faas_workflow_name"]
        optimizer_name = obj.metadata.annotations["optimizer"]
        global optimizer
        optimizer = optimizer_factory.get_optimizer(
            optimizer_name,
            default_device,
            function_profiler,
            cache,
            invocation_infos,
            online_predictor,
        )
        global test_type
        test_type = obj.metadata.annotations["testtype"]

        log.info(f"start to parse workflow {workflow_name}")
        workflow_dag = {}
        workflow_dag["SLA"] = int(base64_decode(obj.data["SLA"]))
        workflow_dag["node"] = eval(base64_decode(obj.data["node"]))
        workflow_dag["edges"] = eval(base64_decode(obj.data["edges"]))
        interval_time_unit = eval(base64_decode(obj.data["interval_time_unit"]))
        function_info_data = eval(base64_decode(obj.data["function_info"]))
        workflow_obj[workflow_name] = (obj, interval_time_unit)
        function_info = get_function_info(function_info_data, function_profiler)
        dag_parser.set_function_info(function_info)
        dag_parser.generate_workflow_dag(workflow_name, workflow_dag)
        graph_df, graph_dfs, entry_node = dag_parser.parse_graph_to_df(workflow_name)
        SLA = dag_parser.get_workflow_SLA(workflow_name)
        function_list = []
        log.info(f"optimizer: {optimizer.__class__.__name__}")
        log.info(f"SLA is {SLA}")
        (
            workflow_running_plan_df,
            function_list,
        ) = optimizer.get_workflow_running_plan_df(
            workflow_name, graph_df, graph_dfs, 0, SLA, interval_time_unit
        )
        update_workflow_functions(
            workflow_name, workflow_running_plan_df, function_list, SLA
        )
        scaler_pool.submit(start_invocation_number_updater, workflow_name)
        faas_operator.update_workflow_cm(namespace, obj)


def del_event(obj):
    """
    Handle an event and delete its associated workflow functions and DAG.

    Args:
        obj: Kubernetes event object to delete.

    Returns:
        None
    """

    if (
        obj.metadata.annotations is not None
        and "faas_workflow_name" in obj.metadata.annotations
    ):
        global optimizer
        workflow_name = obj.metadata.annotations["faas_workflow_name"]
        workflow_functions = dag_parser.get_workflow_functions(workflow_name)
        faas_operator.delete_function(namespace, workflow_functions, workflow_name)
        dag_parser.delete_workflow(workflow_name)
        for func in workflow_functions:
            online_predictor.stop_monitor(func)
        global workflow_obj
        del workflow_obj[workflow_name]
        updater[workflow_name] = False
        optimizer.remove_workflow(workflow_name)
        online_predictor.reset_all()


def start_invocation_number_updater(workflow_name):
    log.info(f"start to update workflow {workflow_name}")

    while True:
        if workflow_name not in updater:
            break
        if not updater[workflow_name]:
            break
        entry_node = cache.get_workflow_entries(workflow_name)
        IT = online_predictor.predict_inter_arrival_time(
            entry_node, interval_time_unit, optimizer.__class__.__name__
        )
        SLA = cache.get_workflow_sla(workflow_name)
        if IT == 0:
            workflow_running_df, function_list = (
                optimizer.update_workflow_running_plan_df(
                    workflow_name, IT, SLA, interval_time_unit
                )
            )
            update_workflow_functions(workflow_name, workflow_running_df, function_list)
        invocation_number = online_predictor.predict_invocation_number(
            entry_node, IT, optimizer.__class__.__name__, test_type
        )
        functions = cache.get_workflow_function_configs(workflow_name)
        for function in functions:
            if function.device["merge_device"] == "cpu":
                inference_time = function.cpu_running_time
            else:
                inference_time = function.gpu_running_time
            if (
                optimizer.__class__.__name__ == "SMIless"
                or "SMIlessOPT"
                or "SMIlessHomo"
                or "SMIlessNoDag"
            ):
                replicas, spec = auto_scaler.scale_function(
                    invocation_number,
                    function.function_name,
                    function.device["merge_device"],
                    inference_time,
                )
                log.info(
                    f"function {function.function_name} will be invoked{invocation_number} scale to {replicas}"
                )
            else:
                replicas, spec = auto_scaler.default_auto_scaler(
                    invocation_number, function.function_name, function.device
                )
            function.target_replicas = replicas
            if spec != -1:
                function.device["merge_keep_alive_resource"] = spec
            cache.update_func_config(function.function_name, function)
            faas_operator.update_function_cm(namespace, function)
            time.sleep(1.0)


def start_workflow_event_handler():
    while True:
        log.info("begin to handle")
        for event in w.stream(v1.list_namespaced_config_map, namespace):
            if event["type"] == "ADDED" or event["type"] == "MODIFIED":
                i = event["object"]
                add_event(i)
            if event["type"] == "DELETED":
                i = event["object"]
                del_event(i)
        time.sleep(1)


if __name__ == "__main__":
    try:
        kconf.load_incluster_config()
    except Exception as e:
        log.info(f"Failed to initialize the incluster config, {e}")
        kconf.load_kube_config()
    gateway = os.environ.get("gateway", "192.168.0.102")
    port = os.environ.get("port", "31112")
    passwd = os.environ.get("passwd", "sWsVc8uuyJPe")
    default_device = os.environ.get("default_device", "cpu")
    prom_operator = PrometheusOperator(f"http://{gateway}:30091")
    v1 = client.CoreV1Api()
    namespace = "openfaas-fn"
    kube_operator = KubeOperator(v1, namespace)
    invocation_infos = InvocationInfos()
    online_predictor = OnlinePredictor(prom_operator, invocation_infos)
    function_profiler = FunctionProfiler(kube_operator, online_predictor)
    cache = Cache(function_profiler)
    auto_scaler = AutoScaler(function_profiler)
    optimizer_factory = OptimizerFactory()
    global optimizer, test_type
    optimizer, test_type = None, None
    dag_parser = DAGParser()
    faas_operator = FaasOperator(gateway, port, passwd, cache)
    faas_operator.get_faas_template(namespace, "faas-template-cm")
    w = watch.Watch()
    global workflow_obj, updater
    workflow_obj, updater = {}, {}
    scaler_pool = ThreadPoolExecutor(max_workers=32)
    with ThreadPoolExecutor(max_workers=3) as executor:
        task = executor.submit(start_workflow_event_handler)
        task.result()
