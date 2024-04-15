import datetime
from utils.configurate import InferenceConfigurator
from invoker.invoke import INVOKER_FACTORY
import json
from concurrent.futures import ThreadPoolExecutor
import os
import pandas as pd
import numpy as np
from prometheus_api_client import PrometheusConnect
from evaluation.improvement_source.offline_profiling.config_parser import ConfigParser
from utils.kube_operator import KubeOperator
from scipy.signal import fftconvolve
from scipy.optimize import curve_fit
import time
from sklearn.metrics import mean_absolute_percentage_error

SOURCEDIR = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.dirname(os.path.dirname(SOURCEDIR))
CONFIG_FILE = "config/config.json"
RESULT_DIR = "data/total_result"
FUNCTION_CONFIG_DIR = "config/"
TMP_DIR = "tmp"


# update configuration of the function, waiting for the function to be ready
def update_config(args):
    (
        func_name,
        cpu,
        memory,
        gpu,
        backend,
        gateway,
        passwd,
        node_selector,
        namespace,
        max_batch_size,
    ) = args
    configurator.configure(
        func_name,
        cpu,
        memory,
        gpu,
        backend,
        node_selector,
        namespace,
        max_batch_size,
    )
    print(f"invoke function {func_name}")
    # use faas-cli to update the function configuration
    configuration_file_path = os.path.join(
        SOURCEDIR, FUNCTION_CONFIG_DIR, f"{func_name}.yml"
    )
    update_cmd = f"bash {SOURCEDIR}/bash_op/update.sh {gateway} {passwd} {configuration_file_path} {func_name}"

    os.system(update_cmd)
    kubeOp.check_pod_ready(func_name)
    configurator.delete_tmp_config_file(func_name, namespace)


def remove_func(args):
    func = args[0]
    gateway = args[1]
    passwd = args[2]
    namespace = args[3]
    configuration_file_path = os.path.join(
        SOURCEDIR, FUNCTION_CONFIG_DIR, namespace, "inference.yml"
    )
    remove_cmd = f"bash {SOURCEDIR}/bash_op/remove.sh {gateway} {passwd} {configuration_file_path} {func}"
    os.system(remove_cmd)
    kubeOp.check_pod_terminated(func)


def invocation_args_generator(
    backends,
    namespace,
    server,
    loc,
    total_length,
    skip_functions,
    test_types,
    repeat,
    rerun,
):
    """
    Generate invocation args according to config file
    """
    function_names = config.get_function_profiles()
    alternative_params = config.get_alternative_resource_quantity()
    alternative_batch_size = config.get_alternative_batch_size()
    alternative_warm_or_cold = config.get_alternative_warm_or_cold()
    (
        default_cpu,
        default_mem,
        default_gpu,
        default_input_size,
        default_warm_cold,
        default_batch_size,
        default_invoking_type,
        default_max_batch_size,
    ) = config.get_default_params()
    args = []
    for i in range(len(function_names)):
        if i % total_length != loc:
            continue
        function_name = function_names[i]
        if len(rerun) == 0 and function_name in skip_functions:
            continue
        elif len(rerun) > 0 and function_name not in rerun:
            continue
        for backend in backends:
            for test_type in test_types:
                if test_type == "resource_quantity":
                    repeat = 1
                    if backend == "cuda":
                        for gpu in alternative_params[function_name]["cuda"]:
                            for batch_size in alternative_batch_size[function_name]:
                                args.append(
                                    (
                                        url,
                                        function_name,
                                        default_input_size,
                                        default_cpu,
                                        default_mem,
                                        gpu,
                                        backend,
                                        server,
                                        default_warm_cold,
                                        repeat,
                                        namespace,
                                        batch_size,
                                        default_invoking_type,
                                        default_max_batch_size,
                                    )
                                )
                    else:
                        for cpu in alternative_params[function_name]["cpu"]:
                            for batch_size in alternative_batch_size[function_name]:
                                args.append(
                                    (
                                        url,
                                        function_name,
                                        default_input_size,
                                        cpu,
                                        default_mem,
                                        default_gpu,
                                        backend,
                                        server,
                                        default_warm_cold,
                                        repeat,
                                        namespace,
                                        batch_size,
                                        default_invoking_type,
                                        default_max_batch_size,
                                    )
                                )
                elif test_type == "warm_or_cold":
                    for warm_or_cold in alternative_warm_or_cold[function_name]:
                        args.append(
                            (
                                url,
                                function_name,
                                default_input_size,
                                default_cpu,
                                default_mem,
                                default_gpu,
                                backend,
                                server,
                                warm_or_cold,
                                repeat,
                                namespace,
                                default_batch_size,
                                default_invoking_type,
                                default_max_batch_size,
                            )
                        )
    return args


def prepare_function_profile(
    function_name,
    cpu,
    memory,
    gpu,
    backend,
    server,
    namespace,
    max_batch_size,
):
    """
    prepare to profile functions. This method  1. remove all old functions 2. update the function configuration
    """

    remove_func([function_name, gateway, passwd, namespace])

    time.sleep(3)
    update_config(
        [
            function_name,
            cpu,
            memory,
            gpu,
            backend,
            gateway,
            passwd,
            server,
            namespace,
            max_batch_size,
        ]
    )


def invoke_function(
    url,
    function_name,
    input_size,
    cpu,
    memory,
    gpu,
    backend,
    server,
    warm_cold,
    repeat,
    namespace,
    batch_size,
    invoking_type,
    max_batch_size,
):
    """
    invoke function, if there are multiple functions in the chain, invoke them one by one
    Args:
        args: the arguments of the function, including:
        url: the invoke url of the function
        function_name: the function name
        input_size: the input size of the function
        cpu: the cpu limit of the function
        memory: the memory limit of the function
        gpu: the gpu percentage of the function
        backend: the backend of the function, cpu or gpu
        use_model_controller: whether to use model controller, if true, the model of function will be loaded by model controller, otherwise, the model will be loaded by the function itself
        server: the server where the function will be deployed
    """
    prepare_function_profile(
        function_name,
        cpu,
        memory,
        gpu,
        backend,
        server,
        namespace,
        invoking_type,
        max_batch_size,
    )
    if warm_cold == "cold":
        kubeOp.change_deployment(function_name, 0)
        kubeOp.check_pod_terminated(function_name)
    else:
        kubeOp.change_deployment(function_name, 1)
        kubeOp.check_pod_ready(function_name)
    invoker = INVOKER_FACTORY[function_name]()
    result = {
        "function_name": [],
        "start_time": [],
        "function_start_time": [],
        "query_time": [],
        "running_time": [],
        "model_load_time": [],
        "model_trans_time": [],
        "cpu": [],
        "memory": [],
        "cuda": [],
        "backend": [],
        "server": [],
        "warm_cold": [],
        "batch_size": [],
        "round": [],
    }
    header = {"Content-Type": "text/plain"}
    for i in range(repeat):
        if warm_cold == "cold":
            import time

            kubeOp.change_deployment(function_name, 0)
            kubeOp.check_pod_terminated(function_name)
            time.sleep(1)
        query_begin = datetime.datetime.now()

        # if invoking_type == "concurrency":
        pool = ThreadPoolExecutor(batch_size)
        tmp_query_time = []
        tmp_function_start_time = []
        tmp_running_time = []
        tmp_model_load_time = []
        tmp_model_trans_time = []

        for res in pool.map(
            invoker.invoke,
            [(url, namespace, 1, str(header), "function-profiling", 60)] * batch_size,
        ):
            tmp_query_time.append(res["query_time"])
            tmp_function_start_time.append(res["runtime"]["start_time"])
            if warm_cold == "cold":
                tmp_running_time.append(res["runtime"]["runtime"])
            elif warm_cold == "warm" and i != 0:
                tmp_running_time.append(res["runtime"]["runtime"])
            elif warm_cold == "warm" and i == 0:
                tmp_running_time.append(res["runtime"]["runtime"])
            tmp_model_load_time.append(res["runtime"]["model_load_time"])
            tmp_model_trans_time.append(res["runtime"]["model_trans_time"])
        result["query_time"].append(np.mean(tmp_query_time))
        result["function_start_time"].append(np.mean(tmp_function_start_time))
        result["running_time"].append(np.mean(tmp_running_time))
        result["model_load_time"].append(np.mean(tmp_model_load_time))
        result["model_trans_time"].append(np.mean(tmp_model_trans_time))
        result["function_name"].append(function_name)
        result["start_time"].append(int(query_begin.timestamp()))
        result["round"].append(i)
        result["input_size"] = input_size
        result["cpu"] = cpu
        result["memory"] = memory
        result["cuda"] = gpu
        result["backend"] = backend
        result["server"] = server
        result["batch_size"] = batch_size
        result["warm_cold"] = warm_cold
        result["queue_size"] = batch_size
        result["invoking_type"] = invoking_type
        for col in result.columns:
            if "time" in col:
                result.loc[result.index, col] = result[col] / 1000000
    remove_func([function_name, gateway, passwd, namespace])
    return result


def start_invoke(
    backends,
    namespace,
    server,
    loc,
    total_length,
    skip_functions,
    test_types,
    rerun,
):
    """
    start invoke function. The method will first parse the config file to get the function chains that start with the function_name (the function chains are definded in config.functions_chain). Then it will invoke the function chains parallely (the concurrency is defined in config.concurrency). Total invocation will repeat 10 times for each function chain. The result will be saved in df.
    Args:
        function_name: the function name
        df: the dataframe to save the result
        input_size: the input size of the function, default is 10
        cpu: the cpu limit of the function, default is 2
        memory: the memory limit of the function, default is 4096
        gpu: the gpu percentage of the function, default is 20
    Return:
        df: the dataframe to save the result
    """
    # concurrency = len(servers)
    repeat = config.get_repeat()
    invocation_args = invocation_args_generator(
        backends,
        namespace,
        server,
        loc,
        total_length,
        skip_functions,
        test_types,
        repeat,
        rerun,
    )
    if os.path.exists(os.path.join(SOURCEDIR, TMP_DIR, f"tmp{server}.csv")):
        df = pd.read_csv(os.path.join(SOURCEDIR, TMP_DIR, f"tmp{server}.csv"))
    else:
        df = pd.DataFrame()
    for invoke_args in invocation_args:
        result = invoke_function(*invoke_args)
        tmp_df = pd.DataFrame(result)
        df = pd.concat([df, tmp_df], ignore_index=True)
        df.to_csv(os.path.join(SOURCEDIR, TMP_DIR, f"tmp{server}.csv"), index=False)
    return df


def merge_result(key, servers):
    """
    merge the result from different servers
    Args:
        servers: the servers to merge the result
    Return:
        df: the dataframe to save the result
    """
    import shutil
    import datetime

    result_path = os.path.join(SOURCEDIR, RESULT_DIR, f"result_{key}.csv")
    if os.path.exists(result_path):
        shutil.copy(
            result_path,
            os.path.join(
                SOURCEDIR,
                RESULT_DIR,
                f"result_{key}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.csv",
            ),
        )
    df = pd.DataFrame()
    for server in servers:
        tmp_result_csv = os.path.join(SOURCEDIR, TMP_DIR, f"tmp{server}.csv")
        if os.path.exists(tmp_result_csv):
            tmp_df = pd.read_csv(tmp_result_csv)
        else:
            tmp_df = pd.DataFrame()
        df = pd.concat([df, tmp_df], ignore_index=True)
    df["test_type"] = key
    df.to_csv(result_path, index=False)
    for server in servers:
        tmp_result_csv = os.path.join(SOURCEDIR, TMP_DIR, f"tmp{server}.csv")
        if os.path.exists(tmp_result_csv):
            os.remove(tmp_result_csv)
    return df


def get_cs_extra_time_profiler(result):
    for col in result.columns:
        if "time" in col:
            result.loc[result.index, col] = result[col] / 1000000
    function_names = result["function_name"].unique()
    function_cs_extra_times = {}
    function_running_time_cuda_warm_cold = result.query(
        "test_type == 'warm_cold' and backend == 'cuda'"
    )
    function_running_time_cpu_warm_cold = result.query(
        "test_type == 'warm_cold' and backend == 'cpu'"
    )
    for function_name in function_names:
        function_cs_extra_times[function_name] = {"cpu": 0, "gpu": 0}
        gpu_cs_running_time = function_running_time_cuda_warm_cold.query(
            "function_name == @function_name and warm_cold == 'cold'"
        )["running_time"].max()
        gpu_warm_running_time = function_running_time_cuda_warm_cold.query(
            "function_name == @function_name and warm_cold == 'warm'"
        )["running_time"].max()
        function_cs_extra_times[function_name]["gpu"] = max(
            0, (gpu_cs_running_time - gpu_warm_running_time)
        )
        cpu_cs_time = function_running_time_cpu_warm_cold.query(
            "function_name == @function_name and warm_cold == 'cold'"
        )["running_time"]
        cpu_warm_time = function_running_time_cpu_warm_cold.query(
            "function_name == @function_name and warm_cold == 'warm'"
        )["running_time"]
        cpu_cs_running_time = cpu_cs_time.max()
        cpu_warm_running_time = cpu_warm_time.max()
        cpu_cs_extra_time = cpu_cs_running_time - (
            cpu_warm_running_time
            if cpu_cs_running_time > cpu_warm_running_time
            else cpu_cs_running_time
        )
        function_cs_extra_times[function_name]["cpu"] = cpu_cs_extra_time
    return function_cs_extra_times


def update_function_initialization_config(result):
    function_names = result["function_name"].unique()
    function_initializaiton_config = {}
    for function_name in function_names:
        function_running_status = result.query(
            "function_name == @function_name and test_type == 'warm_cold'"
        )

        function_cpu_cold_start = function_running_status.query(
            "backend == 'cpu' and warm_cold == 'cold'"
        )
        function_cpu_cold_start.columns = [
            col + "_cold" for col in function_cpu_cold_start.columns
        ]
        function_cpu_warm_start = function_running_status.query(
            "backend == 'cpu' and warm_cold == 'warm'"
        )
        function_cpu_warm_start.columns = [
            col + "_warm" for col in function_cpu_warm_start.columns
        ]
        function_gpu_cold_start = function_running_status.query(
            "backend == 'cuda' and warm_cold == 'cold'"
        )

        function_gpu_cold_start.columns = [
            col + "_cold" for col in function_gpu_cold_start.columns
        ]
        function_gpu_warm_start = function_running_status.query(
            "backend == 'cuda' and warm_cold == 'warm'"
        )
        function_gpu_warm_start.columns = [
            col + "_warm" for col in function_gpu_warm_start.columns
        ]

        function_cpu_cold_start.reset_index(drop=True, inplace=True)
        function_cpu_warm_start.reset_index(drop=True, inplace=True)
        function_gpu_cold_start.reset_index(drop=True, inplace=True)
        function_gpu_warm_start.reset_index(drop=True, inplace=True)
        function_cpu_start = pd.concat(
            [function_cpu_cold_start, function_cpu_warm_start], axis=1
        )
        function_gpu_start = pd.concat(
            [function_gpu_cold_start, function_gpu_warm_start], axis=1
        )
        function_cpu_start["cold_start_time"] = (
            function_cpu_start["query_time_cold"]
            - function_cpu_start["query_time_warm"]
        )
        function_gpu_start["cold_start_time"] = (
            function_gpu_start["query_time_cold"]
            - function_gpu_start["query_time_warm"]
        )
        cpu_mu = function_cpu_start["cold_start_time"].mean()
        gpu_mu = function_gpu_start["cold_start_time"].mean()
        cpu_sigma = function_cpu_start["cold_start_time"].std()
        gpu_sigma = function_gpu_start["cold_start_time"].std()
        function_initializaiton_config[function_name] = {
            "cpu_mu": cpu_mu,
            "gpu_mu": gpu_mu,
            "cpu_sigma": cpu_sigma,
            "gpu_sigma": gpu_sigma,
        }
    function_cs_extra_time = get_cs_extra_time_profiler(result)
    data = {
        "initialization_config": json.dumps(function_initializaiton_config),
        "cs_time": json.dumps(function_cs_extra_time),
        "coefficient": json.dumps(3),
    }
    kubeOp.create_or_update_configmap("faas-initialization-cm", data)


def update_function_coefficient(coefficient=3):
    current_function_initialization_config = kubeOp.get_configmap()
    data = {
        "initialization_config": current_function_initialization_config["data"][
            "initialization_config"
        ],
        "coefficient": json.dumps(coefficient),
    }
    kubeOp.create_or_update_configmap("faas-initialization-cm", data)


def function_running_time_func(X, a, b, c, d):
    qps, resource_amount = X[0], X[1]
    return a * qps * (b / resource_amount + c) + d


def update_exec_time_profiler(result):
    function_running_time_profilers = {}
    function_running_time_params = result.query(
        "test_type == 'resource_quantity' or test_type == 'batch_size'"
    )
    for function_name in list(result["function_name"].unique()):
        function_cpu_running_status = function_running_time_params.query(
            "function_name == @function_name and backend == 'cpu'"
        )
        function_cpu_running_status = (
            function_cpu_running_status[["query_time", "cpu", "batch_size"]]
            .groupby(["cpu", "batch_size"])
            .min()
            .reset_index()
        )
        X = (
            function_cpu_running_status["batch_size"],
            function_cpu_running_status["cpu"],
        )
        y = function_cpu_running_status["query_time"]
        popt, _ = curve_fit(
            function_running_time_func,
            X,
            y,
        )
        cpu_running_time_profiler = (
            popt[0],
            popt[1],
            popt[2],
            popt[3],
        )
        function_cpu_running_status = function_running_time_params.query(
            "function_name == @function_name and backend == 'cuda'"
        )
        function_cpu_running_status = (
            function_cpu_running_status[["query_time", "cuda", "batch_size"]]
            .groupby(["cuda", "batch_size"])
            .min()
            .reset_index()
        )
        X = (
            function_cpu_running_status["batch_size"],
            function_cpu_running_status["cuda"],
        )
        y = function_cpu_running_status["query_time"]
        popt, _ = curve_fit(
            function_running_time_func,
            X,
            y,
        )
        gpu_running_time_profiler = (
            popt[0],
            popt[1],
            popt[2],
            popt[3],
        )
        function_running_time_profilers[function_name] = {
            "cpu": cpu_running_time_profiler,
            "gpu": gpu_running_time_profiler,
        }
    data = {"running_time_profiler": json.dumps(function_running_time_profilers)}
    kubeOp.create_or_update_configmap("function-running-time-profiler-cm", data)
    evaluate_running_time_profiler(result, function_running_time_profilers)


def evaluate_running_time_profiler(result, function_running_time_profilers):
    mapes = {"cpu": [], "gpu": []}

    for function_name in list(result["function_name"].unique()):
        all_running_time = result[result["function_name"] == function_name][
            ["batch_size", "cpu", "cuda", "device", "running_time"]
        ]
        for index, row in all_running_time.iterrows():
            if row["device"] == "cpu":
                cpu_profiler = function_running_time_profilers[function_name]["cpu"]
                row["predicted_running_time"] = function_running_time_func(
                    (row["batch_size"], row["cpu"]), *cpu_profiler
                )
            elif row["device"] == "cuda":
                gpu_profiler = function_running_time_profilers[function_name]["gpu"]
                row["predicted_running_time"] = function_running_time_func(
                    (row["batch_size"], row["cuda"]), *gpu_profiler
                )
        cpu_running_time = all_running_time[all_running_time["device"] == "cpu"]
        gpu_running_time = all_running_time[all_running_time["device"] == "cuda"]
        mapes["cpu"].append(
            mean_absolute_percentage_error(
                cpu_running_time["running_time"],
                cpu_running_time["predicted_running_time"],
            )
        )
        mapes["gpu"].append(
            mean_absolute_percentage_error(
                gpu_running_time["running_time"],
                gpu_running_time["predicted_running_time"],
            )
        )
    df = pd.DataFrame({"cpu": mapes["cpu"], "gpu": mapes["gpu"]})
    df.to_csv(os.path.join(SOURCEDIR, "data", "inference_time.csv"), index=False)


def evaluate_initialization():
    import subprocess

    available_coefficients = [1, 2, 3]

    for coefficient in available_coefficients:
        update_function_coefficient(coefficient)
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            f"result_coefficient_{coefficient}",
        )
        os.makedirs(save_dir, exist_ok=True)
        exec_command = f"python3 {os.path.join(EVALUATION_DIR,'end_to_end','invoke_workflow.py')} --master_ip 127.0.0.1 --slas '2,' --optimizer 'smiless' --save_dir {save_dir} --test_type 'initialization'"
        result = subprocess.call(exec_command, shell=True)
        print(result)


if __name__ == "__main__":
    config_file = os.path.join(SOURCEDIR, CONFIG_FILE)
    config = ConfigParser(config_file)
    url, gateway, passwd = config.get_openfaas_config()

    configurator = InferenceConfigurator()
    test_config = config.get_test_config()
    active_test = config.get_test()
    result_df = pd.DataFrame()
    for (
        key,
        backend,
        namespace,
        test_type,
        servers,
        save_result,
        skip_functions,
        rerun,
    ) in test_config:
        if key not in active_test:
            continue
        print("start test for", key)
        global kubeOp
        kubeOp = KubeOperator(namespace, SOURCEDIR)
        pool = ThreadPoolExecutor(len(servers))
        locs = [i for i in range(len(servers))]
        for df in pool.map(
            start_invoke,
            [backend] * len(servers),
            [namespace] * len(servers),
            servers,
            locs,
            [len(servers)] * len(servers),
            [skip_functions] * len(servers),
            [test_type] * len(servers),
            [rerun] * len(servers),
        ):
            df["test_type"] = key
            result_df = pd.concat([result_df, df], ignore_index=True)
        merge_result(key, servers)
    update_function_initialization_config(result_df)
    update_exec_time_profiler(result_df)
    evaluate_initialization()
