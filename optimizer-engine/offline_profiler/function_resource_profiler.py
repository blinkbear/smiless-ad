# This module is used to create the profiler between resource allocation and function execution time.

# We only use simple prediction method.
import pandas as pd
import os
import numpy as np
import json
from offline_profiler.util import Util
from ..utils.kube_operator import KubeOperator
from ..online_predictor.online_predictor import OnlinePredictor
from ..utils.prom_operator import PrometheusOperator
import time
from scipy.optimize import curve_fit
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def function_running_time_func(X, a, b, c, d):
    qps, resource_amount = X[0], X[1]
    return a * qps * (b / resource_amount + c) + d


class FunctionProfiler:
    def __init__(
        self,
        kube_op: KubeOperator,
        online_predictor: OnlinePredictor,
        prom_op: PrometheusOperator,
    ):
        self.kube_op = kube_op
        self.online_predictor = online_predictor

        self.function_info_updator = FunctionInfoUpdator(prom_op)
        self.cpu_running_time_profilers, self.gpu_running_time_profilers = (
            self.function_info_updator.get_execution_profilers()
        )
        (
            self.cold_start_time,
            self.cpu_cs_extra_time_profilers,
            self.gpu_cs_extra_time_profilers,
        ) = self.__get_cold_start_time()

        self.function_names = self.__get_function_names()
        self.available_cpu_resource = np.array([1, 2, 4, 8, 16])

    def __get_cold_start_time(self):
        faas_initialization_configmap = self.kube_op.get_configmap(
            "faas-initialization-cm"
        )
        if faas_initialization_configmap is None:
            with open("function_profilers/initialization_time.json", "r") as f:
                faas_initialization_configmap = json.load(f)
        faas_initialization_time = json.loads(
            faas_initialization_configmap["data"]["initialization_time"]
        )
        coefficient = json.loads(faas_initialization_configmap["data"]["coefficient"])
        function_cold_start_time = {}
        function_cuda_cs_extra_time = {}
        function_cpu_cs_extra_time = {}
        self.function_names = list(coefficient.keys())
        faas_initalization_cm = self.kube_op.get_faas_initalization_cm(
            "faas-initialization-cm"
        )
        cs_time = json.loads(faas_initalization_cm["data"]["cs_time"])

        for function_name in self.function_names:
            cpu_mu = faas_initialization_time[function_name]["cpu-mu"]
            cpu_sigma = faas_initialization_time[function_name]["cpu-sigma"]
            gpu_mu = faas_initialization_time[function_name]["gpu-mu"]
            gpu_sigma = faas_initialization_time[function_name]["gpu-sigma"]

            cpu_cold_start_time = cpu_mu + coefficient * cpu_sigma
            gpu_cold_start_time = gpu_mu + coefficient * gpu_sigma
            function_cold_start_time[function_name] = (
                cpu_cold_start_time,
                gpu_cold_start_time,
            )
            function_cuda_cs_extra_time[function_name] = cs_time[function_name]["gpu"]
            function_cpu_cs_extra_time[function_name] = cs_time[function_name]["cpu"]
        return (
            function_cold_start_time,
            function_cpu_cs_extra_time,
            function_cuda_cs_extra_time,
        )

    def update_cold_start_time(self):
        self.cold_start_time = self.__get_cold_start_time()

    def get_resource_by_latency(self, function_name, resource_type, target_latency):
        function_name = function_name.split("-")[0]
        if resource_type == "cpu":
            available_resource = [1, 2, 4, 8, 16]
            opt = self.cpu_running_time_profilers[function_name]
            for i in range(len(available_resource)):
                estimated_execution_time = function_running_time_func(
                    (1, available_resource[i]), opt[0], opt[1], opt[2], opt[3]
                )
                if estimated_execution_time < target_latency:
                    return available_resource[i]

    def get_running_time_profilers(self):
        return self.cpu_running_time_profilers

    def get_idle_time_distribution(self, function_name):
        all_invocation_numbers = {}
        invocation_number = self.online_predictor.get_invocation_number(function_name)
        all_invocation_numbers[function_name] = invocation_number
        df = pd.DataFrame.from_dict(all_invocation_numbers, orient="index").T

        def get_interval(row):
            import itertools
            import pandas as pd

            function_name = row.to_list()[0]
            row = row.to_list()[1:]
            count_row = [len(list(v)) for k, v in itertools.groupby(row) if k == 0]
            from collections import Counter

            result = Counter(count_row)
            result = pd.DataFrame.from_dict(result, orient="index").reset_index()
            result.columns = ["interval", "count"]
            result["function_name"] = function_name
            result["probability"] = result["count"] / result["count"].sum()
            return result

        def fill_df(df, min_interval, max_interval):
            # fill the df from min_interval to max_interval with 1 interval
            intervals = df["interval"].values
            for i in range(min_interval, max_interval + 1):
                if i not in intervals:
                    df = df.append(
                        {
                            "function_name": df["function_name"][0],
                            "interval": i,
                            "count": 0,
                            "probability": 0,
                        },
                        ignore_index=True,
                    )
            return df

        dfs = df.apply(get_interval, axis=1)  # type: ignore
        min_interval, max_interval = 0, 0
        for df in dfs:
            if df["interval"].min() < min_interval:
                min_interval = df["interval"].min()
            if df["interval"].max() > max_interval:
                max_interval = df["interval"].max()
        fill_dfs = []
        for df in dfs:
            fill_dfs.append(fill_df(df, min_interval, max_interval))
        df = pd.concat(fill_dfs)
        df = df.sort_values(by=["function_name", "interval"])
        df = df[["function_name", "interval", "probability"]]
        return df

    def get_idle_time_probability(self, function_name, idle_time):
        idle_time_distribution = self.get_idle_time_distribution(function_name)
        idle_time_probability = idle_time_distribution[
            idle_time_distribution["interval"] <= idle_time
        ]["probability"].sum()

        return idle_time_probability

    def get_resource_profile(self, function_name):
        """ """
        return self.function_cpu_exec_time_profiler[function_name]  # type: ignore

    def get_gpu_execution_time(self, function_name):
        opt = self.gpu_running_time_profilers[function_name]
        gpu_execution_time = self.function_running_time_func((1, 10), *opt)
        # gpu_execution_time = self.gpu_running_time_profilers[function_name]
        return gpu_execution_time

    def get_gpu_cs_extra_time(self, function_name):
        gpu_cs_extra_time = self.gpu_cs_extra_time_profilers[function_name]
        return gpu_cs_extra_time

    def get_cpu_cs_extra_time(self, function_name):
        cpu_cs_extra_time = self.cpu_cs_extra_time_profilers[function_name]
        return cpu_cs_extra_time

    def get_cold_start_time(self, function_name):
        cold_start_time = self.cold_start_time[function_name][0]
        gpu_trans_time = self.cold_start_time[function_name][1] - cold_start_time
        return (
            cold_start_time,
            abs(gpu_trans_time),
        )

    def get_cpu_cost_running_time(
        self, types: str, qps: int, cpu_resource: int, image_size: float
    ):
        opt = self.cpu_running_time_profilers[types]
        predicted_execution_time = self.function_running_time_func(
            (qps, cpu_resource), opt[0], opt[1], opt[2], opt[3]
        )
        cost = Util.resource_cost(cpu_resource)
        return (
            predicted_execution_time,
            cost["cpu"],
            cost["cuda"],
            cost["memory"] * cpu_resource,
            cost["memory"] * 4,
        )

    def calc_keep_alive_time(self, function_name):
        idle_time = 0
        for idle_time in range(0, 1440):
            invoke_probability = self.get_idle_time_probability(
                function_name, idle_time
            )
            if invoke_probability >= 0.99:
                break
        return idle_time

    def get_shared_function_resource_with_qps(self, func_name, qps, backend):
        resources, backend = self.get_shared_function_resource(qps, func_name, backend)
        return backend, resources

    def get_shared_function_resource(self, function_name, qps, backend):
        opt_c = self.cpu_running_time_profilers[function_name]
        inference_time = function_running_time_func(
            (qps, 1), opt_c[0], opt_c[1], opt_c[2], opt_c[3]
        )
        if backend == "cpu":
            lambdax = self.cpu_function_profilers[function_name][0]
            alpha = self.cpu_function_profilers[function_name][1]
            beta = self.cpu_function_profilers[function_name][2]
            gamma = self.cpu_function_profilers[function_name][3]
            if lambdax * qps * beta != inference_time - gamma:
                spec = (
                    np.ceil(
                        (lambdax * qps * alpha)
                        / (inference_time - gamma - lambdax * qps * beta)
                    ),
                    "cpu",
                )
            else:
                spec = (10, "cuda")
        elif backend == "gpu":
            lambdax = self.gpu_function_profilers[function_name][0]
            alpha = self.gpu_function_profilers[function_name][1]
            beta = self.gpu_function_profilers[function_name][2]
            gamma = self.gpu_function_profilers[function_name][3]
            if lambdax * qps * beta != inference_time - gamma:
                spec = (
                    np.ceil(
                        (lambdax * qps * alpha)
                        / (inference_time - gamma - lambdax * qps * beta)
                    ),
                    "cuda",
                )
            else:
                spec = (100, "cuda")
        return spec


class FunctionInfoUpdator:
    def __init__(
        self,
        prometheus_operater: PrometheusOperator,
    ):
        self.prometheus_operator = prometheus_operater
        self.set_profilers()
        with ThreadPoolExecutor(max_workers=32) as executor:
            executor.submit(self.update_function_infos)

    def set_profilers(self):
        faas_function_runtime_profiler = self.kube_op.get_configmap(
            "faas-function-runtime-profiler"
        )
        if faas_function_runtime_profiler is None:
            with open(
                os.path.join(
                    SCRIPT_DIR, "function_profilers/execution_time_profiler.json"
                ),
                "r",
            ) as f:
                function_exec_time_profilers = json.load(f)
        self.profliers = eval(
            function_exec_time_profilers["data"]["function_running_time_profilers"]
        )

    def update_function_infos(self):
        while True:
            et = time.time()
            st = et - 100
            self.prometheus_operator.get_function_name_execution_time_info(st=st, et=et)

    def get_execution_profilers(self):
        with Lock():
            cpu_execution_time_profilers, gpu_execution_time_profilers = (
                self.profilers["cpu"],
                self.profilers["gpu"],
            )
        return cpu_execution_time_profilers, gpu_execution_time_profilers

    def update_function_execution_time(self, function_infos):
        old_profilers = self.profilers
        function_running_time_profilers = {}
        function_names = list(function_infos["function_name"].unique())
        for function_name in function_names:
            cpu_running_time_profiler = old_profilers[function_name]["cpu"]
            gpu_running_time_profiler = old_profilers[function_name]["gpu"]
            function_cpu_running_status = function_infos.query(
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
            if len(y) > 4:
                popt, _ = curve_fit(
                    function_running_time_func,
                    X,
                    y,
                )
                cpu_running_time_profiler = tuple(
                    np.average(np.array([popt, cpu_running_time_profiler]), axis=0)
                )
            function_cpu_running_status = function_infos.query(
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
            if len(y) > 4:
                popt, _ = curve_fit(
                    function_running_time_func,
                    X,
                    y,
                )

                gpu_running_time_profiler = tuple(
                    np.average(np.array([popt, gpu_running_time_profiler]), axis=0)
                )
            function_running_time_profilers[function_name] = {
                "cpu": cpu_running_time_profiler,
                "gpu": gpu_running_time_profiler,
            }
        with Lock():
            self.profilers = function_running_time_profilers
