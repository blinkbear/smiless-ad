# This module is used to create the profiler between resource allocation and function execution time.

# We only use simple prediction method.
import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_NUMBER = -1e9
CPU_COST = {
    # aws Ohio region
    1: 0.034,  # c6g.medium 1c2G10Gbps
    2: 0.068,  # c6g.large 2c4G10Gbps
    4: 0.136,  # c6g.xlarge 4c8G10Gbps
    8: 0.272,  # c6g.2xlarge 8c16G10Gbps
    16: 0.544,  # c6g.4xlarge 16c32G10Gbpsd
}
GPU_COST = {
    1: 0.306  # p3.2xlarge 8c61G10Gbps+V100
}


class FunctionProfiler:
    def __init__(self):
        self.image_sizes = self.__get_image_sizes()
        self.cold_start_time = self.__get_cold_start_time()
        self.cpu_running_time_profilers, self.gpu_running_time_profilers = (
            self.__generate_exec_time_profiler()
        )

    def __get_image_sizes(self):
        with open(
            os.path.join(SCRIPT_DIR, "function_profilers/image_sizes.json"), "r"
        ) as f:
            image_sizes = json.load(f)
        image_sizes = image_sizes["image_sizes"]
        return image_sizes
    def __get_cold_start_time(self):
        with open(
            os.path.join(SCRIPT_DIR, "function_profilers/initialization_time.json"), "r"
        ) as f:
            initialization_time = json.load(f)
        faas_initialization_time = json.loads(
            initialization_time["initialization_time"]
        )
        coefficient = json.loads(initialization_time["coefficient"])
        function_cold_start_time = {}
        self.function_names = list(faas_initialization_time.keys())

        for function_name in self.function_names:
            cpu_mu = faas_initialization_time[function_name]["cpu_mu"]
            cpu_sigma = faas_initialization_time[function_name]["cpu_sigma"]
            gpu_mu = faas_initialization_time[function_name]["gpu_mu"]
            gpu_sigma = faas_initialization_time[function_name]["gpu_sigma"]

            cpu_cold_start_time = cpu_mu + coefficient * cpu_sigma
            gpu_cold_start_time = gpu_mu + coefficient * gpu_sigma
            function_cold_start_time[function_name] = (
                cpu_cold_start_time,
                gpu_cold_start_time,
            )
        return function_cold_start_time

    def function_running_time_func(self, X, a, b, c, d):
        qps, resource_amount = X[0], X[1]
        return a * qps * (b / resource_amount + c) + d

    def __generate_exec_time_profiler(self):
        with open(
            os.path.join(SCRIPT_DIR, "function_profilers/execution_time_profiler.json"),
            "r",
        ) as f:
            function_exec_time_profilers = json.load(f)
        running_time_profilers = json.loads(function_exec_time_profilers["running_time_profiler"])
        cpu_running_time_profilers = {}
        gpu_running_time_profilers = {}
        for func_name in running_time_profilers:
            cpu_running_time_profilers[func_name] = running_time_profilers[func_name][
                "cpu"
            ]
            gpu_running_time_profilers[func_name] = running_time_profilers[func_name][
                "gpu"
            ]
        return cpu_running_time_profilers, gpu_running_time_profilers

    def get_running_time_profilers(self):
        return self.cpu_running_time_profilers

    def get_gpu_execution_time(self, function_name):
        opt = self.gpu_running_time_profilers[function_name]
        gpu_execution_time = self.function_running_time_func((1, 10), *opt)
        return gpu_execution_time


    def resource_cost(self, cpu_resource):
        if cpu_resource in CPU_COST:
            cpu_cost_unit = CPU_COST[cpu_resource]
        else:
            cpu_cost_unit = CPU_COST[1] * cpu_resource

        return {"cuda": GPU_COST[1], "cpu": cpu_cost_unit, "memory": cpu_cost_unit}

    def get_cpu_cost_running_time(
        self, types: str, qps: int, cpu_resource: int, image_size: float
    ):
        opt = self.cpu_running_time_profilers[types]
        predicted_execution_time = self.function_running_time_func(
            (qps, cpu_resource), opt[0], opt[1], opt[2], opt[3]
        )
        return (
            predicted_execution_time
        )
