import numpy as np
from .function_profiler import FunctionProfiler

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


class AutoScaler:
    def __init__(self, ):
        function_profiler = FunctionProfiler()
        self.cpu_function_profilers = function_profiler.cpu_running_time_profilers
        self.gpu_function_profilers = function_profiler.gpu_running_time_profilers
        function_names = function_profiler.function_names
        self.image_sizes = function_profiler.image_sizes
        cpu_running_time = {}
        gpu_running_time = {}

        for func_name in function_names:
            new_func_name = function_names.index(func_name)
            self.max_new_func_name = new_func_name
            cpu_running_time[new_func_name] = (
                function_profiler.get_cpu_cost_running_time(func_name,1,1,self.image_sizes[func_name]['image_size'])
            )
            gpu_running_time[new_func_name] = function_profiler.get_gpu_execution_time(
                func_name
            )
            self.cpu_function_profilers[new_func_name] = self.cpu_function_profilers[
                func_name
            ]
            self.gpu_function_profilers[new_func_name] = self.gpu_function_profilers[
                func_name
            ]

        self.cpu_inference_time = cpu_running_time
        self.gpu_inference_time = gpu_running_time

    def calc_specification(self, B, function_name, inference_time, device):
        if device == "cpu":
            lambdax = self.cpu_function_profilers[function_name][0]
            alpha = self.cpu_function_profilers[function_name][1]
            beta = self.cpu_function_profilers[function_name][2]
            gamma = self.cpu_function_profilers[function_name][3]
            if lambdax * B * beta != inference_time - gamma:
                spec = np.ceil(
                    (lambdax * B * alpha)
                    / (inference_time - gamma - lambdax * B * beta)
                )
            else:
                spec = -1
        elif device == "gpu":
            lambdax = self.gpu_function_profilers[function_name][0]
            alpha = self.gpu_function_profilers[function_name][1]
            beta = self.gpu_function_profilers[function_name][2]
            gamma = self.gpu_function_profilers[function_name][3]
            if lambdax * B * beta != inference_time - gamma:
                spec = np.ceil(
                    (lambdax * B * alpha)
                    / (inference_time - gamma - lambdax * B * beta)
                )
            else:
                spec = -1
        return spec

    def calc_cost_unit(self, spec, device):
        if device == "cpu":
            for k in CPU_COST:
                if spec <= k:
                    return CPU_COST[k] * spec
            return CPU_COST[16] * spec
        if device == "gpu":
            return GPU_COST[1] * spec

    def check_constraint(self, G, function_name, device, inference_time):
        if device == "cpu":
            predicted_latency = self.cpu_running_time_func(
                [G, 1], *self.cpu_function_profilers[function_name]
            )
            if predicted_latency > inference_time:
                return False
        elif device == "gpu":
            predicted_latency = self.cpu_running_time_func(
                [G, 1], *self.gpu_function_profilers[function_name]
            )
            if predicted_latency > inference_time:
                return False
        return True

    def cpu_running_time_func(self, X, a, b, c, d):
        x, y = X[0], X[1]
        return a * x * (b / y + c) + d

    def search_result(self, B, G, function_name, inference_time, device):
        instance_number = np.ceil(G / B)
        spec = self.calc_specification(B, function_name, inference_time, device)
        if spec <= 0:
            return np.inf
        cost = self.calc_cost_unit(spec, device)
        # print(instance_number, spec, cost,inference_time, self.IT, device)
        return instance_number * cost * inference_time + cost

    def get_new_function_name(self, funciton_name):
        funciton_name_bytes = funciton_name.encode()
        integer = int.from_bytes(funciton_name_bytes, byteorder="big")
        new_func_name = integer % self.max_new_func_name
        return new_func_name

    def get_container_number(self, G, function_name, device):
        function_name = self.get_new_function_name(function_name)
        if device == "cpu":
            inference_time = self.cpu_inference_time[function_name]
        else:
            inference_time = self.gpu_inference_time[function_name]
        upper_bound = G
        lower_bound = 1
        if not self.check_constraint(
            upper_bound, function_name, device, inference_time
        ):
            overall_cost = (
                self.calc_cost_unit(
                    self.calc_specification(1, function_name, inference_time, device),
                    device,
                )
                * G
            )
        else:
            if G == 0:
                return 0
            else:
                return 1
        B = lower_bound + upper_bound
        while lower_bound <= upper_bound:
            B = max((lower_bound + upper_bound) // 2, 1)
            cost = self.search_result(B, G, function_name, inference_time, device)
            if cost < overall_cost:
                upper_bound = B - 1
            else:
                lower_bound = B + 1

        return np.ceil(G / B)
