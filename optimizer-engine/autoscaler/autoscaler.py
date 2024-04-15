import numpy as np
from offline_profiler.function_resource_profiler import FunctionProfiler

MIN_NUMBER = -1e9
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


class AutoScaler:
    def __init__(self, function_profilers: FunctionProfiler):
        self.function_cpu_running_time_profilers = (
            function_profilers.cpu_running_time_profilers
        )
        self.function_gpu_running_time_profilers = (
            function_profilers.gpu_running_time_profilers
        )

    def calc_specification(self, B, function_name, inference_time, device):
        spec = -1
        if device == "cpu":
            lambdax = self.function_cpu_running_time_profilers[function_name][0]
            alpha = self.function_cpu_running_time_profilers[function_name][1]
            beta = self.function_cpu_running_time_profilers[function_name][2]
            gamma = self.function_cpu_running_time_profilers[function_name][3]
            if lambdax * B * beta < inference_time - gamma:
                spec = np.ceil(
                    (lambdax * B * alpha)
                    / (inference_time - gamma - lambdax * B * beta)
                )
            else:
                spec = -1
        elif device == "cuda":
            lambdax = self.function_gpu_running_time_profilers[function_name][0]
            alpha = self.function_gpu_running_time_profilers[function_name][1]
            beta = self.function_gpu_running_time_profilers[function_name][2]
            gamma = self.function_gpu_running_time_profilers[function_name][3]
            if lambdax * B * beta < inference_time - gamma:
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
                    return CPU_COST[k]
        if device == "cuda":
            # for k in GPU_COST:
            #     if spec <= k:
            return GPU_COST[1] * spec

    def search_result(self, B, G, function_name, inference_time, device):
        instance_number = np.ceil(G / B)
        spec = self.calc_specification(B, function_name, inference_time, device)
        if spec < 0:
            return np.inf, -1
        cost = self.calc_cost_unit(spec, device)
        return instance_number * cost

    def cpu_running_time_func(self, X, a, b, c, d):
        x, y = X[0], X[1]
        return a * x * (b / y + c) + d

    def check_constraint(self, G, function_name, device, inference_time):
        if device == "cpu":
            predicted_latency = self.cpu_running_time_func(
                (G, 1), *self.function_cpu_running_time_profilers[function_name]
            )
            if predicted_latency > inference_time:
                return False
        elif device == "cuda":
            predicted_latency = self.cpu_running_time_func(
                (G, 1), *self.function_gpu_running_time_profilers[function_name]
            )
            if predicted_latency > inference_time:
                return False
        return True

    def scale_function(self, G, function_name, device, inference_time):
        """
        Scale the given graph `G` based on the `function_name`, `device`, and `inference_time`.

        Parameters:
            G (int): The graph to be scaled.
            function_name (str): The name of the function.
            device (str): The device to be used.
            inference_time (float): The time taken for inference.

        Returns:
            tuple: A tuple containing the instance number and the best specification.

        """
        upper_bound = G
        lower_bound = 1
        if not self.check_constraint(
            upper_bound, function_name, device, inference_time
        ):
            overall_cost = np.inf
        else:
            overall_cost = self.calc_cost_unit(
                self.calc_specification(G, function_name, inference_time, device),
                device,
            )
        best_spec = self.calc_specification(G, function_name, inference_time, device)
        while lower_bound <= upper_bound:
            B = (lower_bound + upper_bound) // 2
            cost = self.search_result(B, G, function_name, inference_time, device)
            if cost < overall_cost:
                lower_bound = B + 1
            else:
                upper_bound = B - 1
        best_spec = self.calc_specification(B, function_name, inference_time, device)
        instance_number = int(np.ceil(G / B))

        return instance_number, best_spec

    def default_auto_scaler(
        self,
        G,
        function_name,
        device,
    ):
        if device == "cpu":
            return G, 1
        if device == "cuda":
            return np.ceil(G / 32), 10
