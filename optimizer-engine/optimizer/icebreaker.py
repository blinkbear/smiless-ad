import numpy as np
import pandas as pd
from numpy import fft
from .optimizer import Optimizer
from offline_profiler.function_resource_profiler import FunctionProfiler
from cache.invocation_infos import InvocationInfos
from threading import Lock


class IceBreaker(Optimizer):
    def __init__(
        self, invocation_infos: InvocationInfos, function_profiler: FunctionProfiler
    ):
        self.max_main_memory = 32
        self.workflow_strategy = {}
        self.prediction_history_window = 10
        self.harmonics = 10
        self.local_window = 60
        self.mem_lists = {}
        self.real_lists = {}
        self.pred_lists = {}
        self.mem_weights = {}
        self.cs_weights = {}
        self.tn_weights = {}
        self.fp_weights = {}
        self.exe_time_costly = {}
        self.exe_time_cheap = {}
        self.cs_time_costly = {}
        self.cs_time_cheap = {}
        self.invocation_infos = invocation_infos
        self.trace = {}
        self.available_cpu_resource = 1
        self.function_profilers = function_profiler

    def get_cpu_time(self, df):
        df[
            [
                "cpu_running_time",
                "cpu_running_cost",
                "gpu_running_cost",
                "cpu_keep_alive_cost",
                "gpu_keep_alive_cost",
            ]
        ] = df.apply(
            lambda x: self.function_profilers.get_cpu_cost_running_time(
                x["types"], x["qps"], self.available_cpu_resource, x["image_size"]
            ),
            axis=1,
            result_type="expand",
        )
        return df

    def get_workflow_information(self, graph_df):
        entry_point = graph_df.iloc[0]["node"]
        for i, row in graph_df.iterrows():
            self.mem_lists[row["node"]] = row["image_size"]
            self.exe_time_costly[row["node"]] = row["gpu_running_time"]
            self.exe_time_cheap[row["node"]] = row["cpu_running_time"]
            self.cs_time_costly[row["node"]] = (
                row["cold_start_time"] + row["gpu_trans_time"]
            )
            self.cs_time_cheap[row["node"]] = row["cold_start_time"]
            self.load_function_trace(row["node"])
            if row["node"] not in self.pred_lists:
                self.pred_lists[row["node"]] = []
            if row["node"] not in self.real_lists:
                self.real_lists[row["node"]] = []
            self.get_weights_by_profiling(row["node"], entry_point)

    def get_keep_alive_time(self, graph_df):
        graph_df["keep_alive_time"] = 0
        for i, row in graph_df.iterrows():
            graph_df.loc[i, "keep_alive_time"] = self.pred_lists[row["node"]][-1]
        return graph_df

    def update_workflow_running_plan_df(
        self, workflow_name, IT, SLA, interval_time_unit
    ):
        graph_df = self.workflow_strategy[workflow_name]
        graph_df, nodes = self.get_workflow_running_plan_df(
            workflow_name=workflow_name,
            graph_df=graph_df,
            graph_dfs=None,
            IT=IT,
            SLA=SLA,
            interval_time_unit=interval_time_unit,
        )
        return graph_df, nodes

    def get_workflow_running_plan_df(
        self, workflow_name, graph_df, graph_dfs, IT, SLA, interval_time_unit, 
    ):
        graph_df = self.get_cpu_time(graph_df)
        # entry point is the first node in graph df
        self.get_workflow_information(graph_df)
        graph_df = self.get_device(graph_df)
        nodes = graph_df["node"].tolist()
        graph_df["keep_alive_resource"] = self.available_cpu_resource
        graph_df = self.get_keep_alive_time(graph_df)
        for i, row in graph_df.iterrows():
            if row["cold_start_stage"] == "False":
                graph_df["function_prewarm_time"] = -1
            else:
                graph_df["function_prewarm_time"] = 0
        graph_df["knee_point"] = graph_df.apply(
            lambda x: 32 if x["types"] == "cuda" else 1, axis=1
        )
        self.workflow_strategy[workflow_name] = graph_df
        return graph_df, nodes

    def memory_component(self, r_list, p_list, function_name):
        return 1 - (1.0 * self.mem_lists[function_name] / self.max_main_memory)

    def cs_component(self, r_list, p_list, function_name):
        # return(statistics.mean([cs_time_costly[index]/exe_time_costly[index],
        #                         cs_time_cheap[index]/exe_time_cheap[index]]))

        return 1 - (
            self.cs_time_costly[function_name] + self.exe_time_costly[function_name]
        ) / (self.cs_time_cheap[function_name] + self.exe_time_cheap[function_name])

    ##determine the number of cold starts experienced due to mis-prediction in the prediction_history_window
    def tn_component(self, r_list, p_list, index):
        if len(r_list) > self.prediction_history_window:
            try:
                rl = r_list[-1 * self.prediction_history_window :]
                pl = p_list[-1 * self.prediction_history_window :]
                ret = sum(
                    [rl[i] - pl[i] if rl[i] - pl[i] > 0 else 0 for i in range(len(pl))]
                ) / sum(rl)
                if ret < 1:
                    return ret
                else:
                    return 1
            except Exception:  ##div by 0
                return 0
        else:
            return 1

    ##determine the number of time it was predicted to occur but it did not appear wasting in memory cost
    def fp_component(self, r_list, p_list, index):
        if len(r_list) > self.prediction_history_window:
            try:
                rl = r_list[-1 * self.prediction_history_window :]
                pl = p_list[-1 * self.prediction_history_window :]
                ret = 1 - (
                    sum(
                        [
                            pl[i] - rl[i] if pl[i] - rl[i] > 0 else 0
                            for i in range(len(pl))
                        ]
                    )
                    / sum(rl)
                )
                if ret > 0:
                    return ret
                else:
                    return 0
            except Exception:  ##div by 0
                return 0
        else:
            return 1

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

    def get_weights_by_profiling(self, function_name, entry_point):
        trace = self.get_traces(entry_point)
        r_list = []
        p_list = []
        for j in range(self.local_window, len(trace)):
            training_trace = np.array(trace[j - self.local_window : j])
            n_predict = 1
            extrapolation = self.fourier_extrapolation(training_trace, n_predict)
            pred_value = extrapolation[len(extrapolation) - 1]
            if pred_value < 0:
                pred_value = 0
            else:
                pred_value = round(pred_value)
            real_value = trace[j]

            r_list.append(real_value)
            p_list.append(pred_value)

        mem_ratio = self.memory_component(r_list, p_list, function_name)
        cs_ratio = self.cs_component(r_list, p_list, function_name)
        tn_ratio = self.tn_component(r_list, p_list, function_name)
        fp_ratio = self.fp_component(r_list, p_list, function_name)

        mem_weight = (
            1.0 / 3 * (1 - (mem_ratio / (mem_ratio + cs_ratio + tn_ratio + fp_ratio)))
        )
        cs_weight = (
            1.0 / 3 * (1 - (cs_ratio / (mem_ratio + cs_ratio + tn_ratio + fp_ratio)))
        )
        tn_weight = (
            1.0 / 3 * (1 - (tn_ratio / (mem_ratio + cs_ratio + tn_ratio + fp_ratio)))
        )
        fp_weight = (
            1.0 / 3 * (1 - (fp_ratio / (mem_ratio + cs_ratio + tn_ratio + fp_ratio)))
        )

        self.mem_weights[function_name] = mem_weight
        self.cs_weights[function_name] = cs_weight
        self.tn_weights[function_name] = tn_weight
        self.fp_weights[function_name] = fp_weight

        return (mem_weight, cs_weight, tn_weight, fp_weight)

    def load_function_trace(self, function_name):
        """
        load application call distribution
        """
        function_trace = self.invocation_infos.get_invocation_number(function_name)
        if len(function_trace) < self.local_window + self.prediction_history_window:
            function_type = function_name.split("-")[0]
            function_trace = pd.read_csv(
                f"baseline/trace/{function_type}.txt", header=None, nrows=120
            )
            function_trace = function_trace[0].tolist()
        with Lock():
            self.trace[function_name] = function_trace

    def get_traces(self, function_name):
        function_trace = self.invocation_infos.get_invocation_number(function_name)
        with Lock():
            self.trace[function_name] = self.trace[function_name].extend(function_trace)
        return self.trace[function_name]

    def get_device(self, graph_df):
        graph_df["device"] = "gpu_running_time"
        graph_df["cold_start_stage"] = "False"
        entry_point = graph_df.iloc[0]["node"]
        for i, row in graph_df.iterrows():
            function_name = row["node"]
            trace = self.get_traces(entry_point)
            val_list = []
            selected_system = 2
            if function_name not in self.real_lists:
                real_list = self.real_lists[function_name]
                predicted_list = self.pred_lists[function_name]
            else:
                real_list, predicted_list = [], []
            training_trace = np.array(trace[-self.local_window :])
            n_predict = 1
            extrapolation = self.fourier_extrapolation(training_trace, n_predict)
            pred_value = extrapolation[len(extrapolation) - 1]
            if pred_value < 0:
                pred_value = 0
            else:
                pred_value = round(pred_value)

            real_value = self.get_traces(function_name)[-1]
            real_list.append(real_value)
            predicted_list.append(pred_value)

            mem_ratio = self.memory_component(real_list, predicted_list, function_name)
            cs_ratio = self.cs_component(real_list, predicted_list, function_name)
            tn_ratio = self.tn_component(real_list, predicted_list, function_name)
            fp_ratio = self.fp_component(real_list, predicted_list, function_name)

            if len(real_list) > self.prediction_history_window:
                val = (
                    self.mem_weights[function_name] * mem_ratio
                    + self.cs_weights[function_name] * cs_ratio
                    + self.tn_weights[function_name] * tn_ratio
                    + self.fp_weights[function_name] * fp_ratio
                )
                val_list.append(val)

                if max(val_list) == min(val_list):
                    mm = max(val_list) - 0.0001
                else:
                    mm = min(val_list)

                weighted_val = (1.0 * val - mm) / (max(val_list) - mm)
                if weighted_val < 0.33:
                    selected_system = 0
                if (weighted_val >= 0.33) and (weighted_val < 0.66):
                    selected_system = 1
                if weighted_val >= 0.66:
                    selected_system = 2
            self.real_lists[function_name] = real_list
            self.pred_lists[function_name] = predicted_list
            if selected_system == 0:
                graph_df.loc[i, "cold_start_stage"] = "True"
                graph_df.loc[i, "device"] = "cpu"
            elif selected_system == 1:
                graph_df.loc[i, "cold_start_stage"] = "False"
                graph_df.loc[i, "device"] = "cpu"
            else:
                graph_df.loc[i, "cold_start_stage"] = "False"
                graph_df.loc[i, "device"] = "cuda"
        return graph_df

    def remove_workflow(self, workflow_name):
        del self.workflow_strategy[workflow_name]
