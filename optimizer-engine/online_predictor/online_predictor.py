import torch
import time
import os
import pandas as pd
import numpy as np
from online_predictor.prometheus_operator import PrometheusOperator
from concurrent.futures import ThreadPoolExecutor
from cache.invocation_infos import InvocationInfos
from threading import Lock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class OnlinePredictor:
    def __init__(
        self,
        prom_operator: PrometheusOperator,
        invocation_infos: InvocationInfos,
        local_mode=False,
    ):
        self.prom_operator = prom_operator
        self.load_model_file_from_local()
        self.inter_arrival_time = {}
        self.old_inter_arrival_time = {}
        self.old_invocation_number = {}
        self.invocation_infos = invocation_infos
        self.function_names = [
            "objectdetection",
            "imagerecognition",
            "speechrecognition",
        ]
        invocation_numbers = self._load_invocation_number_series()
        for function_name in self.function_names:
            self.invocation_infos.init_function_invocation_number_from_list(
                function_name, invocation_numbers[function_name]
            )
        if not local_mode:
            self.monitor = {}
            self.monitor_executor = ThreadPoolExecutor(max_workers=32)

    def start_monitor(self, function_name):
        with Lock():
            self.monitor[function_name] = True
        self.monitor_executor.submit(self.__monitor_function, function_name)

    def stop_monitor(self, function_name):
        with Lock():
            self.monitor[function_name] = False
            # remove function monitor from dict self.monitor
            del self.monitor[function_name]

    def __monitor_function(self, function_name):
        while True:
            with Lock():
                if function_name not in self.monitor:
                    break
                if not self.monitor[function_name]:
                    break
            self.update_invocation_number_inter_arrival_time_from_gateway(function_name)
            time.sleep(1)

    def update_invocation_number_inter_arrival_time_from_gateway(self, function_name):
        data = self.prom_operator.get_function_invocation_number(
            function_name, "openfaas-fn", "gateway_service_count"
        )
        invocation_number = data[function_name]["gateway_service_count"]
        self.invocation_infos.update_invocation_info(function_name, invocation_number)

    def _load_invocation_number_series(self):
        df = pd.read_csv(os.path.join(SCRIPT_DIR, "selected_function.csv"))
        df = df.T
        df.columns = df.iloc[0]
        df = df.drop(df.index[0]).reset_index(drop=True)
        df = df.astype("float32")
        df = df[self.function_names]
        invocation_numbers = {}
        for function_name in self.function_names:
            invocation_numbers[function_name] = list(df[function_name].values)
        return invocation_numbers

    def update_inter_arrival_time(self, workflow_entry, work_round):
        self.invocation_infos.update_function_inter_arrival_time_from_list(
            workflow_entry, work_round
        )

    def load_model_file_from_local(self):
        (
            self.inter_arrival_time_models,
            self.inter_arrival_time_n_ins,
            self.inter_arrival_time_n_outs,
            self.inter_arrival_time_intervals,
        ) = self.__load_files_from_dir(
            os.path.join(SCRIPT_DIR, "inter_arrival_time_model"),
            model_type="inter_arrival_time",
        )
        (
            self.invocation_number_models,
            self.invocation_number_n_ins,
            self.invocation_number_n_outs,
            self.invocation_number_intervals,
        ) = self.__load_files_from_dir(
            os.path.join(SCRIPT_DIR, "invocation_number_model"),
            model_type="invocation_number",
        )
        (
            self.bursty_invocation_number_models,
            self.bursty_invocation_number_n_ins,
            self.bursty_invocation_number_n_outs,
            self.bursty_invocation_number_intervals,
        ) = self.__load_files_from_dir(
            os.path.join(SCRIPT_DIR, "bursty_invocation_number_model"),
        )

    def __load_files_from_dir(self, directory, model_type):
        models = {}
        n_ins = {}
        n_outs = {}
        intervals = {}
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                if filename.endswith(".pt"):
                    filenames = os.path.splitext(filename)
                    function_name = filenames[0].split("-")[0]
                    n_in = int(filenames[0].split("-")[1])
                    if model_type == "inter_arrival_time":
                        n_out = int(filenames[0].split("-")[2])
                        interval = 1
                    elif model_type == "invocation_number":
                        n_out = 1
                        interval = int(filenames[0].split("-")[2])
                    models[function_name] = torch.load(
                        os.path.join(directory, filename)
                    )
                    n_ins[function_name] = n_in
                    n_outs[function_name] = n_out
                    interval[function_name] = interval
        return models, n_ins, n_outs, intervals

    def min_max_scaler(self, inter_arrival_time, invocation_number):
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()
        scaler1 = MinMaxScaler()

        scaling_inter_arrival_time = inter_arrival_time
        scaler.fit(scaling_inter_arrival_time)
        scaling_invocation_number = invocation_number
        scaler1.fit(scaling_invocation_number)
        return scaler, scaler1

    def __predict_inter_arrival_time(
        self, model, n_in, n_out, scaler, scaler1, input_series, input_invocation_number
    ):
        scaling_inter_arrival_time = scaler.transform(input_series)
        scaling_invocation_number = scaler1.transform(input_invocation_number)
        input_data = torch.tensor(
            np.array(
                [
                    [i, j]
                    for i, j in zip(
                        scaling_inter_arrival_time, scaling_invocation_number
                    )
                ]
            ).astype(np.float32)
        )
        with torch.no_grad():
            output = model(input_data)
            output = output.detach().numpy().flatten()[0]
        output = scaler.inverse_transform(max(round(output), 0))
        return output

    def predict_inter_arrival_time(
        self, function_name, interval_time_unit, optimizer=None
    ):
        inter_arrival_time = self.invocation_infos.get_inter_arrival_time(function_name)
        # inter_arrival_time = self.inter_arrival_time[function_name]
        n_in = self.inter_arrival_time_n_ins[function_name]
        model = self.inter_arrival_time_models[function_name]
        if optimizer == "SMIlessOPT":
            inter_arrival_times = self.old_inter_arrival_time[function_name]
            inter_arrival_times_str = ",".join([str(i) for i in inter_arrival_times])
            inter_arrival_time_str = ",".join(
                [str(i) for i in inter_arrival_time[-n_in:]]
            )
            position = inter_arrival_times_str.find(inter_arrival_time_str)
            if position == -1:
                return 0
            else:
                return inter_arrival_times[position + n_in] * interval_time_unit

        invocation_number = self.invocation_infos.get_invocation_number(function_name)
        if len(inter_arrival_time) < n_in or len(invocation_number) < n_in:
            return 0

        scaler, scaler1 = self.min_max_scaler(inter_arrival_time, invocation_number)

        input_series = inter_arrival_time[-n_in:]
        input_invocation_number = invocation_number[-n_in:]
        output = self.__predict_inter_arrival_time(
            model, n_in, 1, scaler, scaler1, input_series, input_invocation_number
        )
        return output * interval_time_unit

    def _predict_invocation_number(self, n_in, interval, invocation_number, model):
        if len(invocation_number) == 0:
            return 1
        if len(invocation_number) < n_in:
            return round(max(max(invocation_number), 1))
        else:
            input_series = invocation_number[-n_in:]
            input_data = torch.tensor(np.array([input_series]).astype(np.float32))
            with torch.no_grad():
                output = model(input_data)
                output = output.detach().numpy().flatten()[0]
            output = max(round(output), 0) * interval
            return output

    def predict_invocation_number(
        self, function_name, inter_arrival_time, optimizer=None, test_type=None
    ):
        # suppose the invocation number during the inter_arrival_time is 0, then predicted invocation number of each time of next %inter_arrival_time and return the maximum value as the predicted invocation number.
        invocation_number = self.invocation_infos.get_invocation_number(function_name)
        if optimizer == "SMIlessOPT":
            invocation_numbers = self.old_invocation_number[function_name]
            invocation_numbers_str = ",".join([str(i) for i in invocation_numbers])
            invocation_number_str = ",".join([str(i) for i in invocation_number])
            position = invocation_numbers_str.find(invocation_number_str)
            if position == -1:
                return 1
            else:
                return invocation_numbers[position + 1]
        if test_type == "bursty":
            model = self.bursty_invocation_number_models[function_name]
            n_in = self.bursty_invocation_number_n_ins[function_name]
            interval = self.bursty_invocation_number_intervals[function_name]
        else:
            model = self.invocation_number_models[function_name]
            n_in = self.invocation_number_n_ins[function_name]
            interval = self.invocation_number_intervals[function_name]
        predicted_invocation_numbers = []
        if inter_arrival_time > 1:
            for i in range(inter_arrival_time - 1):
                mock_invocation_number = invocation_number.append(0)
                predicted_invocation_numbers.append(
                    self._predict_invocation_number(
                        n_in, interval, mock_invocation_number, model
                    )
                )
        else:
            predicted_invocation_numbers.append(
                self._predict_invocation_number(
                    n_in, interval, invocation_number, model
                )
            )
        return max(predicted_invocation_numbers)

    def get_local_inter_arrival_time(self, function_name):
        inter_arrival_time = self.invocation_infos.get_inter_arrival_time(function_name)
        n_in = self.inter_arrival_time_n_ins[function_name]
        if len(inter_arrival_time) < n_in:
            return [0] * n_in
        else:
            return inter_arrival_time[-n_in:]

    def get_invocation_number(self, function_name):
        invocation_number = self.invocation_infos.get_invocation_number(function_name)
        return invocation_number

    def reset_all(self):
        self.old_inter_arrival_time = self.invocation_infos.get_all_inter_arrival_time()
        self.old_invocation_number = self.invocation_infos.get_all_invocation_number()
        self.invocation_infos.clear_all()
