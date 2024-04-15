from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# example of bayesian optimization for a 1d function from scratch
from numpy import argmax
from scipy.stats import norm
from warnings import catch_warnings
from warnings import simplefilter
import pandas as pd
import numpy as np
from .optimizer import Optimizer


class Aquatope(Optimizer):
    def __init__(self, file_path, function_profiler):
        self.file_path = "offline_training_data/result.csv"
        self.datasets, self.functions = self.get_datasets()
        self.device = {"cpu": 0, "cuda": 1}
        self.function_profiler = function_profiler
        self.warm_or_cold = {"warm": 0, "cold": 1}
        self.workflow_strategy = {}

    def get_total_cost(self, result):
        import warnings

        warnings.filterwarnings("ignore")
        result["total_cost"] = result["running_cost"] + result["keep_alive_cost"]
        running_status = result[
            [
                "optimizer",
                "process_time",
                "pod_name",
                "function_name",
                "running_cost",
                "keep_alive_cost",
                "total_cost",
                "sharing",
                "sla",
                "round",
            ]
        ]
        model_total_load_cost = (
            result[
                [
                    "optimizer",
                    "function_name",
                    "pod_name",
                    "model_load_cost",
                    "sharing",
                    "sla",
                ]
            ]
            .drop_duplicates(
                subset=["optimizer", "function_name", "pod_name", "sharing", "sla"]
            )
            .groupby(["optimizer", "function_name", "sharing", "sla"])
            .sum()
            .reset_index()
        )

        function_running_cost = (
            running_status[
                [
                    "optimizer",
                    "pod_name",
                    "function_name",
                    "running_cost",
                    "sharing",
                    "sla",
                    "round",
                ]
            ]
            .drop_duplicates()
            .groupby(["optimizer", "function_name", "sharing", "sla"])
            .sum()
            .reset_index()[
                ["optimizer", "function_name", "running_cost", "sharing", "sla"]
            ]
        )
        function_running_cost = pd.merge(
            function_running_cost,
            model_total_load_cost,
            on=["optimizer", "function_name", "sharing", "sla"],
        )
        function_keep_alive_cost = (
            running_status[
                [
                    "optimizer",
                    "pod_name",
                    "function_name",
                    "keep_alive_cost",
                    "sharing",
                    "sla",
                ]
            ]
            .drop_duplicates()
            .groupby(["optimizer", "pod_name", "sharing", "sla"])
            .max()
            .reset_index()
            .groupby(["optimizer", "function_name", "sharing", "sla"])
            .sum()
            .reset_index()[
                ["optimizer", "function_name", "keep_alive_cost", "sharing", "sla"]
            ]
        )
        function_costs = pd.merge(
            function_running_cost,
            function_keep_alive_cost,
            on=["optimizer", "function_name", "sharing", "sla"],
        )
        function_costs["total_cost"] = (
            function_costs["running_cost"]
            + function_costs["keep_alive_cost"]
            + function_costs["model_load_cost"]
        )
        return function_costs["total_cost"].sum()

    def extend(self, x):
        tmp = []
        data_function_names = x["function_name"].unique().tolist()
        for func in self.functions:
            if func in data_function_names:
                tmp.append(
                    self.device[x[x["function_name"] == func]["device"].values[0]]
                )
                tmp.append(x[x["function_name"] == func]["resource_quantity"].values[0])
                tmp.append(
                    self.warm_or_cold[
                        x[x["function_name"] == func]["warm_cold"].values[0]
                    ]
                )
            else:
                tmp.append(-1)
                tmp.append(0)
                tmp.append(-1)
        total_cost = self.get_total_cost(x)
        tmp.append(x["workflow_running_time"].max())
        tmp.append(total_cost)
        return tmp

    def get_function_history_data(self, path):
        history_data = pd.read_csv(path)
        functions = history_data["function_name"].unique().tolist()
        functions = sorted(functions)
        return history_data, functions

    def get_datasets(self):
        history_data, functions = self.get_function_history_data(self.file_path)
        datasets = (
            history_data.groupby(["workflow_name", "sharing", "round", "optimizer"])
            .apply(self.extend)
            .apply(pd.Series)
            .reset_index()[[i for i in range(0, 3 * len(self.functions) + 2)]]
        )
        return datasets, functions

    def gpModel(self, X, y):
        kernel = Matern(length_scale=1.0, nu=2.5)
        model = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-7, n_restarts_optimizer=10
        )
        model.fit(X, y)
        return model

    def surrogate(self, model, X):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return np.array(model.predict(X, return_std=True))

    def acquisition_SLA(self, X, Xsamples, model, SLA):
        # calculate the best surrogate score found so far
        yhat, _ = self.surrogate(model, X)
        yhat = yhat[yhat <= SLA]
        best = max(yhat)
        # calculate mean and stdev via surrogate function
        mu, std = self.surrogate(model, Xsamples)
        # mu = mu[:, 0]
        # calculate the probability of improvement
        probs = norm.cdf((mu - best) / (std + 1e-9))
        return probs

    def acquisition_cost(self, X, Xsamples, model):
        # calculate the best surrogate score found so far
        yhat, _ = self.surrogate(model, X)
        yhat = -yhat
        best = max(yhat)
        # calculate mean and stdev via surrogate function
        mu, std = self.surrogate(model, Xsamples)
        probs = norm.cdf((mu - best) / (std + 1e-9))
        return probs

    def generate_search_space(self, expected_function_names):
        import random

        search_space = []
        device_space = [0, 1]
        resource_quantity_space = [1, 2, 3, 4]
        warm_cold_space = [0, 1]
        for i in range(0, 20):
            tmp = []
            for func in self.functions:
                if func in expected_function_names:
                    tmp.append(device_space[random.randint(0, 1)])
                    tmp.append(resource_quantity_space[random.randint(0, 3)])
                    tmp.append(warm_cold_space[random.randint(0, 1)])
                else:
                    tmp.append(-2)
                    tmp.append(0)
                    tmp.append(0)
            search_space.append(tmp)
        return search_space

    def trans_to_execute_strategy(self, x, expected_function_names):
        import numpy as np

        device = {0: "cpu", 1: "cuda"}
        warm_or_cold = {0: "True", 1: "False"}
        tmp = {}
        # split x into 10 groups, each group has 3 elements
        function_strategies = np.array_split(x, 10)
        count = 0
        for strategy in function_strategies:
            if strategy[0] == -2:
                continue
            tmp[expected_function_names[count]] = {
                "device": device[strategy[0]],
                "resource_quantity": strategy[1],
                "warm_cold": warm_or_cold[strategy[2]],
            }
            count += 1
        return tmp

    def opt_acquisition(self, expected_function_names, X, model1, model2, SLA):
        # random search, generate random samples
        Xsamples = self.generate_search_space(expected_function_names)
        # Xsamples = Xsamples.reshape(len(Xsamples), 1)
        # calculate the acquisition function for each sample
        sla_scores = self.acquisition_SLA(X, Xsamples, model1, SLA)
        cost_scores = self.acquisition_cost(X, Xsamples, model2)
        scores = sla_scores + cost_scores
        # locate the index of the largest scores
        ix = argmax(scores)
        return Xsamples[ix]

    def get_workflow_running_plan_df(
        self,
        workflow_name,
        graph_df,
        graph_dfs,
        IT,
        SLA,
        interval_time_unit,
    ):
        expected_function_names = graph_df["types"].unique().tolist()
        X_train = self.datasets[[i for i in range(0, 3 * len(self.functions))]]
        y_train_sla = self.datasets[[3 * len(self.functions)]]
        y_train_cost = self.datasets[[3 * len(self.functions) + 1]]
        model_sla = self.gpModel(X_train, y_train_sla)
        model_cost = self.gpModel(X_train, y_train_cost)
        x = self.opt_acquisition(
            expected_function_names, X_train, model_sla, model_cost, SLA
        )
        result = self.trans_to_execute_strategy(x, expected_function_names)
        graph_df["device"] = graph_df["types"].apply(lambda x: result[x]["device"])
        graph_df["resource_quantity"] = graph_df["types"].apply(
            lambda x: result[x]["resource_quantity"]
        )
        graph_df["keep_alive_resource"] = graph_df["resource_quantity"]
        graph_df["cold_start_stage"] = graph_df["types"].apply(
            lambda x: result[x]["warm_cold"]
        )
        graph_df["function_prewarm_time"] = graph_df["types"].apply(
            lambda x: -1 if result[x]["warm_cold"] == "warm" else 0
        )
        graph_df["knee_point"] = 1
        graph_df["keep_alive_time"] = graph_df["types"].apply(
            lambda x: -1 if result[x]["warm_cold"] == "warm" else 0
        )
        graph_df[
            [
                "cpu_running_time",
                "cpu_running_cost",
                "gpu_running_cost",
                "cpu_keep_alive_cost",
                "gpu_keep_alive_cost",
            ]
        ] = graph_df.apply(
            lambda x: self.function_profiler.get_cpu_cost_running_time(
                x["types"], x["resource_quantity"], x["image_size"]
            ),
            axis=1,
            result_type="expand",
        )
        nodes = graph_df["node"].tolist()
        self.workflow_strategy[workflow_name] = graph_df
        return graph_df, nodes

    def update_datasets(self):
        self.datasets, self.functions = self.get_datasets()

    def update_workflow_running_plan_df(
        self, workflow_name, IT, SLA, interval_time_unit, 
    ):
        last_graph_df = self.workflow_strategy[workflow_name]
        self.update_datasets()
        graph_df, nodes = self.get_workflow_running_plan_df(
            workflow_name=workflow_name,
            graph_df=last_graph_df,
            graph_dfs=None,
            IT=IT,
            SLA=SLA,
            interval_time_unit=interval_time_unit,
        )
        return graph_df, nodes

    def remove_workflow(self, workflow_name):
        del self.workflow_strategy[workflow_name]
