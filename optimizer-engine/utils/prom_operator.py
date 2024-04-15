from prometheus_api_client import PrometheusConnect
from pandas import DataFrame


class PrometheusOperator:
    def __init__(self, start_time, prom_url, namespace):
        self.prom_url = prom_url
        self.namespace = namespace
        self.prom = PrometheusConnect(url=prom_url, disable_ssl=True)
        self.start_time = start_time

    def get_all_pod_life_time_from_ns(self, st, et):
        # get all pod from given namespace from st to et
        import datetime

        exec_time_result = {
            "pod_name": [],
            "pod_start_time": [],
            "pod_end_time": [],
            "pod_exec_time": [],
            "container": [],
            "node": [],
        }
        query = f"kube_pod_info{{namespace='{self.namespace}',container!~'POD|kube-state-metrics|openfaas'}}"
        result = self.prom.custom_query_range(
            query, start_time=st, end_time=et, step="100ms"
        )
        # record all pod start time and end time
        if len(result) > 0:
            for r in result:
                pod = r["metric"]["pod"]
                node = r["metric"]["node"]
                pod_st = r["values"][0][0]
                pod_et = r["values"][-1][0]
                pod_exec_time = pod_et - pod_st
                exec_time_result["pod_name"].append(pod)
                exec_time_result["pod_start_time"].append(pod_st)
                exec_time_result["pod_end_time"].append(pod_et)
                exec_time_result["pod_exec_time"].append(pod_exec_time)
                exec_time_result["container"].append(pod.split("-")[0])
                exec_time_result["node"].append(node)
        all_pod_df = DataFrame(exec_time_result)
        return all_pod_df

    def _parse_function_info_result(result, key):
        tmp = {
            "namespace": [],
            "pod_name": [],
            "function_name": [],
            "backend": [],
            "batch_size": [],
            "resource_quantity": [],
            "request_id": [],
            key: [],
        }
        for r in result:
            tmp["namespace"].append(r["metric"]["namespace"])
            tmp["pod_name"].append(r["metric"]["pod_name"])
            tmp["function_name"].append(r["metric"]["function_name"])
            tmp["backend"].append(r["metric"]["backend"])
            tmp["batch_size"].append(r["metric"]["batch_size"])
            tmp["resource_quantity"].append(r["metric"]["resource_quantity"])
            tmp["request_id"].append(r["metric"]["request_id"])
            tmp[key].append(eval(r["value"][1]))
        return DataFrame(tmp)

    def _parse_workflow_info_result(result):
        tmp = {"namespace": [], "request_id": [], "workflow_running_time": []}
        for r in result:
            tmp["namespace"].append(r["metric"]["namespace"])
            tmp["request_id"].append(r["metric"]["request_id"])
            tmp["workflow_running_time"].append(eval(r["value"][1]))
        return DataFrame(tmp)

    def get_function_name_execution_time_info(self, st, et):
        process_time_query = f"function_process_time{{namespace='{self.namespace}'}}"
        model_predict_time_query = (
            f"function_model_predict_time{{namespace='{self.namespace}'}}"
        )
        model_load_time_query = (
            f"function_model_load_time{{namespace='{self.namespace}'}}"
        )
        model_transfer_time = (
            f"function_model_transfer_time{{namespace='{self.namespace}'}}"
        )
        workflow_running_time = (
            f"function_workflow_running_time{{namespace='{self.namespace}'}}"
        )
        process_time_result = self.prom.custom_query_range(
            process_time_query, start_time=st, end_time=et, step="100ms"
        )
        model_predict_time_result = self.prom.custom_query_range(
            model_predict_time_query, start_time=st, end_time=et, step="100ms"
        )
        model_load_time_result = self.prom.custom_query_range(
            model_load_time_query, start_time=st, end_time=et, step="100ms"
        )
        model_transfer_time_result = self.prom.custom_query_range(
            model_transfer_time, start_time=st, end_time=et, step="100ms"
        )
        workflow_running_time_result = self.prom.custom_query_range(
            workflow_running_time, start_time=st, end_time=et, step="100ms"
        )
        process_time_df = self._parse_function_info_result(
            process_time_result, "process_time"
        )
        model_predict_time_df = self._parse_function_info_result(
            model_predict_time_result, "model_predict_time"
        )
        model_load_time_df = self._parse_function_info_result(
            model_load_time_result, "model_load_time"
        )
        model_transfer_time_df = self._parse_function_info_result(
            model_transfer_time_result, "model_transfer_time"
        )
        workflow_running_time_df = self._parse_function_info_result(
            workflow_running_time_result, "workflow_running_time"
        )
        total_result = (
            process_time_df.merge(
                model_predict_time_df,
                on=[
                    "namespace",
                    "pod_name",
                    "function_name",
                    "backend",
                    "batch_size",
                    "resource_quantity",
                    "request_id",
                ],
            )
            .merge(
                model_load_time_df,
                on=[
                    "namespace",
                    "pod_name",
                    "function_name",
                    "backend",
                    "batch_size",
                    "resource_quantity",
                    "request_id",
                ],
            )
            .merge(
                model_transfer_time_df,
                on=[
                    "namespace",
                    "pod_name",
                    "function_name",
                    "backend",
                    "batch_size",
                    "resource_quantity",
                    "request_id",
                ],
            )
        )
        total_result = total_result.merge(
            workflow_running_time_df, on=["namespace", "request_id"]
        )
        return total_result
