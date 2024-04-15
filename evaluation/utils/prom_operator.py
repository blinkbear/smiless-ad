from prometheus_api_client import PrometheusConnect
from pandas import DataFrame


class PodKeepAliveInfo:
    def __init__(self, start_time, prom_url, namespace):
        self.prom_url = prom_url
        self.namespace = namespace
        self.prom = PrometheusConnect(url=prom_url, disable_ssl=True)
        self.start_time = start_time

    def get_all_pod_from_ns(self, st, et, workflow_time):
        # get all pod from given namespace from st to et
        import datetime

        exec_time_result = {
            "pod_name": [],
            "pod_start_time": [],
            "pod_end_time": [],
            "pod_exec_time": [],
            "container":[],
            "node":[],
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
                if pod_et >workflow_time:
                    pod_et = workflow_time
                pod_exec_time = pod_et - pod_st
                exec_time_result["pod_name"].append(pod)
                exec_time_result["pod_start_time"].append(pod_st)
                exec_time_result["pod_end_time"].append(pod_et)
                exec_time_result["pod_exec_time"].append(pod_exec_time)
                exec_time_result['container'].append(pod.split("-")[0])
                exec_time_result['node'].append(node)
        all_pod_df = DataFrame(exec_time_result)
        return all_pod_df

   