from prometheus_api_client import PrometheusConnect
from pandas import DataFrame


class PodKeepAliveCost:
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
        # pod_info_query = f"container_cpu_cfs_periods_total{{namespace='{self.namespace}',container!~'POD|kube-state-metrics|openfaas',image!='k8s.gcr.io/pause:3.6',name!=''}}"
        # pod_info_result = {
        #     "pod_name": [],
        #     "node": [],
        #     "container": [],
        # }
        # result = self.prom.custom_query_range(
        #     pod_info_query, start_time=st, end_time=et, step="100ms"
        # )
        # if len(result) > 0:
        #     for r in result:
        #         pod = r["metric"]["pod"]
        #         node = r["metric"]["node"]
        #         container = r["metric"]["container"]
        #         pod_info_result["pod_name"].append(pod)
        #         pod_info_result["node"].append(node)
        #         pod_info_result["container"].append(container)
        # pod_info_df = DataFrame(pod_info_result)
        # print(pod_info_df)
        # all_pod_df = all_pod_df.merge(pod_info_df, on="pod_name")
        return all_pod_df

    # def get_pod_create_timestamp(self, pod_name):
    #     import datetime

    #     end_time = datetime.datetime.now()
    #     query = f"container_start_time_seconds{{namespace='{self.namespace}',pod='{pod_name}',pod!='POD',image!='k8s.gcr.io/pause:3.6',name!=''}}"
    #     # query = f"container_start_time_seconds{{namespace='{self.namespace}',pod='{pod_name}'}}"
    #     # print(query)
    #     start_time = self.prom.custom_query_range(
    #         query, start_time=self.start_time, end_time=end_time, step="1s"
    #     )
    #     create_timestamp = []
    #     if len(start_time) > 0:
    #         for i in range(0, len(start_time[0]["values"])):
    #             create_timestamp.append(int(start_time[0]["values"][i][1]))
    #         return min(create_timestamp)
    #     else:
    #         return None

    # def get_pod_terminate_timestamp(self, pod_name):
    #     import datetime

    #     end_time = datetime.datetime.now()
    #     query = f"container_last_seen{{namespace='{self.namespace}',pod='{pod_name}',pod!='POD',image!='k8s.gcr.io/pause:3.6',name!=''}}"
    #     # query = f"container_last_seen{{namespace='{self.namespace}',pod='{pod_name}'}}"
    #     last_seen = self.prom.custom_query_range(
    #         query, start_time=self.start_time, end_time=end_time, step="1s"
    #     )
    #     terminate_timestamp = []
    #     if len(last_seen) > 0:
    #         for i in range(0, len(last_seen[0]["values"])):
    #             terminate_timestamp.append(int(last_seen[0]["values"][i][1]))
    #         return max(terminate_timestamp)
    #         # return int(last_seen[0]["values"][len(last_seen[0]["values"])-1][1])
    #     else:
    #         return None

    # def get_pod_keep_alive_cost(self, pod_name, start_time):
    #     create_time = self.get_pod_create_timestamp(pod_name)
    #     if create_time < start_time:
    #         create_time = start_time
    #     terminate_time = self.get_pod_terminate_timestamp(pod_name)
    #     if create_time is not None and terminate_time is not None:
    #         return terminate_time - create_time
    #     else:
    #         return -1
