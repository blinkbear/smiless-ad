import prometheus_api_client as prom_api
import time
from rich.traceback import install

install()


class PrometheusOperator:
    def __init__(self, prometheus_url):
        self.prometheus_url = prometheus_url
        self.prometheus_client = prom_api.PrometheusConnect(
            url=prometheus_url, disable_ssl=True
        )

    def get_function_metrics_range_data(
        self, function_name, namespace, metric_name, start_time, end_time
    ):
        metrics_range_data = {}
        query_function_name = f"{function_name}.{namespace}"
        label_config = {"function_name": query_function_name, "job": "openfaas-gateway"}
        metrics = self.prometheus_client.get_metric_range_data(
            metric_name,
            label_config=label_config,
            start_time=start_time,
            end_time=end_time,
        )
        for m in metrics:
            ts = [m["values"][i][0] for i in range(len(m["values"]))]
            values = [int(m["values"][i][1]) for i in range(len(m["values"]))]
            metrics_range_data[function_name] = {
                "function_name": [query_function_name] * len(ts),
                "ts": ts,
                metric_name: values,
            }
        return metrics_range_data

    def get_function_invocation_number(self, function_name, namespace, metric_name):
        metric_data = {}
        query_function_name = f"{function_name}.{namespace}"
        label_config = {"function_name": query_function_name, "job": "openfaas-gateway"}
        metrics = self.prometheus_client.get_metric_irate_value(
            metric_name, offset="1s", label_config=label_config
        )
        for m in metrics:
            ts = m["value"][0]
            values = int(m["value"][1])
            metric_data[function_name] = {
                "function_name": query_function_name,
                "ts": ts,
                metric_name: values,
            }
        return metric_data



if __name__ == "__main__":
    st = time.time() - 10
    et = time.time()
    prometheus_url = "http://10.119.46.42:30091"
    prom_operator = PrometheusOperator(prometheus_url)
    print(
        prom_operator.get_function_invocation_number(
            "textgeneration", "openfaas-fn", "gateway_service_count"
        )
    )
