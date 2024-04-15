import os
import json
import yaml

SOURCEDIR = os.path.dirname(os.path.abspath(__file__))
REQUEST_DIR = "config"
INFERENCE_FILE = "inference.yml"
INITIALIZATION_FILE = "faas_initialization_cm.yml"


class InferenceConfigurator(object):
    def configure(
        self,
        func,
        cpu,
        memory,
        gpu,
        backend,
        node_selector,
        namespace,
        max_batch_size,
    ):
        import copy

        use_model_controller = False
        batching = "true"
        self.config = self.parse_faas_function_config_file()
        config = copy.deepcopy(self.config)
        config["functions"][func]["namespace"] = f"{namespace}"
        config["functions"][func]["limits"]["cpu"] = f"{cpu}"
        config["functions"][func]["limits"]["memory"] = f"{memory}Mi"
        config["functions"][func]["requests"]["cpu"] = f"{cpu}"
        config["functions"][func]["requests"]["memory"] = f"{memory}Mi"
        config["functions"][func]["environment"][
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
        ] = f"{gpu}"
        config["functions"][func]["environment"]["BACKEND"] = f"{backend}"
        config["functions"][func]["environment"]["USE_MODEL_CONTROLLER"] = (
            f"{use_model_controller}"
        )
        config["functions"][func]["environment"]["cpu_quantity"] = f"{cpu}"
        config["functions"][func]["environment"]["gpu_quantity"] = f"{gpu}"
        config["functions"][func]["constraints"] = [
            f"kubernetes.io/hostname={node_selector}"
        ]
        config["functions"][func]["environment"]["batching"] = f"{batching}"
        config["functions"][func]["environment"]["batchsize"] = f"{max_batch_size}"
        self.dumps_faas_function_config_file(config, func, namespace)

    def parse_faas_function_config_file(self):
        request_file = os.path.join(SOURCEDIR, REQUEST_DIR, INFERENCE_FILE)
        with open(request_file, "r") as f:
            request_config = yaml.safe_load(f)

        return request_config

    def parse_faas_initialization_config_file(self):
        request_file = os.path.join(SOURCEDIR, REQUEST_DIR, INITIALIZATION_FILE)
        with open(request_file, "r") as f:
            request_config = yaml.safe_load(f)

        return request_config

    def dumps_faas_function_config_file(self, config, func, namespace):
        request_file = os.path.join(SOURCEDIR, REQUEST_DIR, namespace, f"{func}.yml")
        with open(request_file, "w") as f:
            yaml.dump(config, f)

    def dumps_faas_initialization_config_file(self, config):
        request_file = os.path.join(SOURCEDIR, REQUEST_DIR, INITIALIZATION_FILE)
        with open(request_file, "w") as f:
            yaml.dump(config, f)

    def delete_tmp_config_file(self, func, namespace):
        tmp_config_file = os.path.join(SOURCEDIR, REQUEST_DIR, namespace, f"{func}.yml")
        if os.path.exists(tmp_config_file):
            os.remove(tmp_config_file)

    def update_function_initialization_cm(self, result):
        faas_initialization_config_file = self.parse_faas_initialization_config_file()
        function_initialization_data = json.loads(
            faas_initialization_config_file["data"]["initialization_time"]
        )
        for func in result:
            function_initialization_data[func]["cpu-sigma"] = result[func]["cpu-sigma"]
            function_initialization_data[func]["gpu-sigma"] = result[func]["gpu-sigma"]
            function_initialization_data[func]["cpu-mu"] = result[func]["cpu-mu"]
            function_initialization_data[func]["gpu-mu"] = result[func]["gpu-mu"]
        faas_initialization_config_file["data"]["initialization_time"] = json.dumps(
            function_initialization_data
        )
        self.dumps_faas_initialization_config_file(faas_initialization_config_file)
