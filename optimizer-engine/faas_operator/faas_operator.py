import copy
from kubernetes import client
import yaml
import base64
import os
from mpire import WorkerPool
from function_config import FunctionConfig
from cache.cache import Cache
from retrying import retry
from rich import print
from rich.logging import RichHandler
from rich.traceback import install
import logging
import warnings

install(show_locals=True)

FORMAT = "%(message)s"
logging.basicConfig(
    level="INFO", format=FORMAT, datefmt="[%X]", handlers=[RichHandler(level="INFO")]
)
log = logging.getLogger("rich")
warnings.filterwarnings("ignore")


class FaasOperator:
    def __init__(self, gateway, port, passwd, cache: Cache):
        # kconf.load_incluster_config()
        self.gateway = gateway
        self.port = port
        self.passwd = passwd
        self.DEFAULT_CUDA_UTIL = os.environ.get("DEFAULT_CUDA_UTIL", "0.2")
        self.DEFAULT_CPU_CORES = os.environ.get("DEFAULT_CPU_CORES", "1")
        self.DEFAULT_MEMORY = os.environ.get("DEFAULT_MEMORY", "4096")
        self.cache = cache
        os.system(
            f"./faas-cli login --without-output --tls-no-verify -g http://{self.gateway}:{self.port} -p {self.passwd}"
        )

    def get_faas_template(self, namespace, faas_template_name):
        """
        Retrieves a FaaS template from a specified namespace and template name.

        Args:
            namespace (str): The namespace of the FaaS template.
            faas_template_name (str): The name of the FaaS template.

        Returns:
            None: If the FaaS template is not found or there is an error decoding it.
        """

        faas_template_configmap = client.CoreV1Api().read_namespaced_config_map(
            faas_template_name, namespace
        )
        if "faas_template" not in faas_template_configmap.data:  # type: ignore
            print("faas template not found")
            return None
        faas_template_encoder = faas_template_configmap.data["faas_template"]  # type: ignore
        try:
            faas_template = base64.b64decode(faas_template_encoder).decode("utf-8")
        except Exception as e:
            print(f"faas template decode error, {e}")
            return None
        self.faas_template = yaml.safe_load(faas_template)

    def update_faas_template(
        self,
        func,
        types,
        namespace,
        resource,
        node_selector="",
        backend="cpu",
        use_model_controller="False",
        warm_cold="warm",
        batching="false",
    ):
        config = copy.deepcopy(self.faas_template)
        config["functions"][func] = copy.deepcopy(config["functions"][types])
        config["functions"][func]["namespace"] = f"{namespace}"
        cpu = resource["cpu"] if "cpu" in resource else self.DEFAULT_CPU_CORES
        memory = resource["memory"] if "memory" in resource else self.DEFAULT_MEMORY
        gpu = resource["cuda"] if "cuda" in resource else self.DEFAULT_CUDA_UTIL

        config["functions"][func]["limits"]["cpu"] = f"{cpu}"
        config["functions"][func]["limits"]["memory"] = f"{memory}Mi"
        if use_model_controller == "True":
            config["functions"][func]["requests"]["cpu"] = "800m"
        else:
            config["functions"][func]["requests"]["cpu"] = f"{cpu}"
        config["functions"][func]["requests"]["memory"] = f"{memory}Mi"
        config["functions"][func]["environment"][
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
        ] = f"{gpu}"
        config["functions"][func]["environment"]["BACKEND"] = f"{backend}"
        config["functions"][func]["environment"]["USE_MODEL_CONTROLLER"] = (
            f"{use_model_controller}"
        )
        config["functions"][func]["environment"]["batching"] = f"{batching}"
        config["functions"][func]["environment"]["batch_wait_timeout"] = f"{50}"
        config["functions"][func]["environment"]["cpu_quantity"] = f"{cpu}"
        config["functions"][func]["environment"]["memory_quantity"] = f"{memory}"
        config["functions"][func]["environment"]["gpu_quantity"] = f"{gpu}"
        config["functions"][func]["environment"]["warm_cold"] = f"{warm_cold}"
        if node_selector != "":
            config["functions"][func]["constraints"] = [
                f"kubernetes.io/hostname={node_selector}"
            ]
        self.dumps_faas_config(func, config)

    def dumps_faas_config(self, func, config):
        request_file = os.path.join(f"{func}.yml")
        with open(request_file, "w") as f:
            yaml.dump(config, f)

    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def update_deployment(self, func_name, namespace, target_replicas):
        # try:
        deployment = client.AppsV1Api().read_namespaced_deployment(func_name, namespace)
        deployment.spec.replicas = target_replicas  # type: ignore
        try:
            client.AppsV1Api().patch_namespaced_deployment(
                func_name, namespace, deployment
            )
        except Exception as e:
            raise e

    def deploy_faas(self, func_name, namespace):
        request_file = os.path.join(f"{func_name}.yml")
        os.system(
            f"./faas-cli deploy --without-output --tls-no-verify --update -f {request_file} -n {namespace} --filter {func_name}"
        )
        os.remove(request_file)
        self.update_deployment(func_name, namespace, 1)

    def delete_faas(self, func_name, namespace):
        os.system(
            f"./faas-cli remove --tls-no-verify --without-output --gateway=http://{self.gateway}:{self.port} -n {namespace} {func_name} --filter {func_name}"
        )

    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def create_function_cm(self, namespace, function_config: FunctionConfig):
        function_configmap = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
        )
        function_configmap.metadata = client.V1ObjectMeta(
            name=function_config.function_name
        )
        depth = {}
        for key, value in function_config.depth.items():
            depth[key] = str(value)
        function_config.depth = depth
        function_configmap.data = {
            "prewarm_window": str(
                function_config.prewarm_window["merge_prewarm_window"]
            ),
            "keep_alive_time": str(
                function_config.keep_alive_time["merge_keep_alive_time"]
            ),
            "depth": str(function_config.depth["merge_depth"]),
            "resource_type": str(function_config.device["merge_device"]),
            "resource_amount": str(
                function_config.keep_alive_resource["merge_keep_alive_resource"]
            ),
            "sharing": str(function_config.sharing),
        }
        try:
            client.CoreV1Api().create_namespaced_config_map(
                namespace, function_configmap
            )
        except Exception as e:
            raise e

    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def update_workflow_cm(self, namespace, workflow_config):
        workflow_configmap = client.V1ConfigMap()
        workflow_configmap.metadata = client.V1ObjectMeta(
            name=workflow_config.metadata.name,
            namespace=namespace,
            annotations={
                "faas_workflow_name": workflow_config.metadata.annotations[
                    "faas_workflow_name"
                ],
                "dag_parse_finished": "true",
            },
        )
        workflow_configmap.data = workflow_config.data
        try:
            client.CoreV1Api().patch_namespaced_config_map(
                workflow_config.metadata.name, namespace, workflow_configmap
            )
        except Exception as e:
            raise e

    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def update_function_cm(self, namespace, function_config: FunctionConfig):
        function_configmap = client.V1ConfigMap()
        function_configmap.metadata = client.V1ObjectMeta(
            name=function_config.function_name
        )
        function_configmap.data = {
            "prewarm_window": str(
                function_config.prewarm_window["merge_prewarm_window"]
            ),
            "depth": str(function_config.depth["merge_depth"]),
            "keep_alive_time": str(
                function_config.keep_alive_time["merge_keep_alive_time"]
            ),
            "resource_type": str(function_config.device["merge_device"]),
            "sharing": str(function_config.sharing),
            "target_replicas": str(function_config.target_replicas),
        }
        try:
            client.CoreV1Api().patch_namespaced_config_map(
                function_config.function_name, namespace, function_configmap
            )
        except Exception as e:
            raise e

    def get_function_cm(self, function_name, namespace):
        try:
            function_configmap = client.CoreV1Api().read_namespaced_config_map(
                function_name, namespace
            )
            return function_configmap
        except Exception as e:
            return None

    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def delete_function_cm(self, function_name, namespace):
        try:
            client.CoreV1Api().delete_namespaced_config_map(function_name, namespace)
        except Exception:
            print("delete failed")

    def update_function(self, namespace, updated_functions_list):
        cm_op = []

        def _update_function(function_name):
            function_config = self.cache.get_func_config(function_name)

            function_cm_existence = self.check_function_resource_existence(
                "configmap", function_name, namespace
            )
            if not function_cm_existence:
                cm_op.append(("create", namespace, function_config))
            else:
                cm_op.append(("update", namespace, function_config))

        def _cm_op(op, namespace, function_config):
            if op == "create":
                self.create_function_cm(namespace, function_config)
            else:
                self.update_function_cm(namespace, function_config)
            return f"update function {function_config.function_name} successfully"

        if len(updated_functions_list) == 0:
            return
        for function_name in updated_functions_list:
            _update_function(function_name)
        with WorkerPool(10) as pool:
            _ = pool.map(_cm_op, cm_op)

    def check_function_resource_existence(
        self, resource_name, function_name, namespace
    ):
        function_resource = None
        if resource_name == "configmap":
            function_resource = self.get_function_cm(function_name, namespace)
        if function_resource is None:
            return False
        else:
            return True

    def delete_function(self, namespace, workflow_functions, workflow_name):
        def _delete_function(function_name):
            function_config = self.cache.get_func_config(function_name)
            updated_func_config, update_or_not = self.cache.delete_function(
                function_name, workflow_name
            )
            if updated_func_config is None:
                self.delete_faas(function_name, namespace)
                self.delete_function_cm(function_name, namespace)
            elif update_or_not:
                self.update_function_cm(namespace, updated_func_config)
                resource = {}
                if function_config.device == "cpu":
                    resource["cpu"] = function_config.keep_alive_resource[
                        "merge_keep_alive_resource"
                    ]
                else:
                    resource["cuda"] = function_config.keep_alive_resource[
                        "merge_keep_alive_resource"
                    ]
                prewarm_window = function_config.prewarm_window["merge_prewarm_window"]
                warm_cold = "warm"
                if prewarm_window == -1:
                    warm_cold = "warm"
                else:
                    warm_cold = "cold"
                types = function_name.split("-")[0]
                self.update_faas_template(
                    function_name,
                    types,
                    namespace,
                    resource,
                    backend=function_config.device["merge_device"],
                    use_model_controller="False",
                    warm_cold=warm_cold,
                    batching="true",
                )
                self.deploy_faas(function_name, namespace)
            log.debug(f"delete function {function_name} successfully")
            return (function_name, updated_func_config)

        for function_name in workflow_functions:
            result = _delete_function(function_name)
            if result[1] is not None:
                self.cache.update_func_config(result[0], result[1])
            else:
                self.cache.remove_func_config(result[0])
        self.cache.delete_workflow_functions(workflow_name)

    def deploy_function(self, namespace, function_list):
        """
        Deploy a list of functions to a Kubernetes cluster.

        Args:
            function_list (list): A list of function names (strings).

        Returns:
            None

        Raises:
            None

        The function uses a `WorkerPool` to deploy each function in parallel.
        For each function, it retrieves its configuration from a cache, and
        uses the `faas_operator` object to update the Kubernetes deployment
        template with the appropriate resources and image. It also sets some
        function-specific parameters such as the backend device and batching.
        Finally, it deploys the function using the `faas_operator` and returns
        a success message.

        Note that the function assumes that some variables such as `namespace`
        and `kube_operator` are already defined in the global scope.

        Example usage:

        >>> deploy_function(['myfunc1', 'myfunc2', 'myfunc3'])
        """

        def _deploy_function(function_name):
            function_config = self.cache.get_func_config(function_name)
            resource = {}
            if function_config.device["merge_device"] == "cpu":
                resource["cpu"] = function_config.keep_alive_resource[
                    "merge_keep_alive_resource"
                ]
            else:
                resource["cuda"] = function_config.keep_alive_resource[
                    "merge_keep_alive_resource"
                ]
            types = function_name.split("-")[0]
            prewarm_window = function_config.prewarm_window["merge_prewarm_window"]
            warm_cold = "warm"
            if prewarm_window < 0:
                warm_cold = "warm"
            else:
                warm_cold = "cold"
            self.update_faas_template(
                function_name,
                types,
                namespace,
                resource,
                backend=function_config.device["merge_device"],
                use_model_controller="False",
                warm_cold=warm_cold,
                batching="true",
            )
            self.deploy_faas(function_name, namespace)
            return f"deploy function {function_name} successfully"

        with WorkerPool(10) as pool:
            results = pool.imap(_deploy_function, function_list)
            for result in results:
                log.debug(result)
