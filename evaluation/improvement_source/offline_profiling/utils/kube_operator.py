import json
from kubernetes import client
from kubernetes import config as kconf
import os
import yaml
from time import sleep
from retrying import retry


class KubeOperator:
    def __init__(self, namespace, SOURCEDIR):
        kconf.load_config()
        self.api = client.CoreV1Api()
        self.namespace = namespace
        self.auto_scale_config_path = os.path.join(SOURCEDIR, "config", namespace)
        self.auto_scale_config = self.read_auto_scale_config()


    def change_min_replicas(self, function_name, min_replicas):
        self.auto_scale_config['metadata']['name']=f"{function_name}-autoscale"
        self.auto_scale_config["spec"]["scaleTargetRef"]["name"] = function_name
        self.auto_scale_config["spec"]["minReplicaCount"] = min_replicas
        self.auto_scale_config["spec"]["pollingInterval"] = 20
        self.dump_auto_scale_config(function_name)
        self.apply_auto_scale_config(function_name)
    
    @retry(stop_max_attempt_number=3, wait_fixed=1000)
    def change_deployment(self, function_name, target_replicas):
        self.app_api = client.AppsV1Api()
        function_deploy = self.app_api.read_namespaced_deployment(
            function_name, self.namespace
        )
        function_deploy.spec.replicas = target_replicas
        self.app_api.patch_namespaced_deployment(
            function_name, self.namespace, function_deploy
        )

    def check_pod_ready(self, function_name):
        while True:
            status = []
            ret = self.api.list_namespaced_pod(self.namespace)
            for i in ret.items:
                if (
                    "faas_function" in i.metadata.labels
                    and i.metadata.labels["faas_function"] == function_name
                ):
                    status.append(i.status.phase)
            if len(status) > 1 or (
                "Terminating" in status
                or "Pending" in status
                or "ContainerCreating" in status
            ):
                sleep(1)
            else:
                break

    def check_pod_terminated(self, function_name):
        while True:
            status = []
            ret = self.api.list_namespaced_pod(self.namespace)
            for i in ret.items:
                if (
                    "faas_function" in i.metadata.labels
                    and i.metadata.labels["faas_function"] == function_name
                ):
                    status.append(i.status.phase)
            if len(status) > 0:
                sleep(1)
            else:
                break
            
    def check_configmap(self, configmap_name):
        ret = self.api.list_namespaced_config_map(self.namespace)
        for i in ret.items:
            if i.metadata.name == configmap_name:
                return True
        return False

    def create_or_update_configmap(self, configmap_name, data):
        configmap = client.V1ConfigMap()
        configmap.metadata = client.V1ObjectMeta(name=configmap_name)
        configmap.data = data
        if self.check_configmap():
            self.api.patch_namespaced_config_map(configmap_name, self.namespace, configmap)
        else:
            self.api.create_namespaced_config_map(self.namespace, configmap)

    def get_configmap(self, configmap_name):
        ret = self.api.list_namespaced_config_map(self.namespace)
        for i in ret.items:
            if i.metadata.name == configmap_name:
                return i
    
    
    def create_or_update_workflow_configmap(self,workflow_name,workflow_annotation,data):
        configmap = client.V1ConfigMap()
        configmap.metadata = client.V1ObjectMeta(name=workflow_name,annotations=workflow_annotation)
        configmap.data = data
        if self.check_workflow_configmap(workflow_name):
            self.api.patch_namespaced_config_map(workflow_name, self.namespace, configmap)
        else:
            self.api.create_namespaced_config_map(self.namespace, configmap)
    def check_workflow_configmap(self,workflow_name):
        ret = self.api.list_namespaced_config_map(self.namespace)
        for i in ret.items:
            if i.metadata.name == workflow_name:
                return True
        return False
    def delete_workflow_configmap(self,workflow_name):
        self.api.delete_namespaced_config_map(workflow_name, self.namespace)
        sleep(10)

    def check_workflow_configmap_ready(self, workflow_name):
        while True:
            ret = self.api.list_namespaced_config_map(self.namespace)
            for i in ret.items:
                if i.metadata.name == workflow_name:
                    if "dag_parse_finished" in i.metadata.annotations and i.metadata.annotations['dag_parse_finished'] == "true":
                        return True