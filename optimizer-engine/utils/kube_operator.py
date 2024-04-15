import json
from kubernetes import client
from kubernetes import config as kconf
import os
import yaml
from time import sleep


class KubeOperator:
    def __init__(self, api, namespace):
        kconf.load_config()
        self.api = api
        self.namespace = namespace

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
            self.api.patch_namespaced_config_map(
                configmap_name, self.namespace, configmap
            )
        else:
            self.api.create_namespaced_config_map(self.namespace, configmap)

    def get_configmap(self, configmap_name):
        configmap = self.api.read_namespaced_config_map(configmap_name, self.namespace)
        if configmap:
            return configmap
        else:
            return None