import json
import networkx as nx


class ConfigParser(object):
    def __init__(self,config_path):
        self.config_path = config_path
        self.config = self.parse_config()

    def parse_config(self):
        with open(self.config_path) as f:
            config = json.load(f)
        return config


    def get_optimizers(self):
        return self.config["optimizers"]

    def get_interval_time_unit(self):
        return self.config["interval_time_unit"]
    def get_concurrency(self):
        return self.config["concurrency"]

    def get_repeat(self):
        return self.config["repeat"]

    def get_function_profiles(self):
        return self.config["function_profiles"]

    def set_function_unique(self, workflow, workflow_name):
        nodes = workflow["node"]
        edges = workflow["edges"]
        updated_nodes = []
        updated_edges = []
        entry_point = workflow["entryPoint"] + "-" + workflow_name
        for node in nodes:
            updated_nodes.append(node + "-" + workflow_name)
        for edge in edges:
            updated_edges.append(
                {
                    "source": edge["source"] + "-" + workflow_name,
                    "target": edge["target"] + "-" + workflow_name,
                    "weight": edge["weight"],
                }
            )
        workflow["node"] = updated_nodes
        workflow["edges"] = updated_edges
        workflow["entryPoint"] = entry_point
        return workflow

    def get_workflows(self):
        original_workflows = {
            "ambert-alert": self.config["workflows"]["ambert-alert"],
            "img-query": self.config["workflows"]["img-query"],
            "voice-assistant": self.config["workflows"]["voice-assistant"],
        }
        workflows = {}
        for workflow_name in original_workflows:
            workflow = original_workflows[workflow_name]
            workflow = self.set_function_unique(workflow, workflow_name)
            workflows[workflow_name] = workflow
        return workflows

    def get_slas(self):
        return self.config["SLAs"]
        
    def get_long_workflows(self):
        return {"no-share": self.config["workflows"]["no-share"]}

    def get_functions_info(self):
        return self.config["functions_info"]


    def get_openfaas_config(self):
        return (
            self.config["openfaas"]["url"],
            self.config["openfaas"]["gateway"],
            self.config["openfaas"]["passwd"],
        )
