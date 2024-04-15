import json
import networkx as nx


class ConfigParser(object):
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.parse_config()

    def parse_config(self):
        with open(self.config_file) as f:
            config = json.load(f)
        return config

    def get_test_config(self):
        test_config = self.config["test_config"]
        test_config_args = []
        for key in test_config:
            test_config_args.append(
                (
                    key,
                    test_config[key]["backend"],
                    test_config[key]["namespace"],
                    test_config[key]["test_type"],
                    test_config[key]["server"],
                    eval(test_config[key]["save_result"]),
                    test_config[key]["skip_functions"],
                    test_config[key]["rerun"],
                )
            )
        return test_config_args

    def get_alternative_warm_or_cold(self):
        alternative_warm_cold = {}
        for func in self.config["warm_or_cold"]:
            alternative_warm_cold[func] = self.config["warm_or_cold"][func]
        return alternative_warm_cold

    def get_alternative_batch_size(self):
        alternative_batch_size = {}
        for func in self.config["batch_size"]:
            batch_size_range = self.config["batch_size"][func]
            alternative_batch_size[func] = [
                2**i
                for i in range(
                    batch_size_range["min"],
                    batch_size_range["max"],
                    batch_size_range["step"],
                )
            ]
        return alternative_batch_size

    def get_alternative_resource_quantity(self):
        alter_resource_quantity = {}
        for func in self.config["resource_quantity"]:
            resource_type = self.config["resource_quantity"][func].keys()  # cpu, cuda
            if func not in alter_resource_quantity:
                alter_resource_quantity[func] = {}
                for p in resource_type:
                    alter_resource_quantity[func][p] = []
            for p in resource_type:
                resource_quantity_range = self.config["resource_quantity"][func][p]
                if p == "cuda":
                    for v in range(
                        resource_quantity_range["min"],
                        resource_quantity_range["max"],
                        resource_quantity_range["step"],
                    ):
                        alter_resource_quantity[func][p].append(v)
                if p == "cpu":
                    resource_quantities = [
                        2**i
                        for i in range(
                            resource_quantity_range["min"],
                            resource_quantity_range["max"],
                            resource_quantity_range["step"],
                        )
                    ]
                    alter_resource_quantity[func][p] = resource_quantities

        return alter_resource_quantity

    def get_default_params(self):
        cpu = self.config["default_params"]["cpu"]
        memory = self.config["default_params"]["memory"]
        gpu = self.config["default_params"]["cuda"]
        input_size = self.config["default_params"]["input_size"]
        warm_cold = self.config["default_params"]["warm_cold"]
        batch_size = self.config["default_params"]["batch_size"]
        invoking_type = self.config["default_params"]["invoking_type"]
        max_batch_size = self.config["default_params"]["max_batch_size"]
        return (
            cpu,
            memory,
            gpu,
            input_size,
            warm_cold,
            batch_size,
            invoking_type,
            max_batch_size,
        )

    def get_optimizers(self):
        return self.config["optimizers"]

    def get_repeat(self):
        return self.config["repeat"]

    def get_function_profiles(self):
        return self.config["function_profiles"]

    def get_test(self):
        return self.config["test"]

    def get_openfaas_config(self):
        return (
            self.config["openfaas"]["url"],
            self.config["openfaas"]["gateway"],
            self.config["openfaas"]["passwd"],
        )
