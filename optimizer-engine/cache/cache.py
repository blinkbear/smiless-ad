from typing import Dict, List
from function_config import FunctionConfig
from readerwriterlock import rwlock
from offline_profiler.function_resource_profiler import FunctionProfiler

config_marker = rwlock.RWLockFair()
config_read_marker = config_marker.gen_rlock()
config_write_marker = config_marker.gen_wlock()
scaled_marker = rwlock.RWLockFair()
scaled_object_read_marker = scaled_marker.gen_rlock()
scaled_object_write_marker = scaled_marker.gen_wlock()


class Cache:
    def __init__(self, function_profiler: FunctionProfiler):
        self.func_config_cache: Dict[str, FunctionConfig] = {}
        self.function_profiler = function_profiler
        self.nodes = {}
        self.nodes_to_workflow_name = {}
        self.workflows_entries = {}
        self.workflow_nodes = {}
        self.workflow_sla = {}

    def get_func_config(self, func_name: str) -> FunctionConfig:
        config_read_marker.acquire()
        if func_name in self.func_config_cache:
            func_config = self.func_config_cache[func_name]
        else:
            func_config = None
        config_read_marker.release()
        return func_config  # type: ignore

    def get_workflow_sla(self, workflow_name):
        config_read_marker.acquire()
        sla = self.workflow_sla[workflow_name]
        config_read_marker.release()
        return sla

    def get_workflow_entries(self, workflow_name):
        config_read_marker.acquire()
        entry_node = self.workflows_entries[workflow_name]
        config_read_marker.release()
        return entry_node

    def get_shared_nodes(self, nodes):
        shared_nodes = []
        config_write_marker.acquire()
        shared_nodes = {node: self.nodes[node] for node in nodes if node in self.nodes}
        config_write_marker.release()
        return shared_nodes

    def update_func_config(self, func_name: str, func_config: FunctionConfig):
        config_write_marker.acquire()
        self.func_config_cache[func_name] = func_config
        config_write_marker.release()

    def add_func_config(
        self, func_name: str, func_config: FunctionConfig, workflow_name: str
    ):
        old_func_config = self.get_func_config(func_name)
        if old_func_config is not None:
            old_func_config, update_or_not = self.simple_merge_func_config(
                old_func_config, func_config
            )
            qps = self.nodes[func_name][0]
            self.nodes[func_name] = (qps + 1, old_func_config.device["merge_device"])
            self.update_func_config(func_name, old_func_config)
            update_or_not = True
        else:
            func_config.depth["merge_depth"] = func_config.depth[workflow_name]
            func_config.keep_alive_resource["merge_keep_alive_resource"] = (
                func_config.keep_alive_resource[workflow_name]
            )
            func_config.keep_alive_time["merge_keep_alive_time"] = (
                func_config.keep_alive_time[workflow_name]
            )

            func_config.device["merge_device"] = func_config.device[workflow_name]
            func_config.prewarm_window["merge_prewarm_window"] = (
                func_config.prewarm_window[workflow_name]
            )
            func_config.sharing = False
            self.nodes[func_name] = (1, func_config.device["merge_device"])
            self.update_func_config(func_name, func_config)
            update_or_not = True
        return update_or_not

    def remove_func_config(self, func_name: str):
        config_write_marker.acquire()
        if func_name in self.func_config_cache:
            del self.func_config_cache[func_name]
        config_write_marker.release()

    def delete_function(self, func_name: str, workflow_name: str):
        updated_func_config = None
        update_or_not = False
        if workflow_name in self.workflows_entries:
            self.workflows_entries.pop(workflow_name)
        if func_name in self.func_config_cache:
            func_config = self.get_func_config(func_name)

            updated_func_config, update_or_not = self.remove_merge_func_config(
                func_config, workflow_name
            )
            if func_name in self.nodes:
                qps = self.nodes[func_name][0]
                if qps - 1 == 1 or updated_func_config is None:
                    self.nodes.pop(func_name)
                    self.func_config_cache.pop(func_name)
                else:
                    self.nodes[func_name] = (
                        qps - 1,
                        updated_func_config.device["merge_device"],
                    )
                if func_name in self.nodes_to_workflow_name:
                    left_workflows = self.nodes_to_workflow_name[func_name].remove(
                        workflow_name
                    )
                    if left_workflows is None or len(left_workflows) == 0:
                        self.nodes_to_workflow_name.pop(func_name)
                    else:
                        self.nodes_to_workflow_name[func_name] = left_workflows

        return updated_func_config, update_or_not

    def simple_merge_func_config(
        self,
        old_func_config: FunctionConfig,
        func_config: FunctionConfig,
    ):
        """
        Merge two FunctionConfig objects and return the merged FunctionConfig and
        a boolean indicating whether the new FunctionConfig has been updated or
        not.

        Args:
            old_func_config (FunctionConfig): The old FunctionConfig object.
            func_config (FunctionConfig): The new FunctionConfig object to be
                merged with the old one.

        Returns:
            Tuple[FunctionConfig, bool]: A tuple containing the merged
            FunctionConfig object and a boolean indicating whether the new
            FunctionConfig has been updated or not.
        """
        update_or_not = False

        old_depth = old_func_config.depth
        new_depth = func_config.depth
        # new_depth.update(old_depth)
        new_depth = self.merge_two_maps(old_depth, new_depth)
        depths = list(new_depth.values())
        new_depth["merge_depth"] = min(depths)
        func_config.depth = new_depth
        if old_depth["merge_depth"] != new_depth["merge_depth"]:
            update_or_not = True

        old_keep_alive_resource = old_func_config.keep_alive_resource
        new_keep_alive_resource = func_config.keep_alive_resource
        # new_keep_alive_resource.update(old_keep_alive_resource)
        new_keep_alive_resource = self.merge_two_maps(
            old_keep_alive_resource, new_keep_alive_resource
        )
        keep_alive_resources = list(new_keep_alive_resource.values())
        keep_alive_resources = [x for x in keep_alive_resources if x is not None]
        if len(keep_alive_resources) == 0:
            keep_alive_resources.append(1)
        new_keep_alive_resource["merge_keep_alive_resource"] = max(keep_alive_resources)
        func_config.keep_alive_resource = new_keep_alive_resource
        if (
            old_keep_alive_resource["merge_keep_alive_resource"]
            != new_keep_alive_resource["merge_keep_alive_resource"]
        ):
            update_or_not = True

        old_device = old_func_config.device
        new_device = func_config.device
        # new_device.update(old_device)
        new_device = self.merge_two_maps(old_device, new_device)
        devices = list(new_device.values())
        if "cuda" in devices:
            new_device["merge_device"] = "cuda"
        else:
            new_device["merge_device"] = "cpu"
        func_config.device = new_device
        if old_device["merge_device"] != new_device["merge_device"]:
            update_or_not = True

        old_prewarm_window = old_func_config.prewarm_window
        new_prewarm_window = func_config.prewarm_window
        # new_prewarm_window.update(old_prewarm_window)
        new_prewarm_window = self.merge_two_maps(old_prewarm_window, new_prewarm_window)
        prewarm_windows = list(new_prewarm_window.values())
        new_prewarm_window["merge_prewarm_window"] = min(prewarm_windows)
        func_config.prewarm_window = new_prewarm_window
        if (
            old_prewarm_window["merge_prewarm_window"]
            != new_prewarm_window["merge_prewarm_window"]
        ):
            update_or_not = True

        new_sharing = True
        func_config.sharing = new_sharing

        old_keep_alive_time = old_func_config.keep_alive_time
        new_keep_alive_time = func_config.keep_alive_time
        new_keep_alive_time = self.merge_two_maps(
            old_keep_alive_time, new_keep_alive_time
        )
        keep_alive_times = list(new_keep_alive_time.values())

        new_keep_alive_time["merge_keep_alive_time"] = max(keep_alive_times)
        func_config.keep_alive_time = new_keep_alive_time
        if (
            old_keep_alive_time["merge_keep_alive_time"]
            != new_keep_alive_time["merge_keep_alive_time"]
        ):
            update_or_not = True

        return func_config, update_or_not

    def merge_two_maps(self, old_map: dict, new_map: dict):
        for k, v in new_map.items():
            old_map[k] = v
        return old_map

    def remove_merge_func_config(self, func_config: FunctionConfig, workflow_name: str):
        update_or_not = False
        old_depth = func_config.depth
        del old_depth[workflow_name]
        old_merge_depth = old_depth["merge_depth"]
        del old_depth["merge_depth"]
        if len(old_depth) == 0:
            return None, update_or_not
        old_depth["merge_depth"] = min(list(old_depth.values()))
        func_config.depth = old_depth
        if old_merge_depth != old_depth["merge_depth"]:
            update_or_not = True

        old_keep_alive_resource = func_config.keep_alive_resource
        del old_keep_alive_resource[workflow_name]
        old_merge_keep_alive_resource = old_keep_alive_resource[
            "merge_keep_alive_resource"
        ]
        del old_keep_alive_resource["merge_keep_alive_resource"]
        old_keep_alive_resources = [
            x for x in old_keep_alive_resource.values() if x is not None
        ]
        if len(old_keep_alive_resources) == 0:
            old_keep_alive_resource = [1]
        old_keep_alive_resource["merge_keep_alive_resource"] = max(  # type: ignore
            old_keep_alive_resources
        )
        func_config.keep_alive_resource = old_keep_alive_resource  # type: ignore
        if (
            old_merge_keep_alive_resource
            != old_keep_alive_resource["merge_keep_alive_resource"]
        ):  # type: ignore
            update_or_not = True

        old_device = func_config.device
        del old_device[workflow_name]
        old_merge_device = old_device["merge_device"]
        del old_device["merge_device"]
        if "cuda" in list(old_device.values()):
            old_device["merge_device"] = "cuda"
        else:
            old_device["merge_device"] = "cpu"
        func_config.device = old_device
        if old_merge_device != old_device["merge_device"]:
            update_or_not = True

        old_prewarm_window = func_config.prewarm_window
        del old_prewarm_window[workflow_name]
        old_merge_prewarm_window = old_prewarm_window["merge_prewarm_window"]
        del old_prewarm_window["merge_prewarm_window"]
        old_prewarm_window["merge_prewarm_window"] = min(
            list(old_prewarm_window.values())
        )
        func_config.prewarm_window = old_prewarm_window
        if old_merge_prewarm_window != old_prewarm_window["merge_prewarm_window"]:
            update_or_not = True
        if len(old_prewarm_window) == 2:
            func_config.sharing = False
        old_keep_alive_time = func_config.keep_alive_time
        del old_keep_alive_time[workflow_name]
        old_merge_keep_alive_time = old_keep_alive_time["merge_keep_alive_time"]
        del old_keep_alive_time["merge_keep_alive_time"]
        # if not func_config.sharing:
        old_keep_alive_time["merge_keep_alive_time"] = max(
            list(old_keep_alive_time.values())
        )
        if old_merge_keep_alive_time != old_keep_alive_time["merge_keep_alive_time"]:
            update_or_not = True
        func_config.keep_alive_time = old_keep_alive_time

        return func_config, update_or_not

    def cache_function_result(self, result_df, workflow_name, SLA):
        updated_functions_list = []
        function_list = []
        entry_node = result_df.loc[0, "node"]
        config_write_marker.acquire()
        self.workflows_entries[workflow_name] = entry_node
        config_write_marker.release()
        if "target_replicas" not in result_df:
            result_df["target_replicas"] = 1
        for index, row in result_df.iterrows():
            func_config = FunctionConfig(
                row["node"],
                row["image_size"],
                {workflow_name: int(row["depth"])},
                row["gpu_running_time"],
                row["cold_start_time"],
                row["gpu_trans_time"],
                row["gpu_cs_extra_time"],
                row["cpu_cs_extra_time"],
                row["types"],
                False,
                {workflow_name: row["keep_alive_resource"]},
                row["cpu_running_time"],
                {workflow_name: row["keep_alive_time"]},
                {workflow_name: row["device"]},
                {workflow_name: row["prewarm_window"]},
                {workflow_name: row["target_replicas"]},
                # node_images[row["types"]],
            )
            if row["node"] not in self.nodes_to_workflow_name:
                config_write_marker.acquire()
                self.nodes_to_workflow_name[row["node"]] = [workflow_name]
                config_write_marker.release()
            else:
                config_write_marker.acquire()
                self.nodes_to_workflow_name[row["node"]].append(workflow_name)
                config_write_marker.release()
            update_or_not = self.add_func_config(
                row["node"], func_config, workflow_name
            )
            function_list.append(func_config)
            if update_or_not:
                updated_functions_list.append(row["node"])
        config_write_marker.acquire()
        self.workflow_nodes[workflow_name] = function_list
        self.workflow_sla[workflow_name] = SLA
        config_write_marker.release()
        return updated_functions_list

    def get_workflow_function_configs(self, workflow_name) -> List[FunctionConfig]:
        return self.workflow_nodes[workflow_name]

    def delete_workflow_functions(self, workflow_name):
        config_write_marker.acquire()
        del self.workflow_nodes[workflow_name]
        config_write_marker.release()
