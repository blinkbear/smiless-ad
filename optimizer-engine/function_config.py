from  typing import Dict
class FunctionConfig:
    def __init__(
        self,
        function_name,
        image_size,
        depth:Dict[str,int],
        gpu_running_time,
        cold_start_time,
        gpu_trans_time,
        gpu_cs_extra_time,
        cpu_cs_extra_time,
        types,
        sharing,
        keep_alive_resource:Dict[str,int],
        cpu_running_time,
        keep_alive_time:Dict[str,str],
        device:Dict[str,str],
        prewarm_window:Dict[str,int],
        target_replicas,
        # image
    ):
        self.function_name = function_name
        self.image_size = image_size
        self.depth = depth
        self.gpu_running_time = gpu_running_time
        self.cold_start_time = cold_start_time
        self.gpu_trans_time = gpu_trans_time
        self.gpu_cs_extra_time = gpu_cs_extra_time
        self.cpu_cs_extra_time = cpu_cs_extra_time
        self.types = types
        self.keep_alive_resource = keep_alive_resource
        self.cpu_running_time = cpu_running_time
        self.keep_alive_time = keep_alive_time
        self.device = device
        self.prewarm_window = prewarm_window
        # self.image= image
        self.sharing=sharing
        self.target_replicas = target_replicas 