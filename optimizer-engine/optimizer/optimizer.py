import abc

from offline_profiler.function_resource_profiler import FunctionProfiler
from cache.cache import Cache
from cache.invocation_infos import InvocationInfos
from online_predictor.online_predictor import OnlinePredictor


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def update_workflow_running_plan_df(
        self,
        workflow_name,
        IT,
        SLA,
        interval_time_unit,
    ):
        pass

    @abc.abstractmethod
    def get_workflow_running_plan_df(
        self,
        workflow_name,
        graph_df,
        graph_dfs,
        IT,
        SLA,
        interval_time_unit,
    ):
        pass

    @abc.abstractmethod
    def remove_workflow(self, workflow_name):
        pass


class OptimizerFactory:
    def __init__(self):
        self.factories = {}

    def get_optimizer(
        self,
        name,
        default_device,
        function_profiler: FunctionProfiler,
        cache: Cache,
        invocation_infos: InvocationInfos,
        online_predictor: OnlinePredictor,
    ):
        from .orion import Orion
        from .icebreaker import IceBreaker
        from .aquatope import Aquatope
        from .grandslam import GrandSLAm
        from .smiless import SMIless
        from .smiless_homo import SMIlessHomo
        from .smiless_no_dag import SMIlessNoDag
        from .smiless_opt import SMIlessOPT

        if name == "orion":
            if "orion" not in self.factories:
                orion = Orion(default_device, function_profiler)
                self.factories["orion"] = orion
            return self.factories["orion"]
        if name == "icebreaker":
            if "icebreaker" not in self.factories:
                icebreaker = IceBreaker(invocation_infos, function_profiler)
                self.factories["icebreaker"] = icebreaker
            return self.factories["icebreaker"]
        if name == "aquatope":
            if "aquatope" not in self.factories:
                aquatope = Aquatope(function_profiler)
                self.factories["aquatope"] = aquatope
            return self.factories["aquatope"]
        if name == "grandslam":
            if "grandslam" not in self.factories:
                grandslam = GrandSLAm(default_device, function_profiler)
                self.factories["grandslam"] = grandslam
            return self.factories["grandslam"]
        if name == "smiless" or name == "smiless-bursty":
            if (
                "smiless" not in self.factories
                or "smiless-bursty" not in self.factories
            ):
                smiless = SMIless(cache, function_profiler, online_predictor)
                self.factories["smiless"] = smiless
            return self.factories["smiless"]
        if name == "smiless-homo":
            if "smiless-homo" not in self.factories:
                smiless_homo = SMIlessHomo(cache, function_profiler, online_predictor)
                self.factories["smiless-homo"] = smiless_homo
            return self.factories["smiless-homo"]
        if name == "smiless-no-dag":
            if "smiless-no-dag" not in self.factories:
                smiless_no_dag = SMIlessNoDag(
                    cache, function_profiler, online_predictor
                )
                self.factories["smiless-no-dag"] = smiless_no_dag
            return self.factories["smiless-no-dag"]
        if name == "smiless-opt":
            if "smiless-opt" not in self.factories:
                smiless_opt = SMIlessOPT(cache, function_profiler, online_predictor)
                self.factories["smiless-opt"] = smiless_opt
            return self.factories["smiless-opt"]
