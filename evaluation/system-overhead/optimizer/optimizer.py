import abc


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def update_workflow_running_plan_df(self, workflow_name, SLA, interval_time_unit):
        pass

    @abc.abstractmethod
    def get_workflow_running_plan_df(
        self,
        workflow_name,
        graph_df,
        graph_dfs,
        SLA,
        interval_time_unit,
    ):
        pass

class OptimizerFactory:
    def __init__(self):
        self.factories = {}

    def get_optimizer(
        self,
    ):
        from .smiless import SMIless

        if "smiless" not in self.factories:
            smiless = SMIless()
            self.factories["smiless"] = smiless
        return self.factories["smiless"]
