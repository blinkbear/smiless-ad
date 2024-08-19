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
        from .smiless import SMIlessBFS
        from .smiless import SMIlessDFS
        from .smiless import SMIlessAstar
        from .smiless import SMIlessAug

        if "smiless" not in self.factories:
            smiless = SMIless()
            self.factories["smiless"] = smiless
        if "smiless-aug" not in self.factories:
            self.factories["smiless-aug"] = SMIlessAug(
            )
        if "smiless-bfs" not in self.factories:
                self.factories["smiless-bfs"] = SMIlessBFS(
                )
        if "smiless-dfs" not in self.factories:
            self.factories["smiless-dfs"] = SMIlessDFS(
            )
        if "smiless-astar" not in self.factories:
            self.factories["smiless-astar"] = SMIlessAstar(
            )
        
        return self.factories
