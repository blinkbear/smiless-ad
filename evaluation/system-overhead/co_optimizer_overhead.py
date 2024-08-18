from optimizer.optimizer import OptimizerFactory
from optimizer.dag_parser import DAGParser
import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_co_optimization_overhead():
    with open("workflow_config.json", "r") as f:
        workflow_config = json.load(f)["workflows"]["long-workflow"]

    workflow_name = ""
    dag_parser = DAGParser()
    optimizers = OptimizerFactory().get_optimizer()
    workflow_lengths = [i for i in range(3, 15)]
    SLAs = [i for i in range(2,7)]
    durations = []
    total_workflow_lengths = []
    optimizer_names = []
    for optimizer_name in optimizers:
        optimizer = optimizers[optimizer_name]
        for workflow_length in workflow_lengths:
            max_duration = 0
            for SLA in SLAs:
                dag_parser.generate_workflow_dag(
                    workflow_name, workflow_config, workflow_length
                )

                graph_df, graph_dfs, entry_node = dag_parser.parse_graph_to_df(
                    workflow_name
                )
                result_df, _, execution_time = optimizer.get_workflow_running_plan_df(
                    workflow_name,
                    graph_df,
                    graph_dfs,
                    SLA,
                )
                if execution_time > max_duration:
                    max_duration = execution_time
            total_workflow_lengths.append(workflow_length)
            optimizer_names.append(optimizer_name)
            durations.append(max_duration)
    result = pd.DataFrame({"optimizer_names": optimizer_names, "workflow_length": total_workflow_lengths, "duration": durations})
    result.to_csv(os.path.join(BASE_DIR, "data", "co_optimizer_overhead.csv"),index=False)


get_co_optimization_overhead()
