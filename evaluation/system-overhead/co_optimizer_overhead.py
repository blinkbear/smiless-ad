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
    smiless_optimizer = OptimizerFactory().get_optimizer()
    workflow_lengths = [i for i in range(3, 11)]
    SLAs = [i for i in range(2, 10)]
    durations = []
    for workflow_length in workflow_lengths:
        max_duration = 0
        for SLA in SLAs:
            dag_parser.generate_workflow_dag(
                workflow_name, workflow_config, workflow_length
            )

            graph_df, graph_dfs, entry_node = dag_parser.parse_graph_to_df(
                workflow_name
            )
            _, _, execution_time = smiless_optimizer.get_workflow_running_plan_df(
                workflow_name,
                graph_df,
                graph_dfs,
                SLA,
            )
            
            if execution_time > max_duration:
                max_duration = execution_time

        durations.append(max_duration)
    result = pd.DataFrame({"workflow_length": workflow_lengths, "duration": durations})
    print(result)
    # result.to_csv(os.path.join(BASE_DIR, "data", "co_optimizer_overhead.csv"))


get_co_optimization_overhead()
