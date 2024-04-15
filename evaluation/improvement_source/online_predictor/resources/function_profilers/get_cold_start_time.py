import pandas as pd

def get_cs_extra_time_profiler(result):
    for col in result.columns:
            if "time" in col:
                result.loc[result.index, col] = result[col] / 1000000
    function_names = result["function_name"].unique()
    function_cs_extra_times = {}
    function_running_time_cuda_warm_cold = result.query(
        "test_type == 'warm_cold' and backend == 'cuda'"
    )
    function_running_time_cpu_warm_cold = result.query(
        "test_type == 'warm_cold' and backend == 'cpu'"
    )
    for function_name in function_names:
        function_cs_extra_times[function_name] = {"cpu": 0, "gpu": 0}
        gpu_cs_running_time = function_running_time_cuda_warm_cold.query(
            "function_name == @function_name and warm_cold == 'cold'"
        )["running_time"].max()
        gpu_warm_running_time = function_running_time_cuda_warm_cold.query(
            "function_name == @function_name and warm_cold == 'warm'"
        )["running_time"].max()
        function_cs_extra_times[function_name]["gpu"] = max(
            0, (gpu_cs_running_time - gpu_warm_running_time)
        )
        cpu_cs_time = function_running_time_cpu_warm_cold.query(
            "function_name == @function_name and warm_cold == 'cold'"
        )["running_time"]
        cpu_warm_time = function_running_time_cpu_warm_cold.query(
            "function_name == @function_name and warm_cold == 'warm'"
        )["running_time"]
        cpu_cs_running_time = cpu_cs_time.max()
        cpu_warm_running_time = cpu_warm_time.max()
        cpu_cs_extra_time = cpu_cs_running_time - (
            cpu_warm_running_time
            if cpu_cs_running_time > cpu_warm_running_time
            else cpu_cs_running_time
        )
        function_cs_extra_times[function_name]["cpu"] = cpu_cs_extra_time
    import json
    with open("cs_extra_times.json", "w") as f:
        json.dump({"cs_time": json.dumps(function_cs_extra_times)}, f)
    # print(function_cs_extra_times)
    # return function_cs_extra_times



df = pd.read_csv( "result.csv")
get_cs_extra_time_profiler(df)