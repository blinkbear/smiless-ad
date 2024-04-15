from auto_scaler.auto_scaler import AutoScaler
import random
import time
import pandas as pd
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURE_DIR = os.path.dirname(BASE_DIR)


def get_scaler_overhead():
    auto_scaler = AutoScaler()
    mock_invocation_numbers = random.sample(range(1, 1000), 100)
    autoscaler_times = []
    function_name = "imagerecognition"
    for invocation_number in mock_invocation_numbers:
        start = time.time()
        auto_scaler.get_container_number(invocation_number, function_name, "cpu")
        end = time.time()
        autoscaler_times.append(end - start)
    df = pd.DataFrame(
        {
            "invocation_number": mock_invocation_numbers,
            "autoscaler_time": autoscaler_times,
        }
    )
    df.sort_values(by="invocation_number", inplace=True)
    df.to_csv(os.path.join(BASE_DIR, "data", "auto_scaler_overhead.csv"))
    return df


if __name__ == "__main__":
    get_scaler_overhead()
