import os
import sys

SOURCEDIR = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.dirname(os.path.dirname(SOURCEDIR))
CONFIG_FILE = "config/config.json"
FUNCTION_CONFIG_DIR = "config/"
TMP_DIR = "tmp"


def evaluation_bursty():
    import subprocess

    for optimizer in ["smiless", "icebreaker", "aquatope", "orion", "grandslam"]:
        save_dir = os.path.join(SOURCEDIR, "data", f"bursty_{optimizer}_result.csv")
        exec_command = f"python3 {os.path.join(EVALUATION_DIR,'end_to_end','invoke_workflow.py')} --master_ip {master_ip} --slas '2,' --optimizer '{optimizer}' --save_dir {save_dir} --test_type bursty"
        result = subprocess.call(exec_command, shell=True)
        print(result)


master_ip = sys.argv[1]
evaluation_bursty()
