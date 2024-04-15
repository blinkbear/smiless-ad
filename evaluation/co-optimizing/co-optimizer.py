import os
import sys
SOURCEDIR = os.path.dirname(os.path.abspath(__file__))
EVALUATION_DIR = os.path.dirname(os.path.dirname(SOURCEDIR))
CONFIG_FILE = "config/config.json"
RESULT_DIR = "data/total_result"
FUNCTION_CONFIG_DIR = "config/"
TMP_DIR = "tmp"


def evaluation_co_optimizer():
    import subprocess

    save_dir = os.path.join(SOURCEDIR, "data", "result_co_optimizer.csv")
    exec_command = f"python3 {os.path.join(EVALUATION_DIR,'end_to_end','invoke_workflow.py')} --master_ip {master_ip} --slas '2,' --optimizer 'smiless,smiless-homo,smiless-no-dag' --save_dir {save_dir} --test_type initialization"
    result = subprocess.call(exec_command, shell=True)
    print(result)

master_ip = sys.argv[1]
evaluation_co_optimizer()



