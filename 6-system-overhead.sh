# offline profiling result
pwd=`pwd`
current_ip=`hostname -I | awk '{print $1}'`
# first execute bursty invocations
python3 $pwd/evaluation/system_overhead/auto_scaler_overhead.py
python3 $pwd/evaluation/system_overhead/co_optimizer_overhead.py

# then analyze the result
python3 $pwd/evaluation/system_overhead/analysis.py