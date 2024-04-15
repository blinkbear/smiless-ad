# offline profiling result
pwd=`pwd`
current_ip=`hostname -I | awk '{print $1}'`
# first execute bursty invocations
python3 $pwd/evaluation/bursty_adaptation/bursty_workloads.py current_ip
# then analyze the result 
python3 $pwd/evaluation/bursty_adaptation/analysis.py
