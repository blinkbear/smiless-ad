# offline profiling result
pwd=`pwd`
current_ip=`hostname -I | awk '{print $1}'`
# first execute the workflow in parallel
python3 $pwd/evaluation/improvement_source/offline_profiling/function_profiling.py
# then analyze the result 
python3 $pwd/evaluation/improvement_source/offline_profiling/function_profiling.py


# online predictor
## inter-arrival time predictor
bash $pwd/evaluation/improvement_source/online_predictor/inter_arrival_time_predictor/start_all.sh
python3 $pwd/evaluation/improvement_source/online_predictor/inter_arrival_time_predictor.py

## invocation number predictor

bash $pwd/evaluation/improvement_source/online_predictor/invocation_number_predictor/start_all.sh
python3 $pwd/evaluation/improvement_source/online_predictor/invocation_number_predictor.py
