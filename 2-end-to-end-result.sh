pwd=`pwd`
current_ip=`hostname -I | awk '{print $1}'`
# first execute the workflow in parallel
python3 $pwd/evaluation/end-to-end/invoke_workflow.py --master_ip=$current_ip
# then analyze the result 
python3 $pwd/evaluation/end-to-end/analysis.py