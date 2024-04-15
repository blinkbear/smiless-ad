time=$(date "+%Y-%m-%d %H:%M:%S").$((`date "+%N"`/1000))
echo $time
python3 model_server_for_local.py &
fwatchdog