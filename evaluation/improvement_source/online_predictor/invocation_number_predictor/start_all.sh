pwd=`pwd`
python3 $pwd/arima_predictor.py &
python3 $pwd/xgboost_predictor.py &
python3 $pwd/fip_predictor.py &
python3 $pwd/lstm_predictor.py &

wait

