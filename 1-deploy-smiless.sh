# deploy optimizer engine
pwd=`pwd`
cd optimizer-engine
kubectl apply -f k8s_yaml/
cd $pwd
