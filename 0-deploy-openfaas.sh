# git clone openfaas kernel code from github and patch the code
base_dir=`pwd`
pip install -r requirements.txt
bash images_list.sh
# create ns openfaas
kubectl create ns openfaas
kubectl create ns openfaas-fn

mkdir helm
# install prometheus
cd $base_dir 
cd helm/
helm repo add prometheus-community
helm repo update
helm pull prometheus-community/kube-prometheus-stack --version 39.11.0
tar -zxf kube-prometheus-stack-39.11.0.tgz
cd kube-prometheus-stack
cp  $base_dir/resources/helm_values/prometheus-values.yaml ./values.yaml
helm install -n openfaas openfaas -f ./values.yaml


# install openfaas
cd $base_dir/helm/
helm repo add openfaas https://openfaas.github.io/faas-netes/helm/
helm repo update
helm pull openfaas/openfaas --version 10.2.8
tar -zxf openfaas-10.2.8.tgz
cd openfaas
cp  $base_dir/resources/helm_values/openfaas-values.yaml ./values.yaml
helm install -n openfaas openfaas -f ./values.yaml
export FAAS_PASSWORD=`kubectl -n openfaas get secret basic-auth -o jsonpath="{.data.basic-auth-password}" | base64 --decode`
# get host ip as the openfaas master ip
OPENFAAS_MASTER_IP=`kubectl get nodes -o wide | grep master | awk '{print $6}'`
cd resources
python3 update_inference.py $OPENFAAS_MASTER_IP
kubectl apply -f faas_template_cm/faas_template_cm.yaml

sed -i "s/192.168.0.102/$OPENFAAS_MASTER_IP/g" `grep '192.168.0.102' -rl $base_dir`
sed -i "s/sWsVc8uuyJPe/$FAAS_PASSWORD/g" `grep 'sWsVc8uuyJPe' -rl $base_dir/`