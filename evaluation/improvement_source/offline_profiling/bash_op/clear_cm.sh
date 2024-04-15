kubectl delete cm -n openfaas-fn `kubectl get cm -n openfaas-fn| awk '$3 ~ /s|m|h/ {print $1}'`
faas-cli remove -f inference.yml