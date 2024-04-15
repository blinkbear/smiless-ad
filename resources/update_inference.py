import sys
import yaml
import base64

def update_inference_yaml(openfaas_master_ip):
    with open("inference.yml", "r") as stream:
        try:
            data = yaml.safe_load(stream)
            data["provider"]["gateway"] = openfaas_master_ip
            for i in data["functions"]:
                print(i)
                data['functions'][i]["environment"]["BASIC_URL"] = (
                    f"http://{openfaas_master_ip}:31112/function"
                )

            with open("inference.yml", "w") as outfile:
                yaml.dump(data, outfile, default_flow_style=False)
            inference_base64 = base64.b64encode(open("inference.yml", "rb").read()).decode("utf-8")
            with open("faas_template_cm/faas_template_cm.yaml", "r") as stream:
                try:
                    data = yaml.safe_load(stream)
                    data["data"]["faas_template"] = inference_base64
                    with open("faas_template_cm/faas_template_cm.yaml", "w") as outfile:
                        yaml.dump(data, outfile, default_flow_style=False)
                except yaml.YAMLError as exc:
                    print(exc)
            
            
        except yaml.YAMLError as exc:
            print(exc)

if __name__ == "__main__":
    openfaas_master_ip = sys.argv[1]
    update_inference_yaml(openfaas_master_ip) 