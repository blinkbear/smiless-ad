from turtle import back
import zerorpc
from function.models.model import Model
import os

class ModelForCPU(object):
    def __init__(self, container_path, model_path, backend):
        self.model = Model(container_path, model_path, backend)

    def inference(
        self,
        pod_name,
        backend,
        model_load_path,
        model_path,
        return_result,
        **kwargs,
    ):
        return self.model.predict(return_result, **kwargs)


def main():
    backend = os.environ.get("BACKEND", "cpu")
    if backend == "cpu":
        s = zerorpc.Server(ModelForCPU("","",backend), heartbeat=None)
        print("start bind")
        s.bind("tcp://0.0.0.0:4242")
        s.run()
    else:
        return


main()
