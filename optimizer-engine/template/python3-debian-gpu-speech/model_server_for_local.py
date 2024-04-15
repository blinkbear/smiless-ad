import zerorpc
from function.models.model import Model
import os


class ModelForCPU(object):
    def __init__(self, container_path, model_path, backend, use_model_controller):
        self.model = Model(container_path, model_path, backend, use_model_controller)
        self.model.to(backend)

    def inference(
        self,
        pod_name,
        backend,
        use_model_controller,
        model_load_path,
        model_path,
        return_result,
        kwargs,
    ):
        return self.model.predict(return_result, kwargs)


def main():
    use_model_controller = os.environ.get("USE_MODEL_CONTROLLER", "False")
    backend = os.environ.get("BACKEND", "cpu")
    gpu_quantity = os.environ.get("gpu_quantity", 10)
    os.environ.setdefault("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", gpu_quantity)
    if use_model_controller == "False":
        s = zerorpc.Server(
            ModelForCPU("", "", backend, use_model_controller), heartbeat=None
        )
        print("start bind")
        s.bind("tcp://0.0.0.0:4242")
        s.run()
    else:
        return


main()
