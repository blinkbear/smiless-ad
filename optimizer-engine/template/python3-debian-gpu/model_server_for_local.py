import zerorpc
from function.models.model import Model
import os
# import function.models.model_pb2 as model_pb2
# import function.models.model_pb2_grpc as model_pb2_grpc
from concurrent import futures
import time
# import grpc
import datetime
import logging

log_file = 'app.log'
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelForCPU(object):
    def __init__(self, container_path, model_path, backend, use_model_controller):
        self.init_finished=False
        self.model = Model(container_path, model_path, backend, use_model_controller)
        self.model.to(backend)
        self.init_finished=True

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
        logging.info(f"start inference {datetime.datetime.now()}")
        while True:
            if self.init_finished:
                res= self.model.predict(return_result, kwargs)
                logging.info(f"end inference {datetime.datetime.now()}")
                return res

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
        # os.system("fwatchdog &")
        s.run()
    else:
        return


main()


# class ModelServicer(model_pb2_grpc.ModelServiceServicer):
#     def __init__(self, container_path, model_path, backend, use_model_controller):
#         self.model = Model(
#             container_path, model_path, backend, use_model_controller
#         )  # 创建模型对象
#         self.model.to(backend)

#     def Predict(self, request, context):
#         return_result = request.return_result
#         params = request.params
#         result = self.model.predict(return_result=return_result, params=params)
#         response = model_pb2.PredictResponse(
#             ret=result["ret"],
#             model_predict_time=float(result["model_predict_time"]),
#             model_load_time=float(result["model_load_time"]),
#             model_trans_time=float(result["model_trans_time"]),
#         )
#         return response


# def serve():
#     use_model_controller = os.environ.get("USE_MODEL_CONTROLLER", "False")
#     backend = os.environ.get("BACKEND", "cpu")
#     server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
#     model_pb2_grpc.add_ModelServiceServicer_to_server(
#         ModelServicer("", "", backend, use_model_controller), server
#     )
#     server.add_insecure_port("0.0.0.0:4242")
#     server.start()
#     print("Server started, listening on port 4242...")
#     server.wait_for_termination()



# class ModelServer(object):
#     def __init__(self, container_path, model_path, backend, use_model_controller):
#         self.model = Model(container_path, model_path, backend, use_model_controller)
#         self.model.to(backend)

#     def predict(self, return_result, params):
#         return_result = eval(return_result)
#         return self.model.predict(return_result, params)

# if __name__ == "__main__":
#     import datetime
#     st = datetime.datetime.now()
#     use_model_controller = os.environ.get("USE_MODEL_CONTROLLER", "False")
#     backend = os.environ.get("BACKEND", "cpu")
#     server = msgpackrpc.Server(ModelServer("", "", backend, use_model_controller))
#     server.listen(msgpackrpc.Address("localhost", 4242))
#     et = datetime.datetime.now()
#     logging.info(f"model server start time: {st}, end time{et}")
#     server.start()
