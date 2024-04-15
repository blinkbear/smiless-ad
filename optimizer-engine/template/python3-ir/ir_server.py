import datetime
from PIL import Image
from torchvision import transforms
import json
from function.suppress_stdout_stderr import suppress_stdout_stderr
import os
import zerorpc
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class Model(object):
    def __init__(self):
        SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        model = None
        model_path = os.path.join(SCRIPT_DIR, "models", "imagerecognition.pth")
        class_idx = json.load(
            open(os.path.join(SCRIPT_DIR, "function", "imagenet_class_index.json"), "r")
        )
        self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        backend = os.environ.get("BACKEND", "cpu")
        if backend == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        load_begin = datetime.datetime.now()
        model = torch.load(model_path)
        model.eval()
        load_end = datetime.datetime.now()
        trans_begin = datetime.datetime.now()
        model.to(self.device)
        trans_end = datetime.datetime.now()
        self.model = model
        self.local = os.environ.get("LOCAL", "true")
        if self.local == "true":
            self.image_path = os.path.join(SCRIPT_DIR, "fake-resnet")
            self.batch_size = os.environ.get("BATCH_SIZE", 1)
            self.image_list = self.loadImageFromLocal(self.image_path, self.batch_size)
        self.model_load_time = (load_end - load_begin) / datetime.timedelta(
            microseconds=1
        )
        self.model_trans_time = (trans_end - trans_begin) / datetime.timedelta(
            microseconds=1
        )

    def loadImageFromLocal(self, image_path, batch_size):
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image_path_list = [
            os.path.join(image_path, f)
            for f in os.listdir(image_path)
            if f.endswith(".jpg") or f.endswith(".JPG")
        ]
        image_path_list = image_path_list[:batch_size]
        # with suppress_stdout_stderr():
        image_list = []
        for ids in range(0, len(image_path_list) // batch_size):
            img_list = []
            for img_path in image_path_list[ids * batch_size : (ids + 1) * batch_size]:
                if not os.path.exists(img_path):
                    print(f"file: '{img_path}' dose not exist.")
                img = Image.open(img_path)
                img = preprocess(img)
                img_list.append(img)
            image_list.append(img_list)
        return image_list

    def loadOneImageFromLocal(self):
        image_path_list = [
            os.path.join(self.image_path, f)
            for f in os.listdir(self.image_path)
            if f.endswith(".jpg") or f.endswith(".JPG")
        ]
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        img = Image.open(image_path_list[0])
        img = preprocess(img)
        return img

    def predictLoad(self, input_length):
        predict_start = datetime.datetime.now()
        total_img = []
        with torch.no_grad():
            for i in range(input_length):
                total_img.append(self.loadOneImageFromLocal())
            batch_img = torch.stack(total_img, dim=0).to(self.device)
            output = self.model(batch_img).cpu()
            probs, classes = torch.max(output, dim=1)
        predict_end = datetime.datetime.now()
        predict_time = (predict_end - predict_start) / datetime.timedelta(
            microseconds=1
        )
        result = {
            "predict_time": predict_time,
            "model_load_time": self.model_load_time,
            "model_trans_time": self.model_trans_time,
        }
        return result

    def predictLocal(self, input_length):
        predict_start = datetime.datetime.now()
        total_img = []
        with torch.no_grad():
            for img_list in self.image_list:
                total_img += img_list * input_length
            batch_img = torch.stack(total_img, dim=0).to(self.device)
            output = self.model(batch_img).cpu()
            probs, classes = torch.max(output, dim=1)
        predict_end = datetime.datetime.now()
        predict_time = (predict_end - predict_start) / datetime.timedelta(
            microseconds=1
        )
        result = {
            "predict_time": predict_time,
            "model_load_time": self.model_load_time,
            "model_trans_time": self.model_trans_time,
        }
        return result

    def predictRandom(self, input_length):
        predict_start = datetime.datetime.now()
        with torch.no_grad():
            batch_img = torch.rand(1, 3, 1 * input_length, 1 * input_length).to(
                self.device
            )
            output = self.model(batch_img).cpu()
            probs, classes = torch.max(output, dim=1)
        predict_end = datetime.datetime.now()
        predict_time = (predict_end - predict_start) / datetime.timedelta(
            microseconds=1
        )
        result = {
            "predict_time": predict_time,
            "model_load_time": self.model_load_time,
            "model_trans_time": self.model_trans_time,
        }
        return result

    def predict(self, image_path_list, batch_size):
        predict_start = datetime.datetime.now()
        result = {"idx": [], "ret": []}
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # with suppress_stdout_stderr():
        with torch.no_grad():
            for ids in range(0, len(image_path_list) // batch_size):
                img_list = []
                for img_path in image_path_list[
                    ids * batch_size : (ids + 1) * batch_size
                ]:
                    if not os.path.exists(img_path):
                        print(f"file: '{img_path}' dose not exist.")
                    img = Image.open(img_path)
                    img = preprocess(img)
                    img_list.append(img)

                # batch img
                # 将img_list列表中的所有图像打包成一个batch
                batch_img = torch.stack(img_list, dim=0).to(self.device)
                print("batch_img.shape:", batch_img.shape)
                output = self.model(batch_img).cpu()

                probs, classes = torch.max(output, dim=1)
                # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
                # _, indices = torch.sort(output, descending=True)
                for idx, (_, cla) in enumerate(zip(probs, classes)):
                    result["idx"].append(
                        image_path_list[ids * batch_size + idx].split("/")[-1]
                    )
                    result["ret"].append(self.idx2label[cla.numpy()])
        predict_end = datetime.datetime.now()
        predict_time = (predict_end - predict_start) / datetime.timedelta(
            microseconds=1
        )
        result["predict_time"] = predict_time
        result["model_load_time"] = self.model_load_time
        result["model_trans_time"] = self.model_trans_time
        return result


s = zerorpc.Server(Model(), heartbeat=None)
s.bind("tcp://0.0.0.0:4242")
print("start running")
s.run()
