import datetime
import torch
import os
import zerorpc
import sys
from transformers import AutoTokenizer


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class Model(object):
    def __init__(self):
        SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
        model = None
        model_path = os.path.join(SCRIPT_DIR, "models","textgeneration.pth")
        tokenizer_path = os.path.join(SCRIPT_DIR, "models", "tokenizer")
        backend = os.environ.get("BACKEND", "cpu")
        if backend == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        load_begin = datetime.datetime.now()
        model = torch.load(model_path)
        model.eval()
        load_end = datetime.datetime.now()
        trans_begin = datetime.datetime.now()
        model.to(self.device)
        trans_end = datetime.datetime.now()
        self.tokenizer = tokenizer
        self.model = model
        self.model_load_time = (load_end - load_begin) / datetime.timedelta(
            microseconds=1
        )
        self.model_trans_time = (trans_end - trans_begin) / datetime.timedelta(
            microseconds=1
        )

    def predict(self, input_length=30):
        predict_start = datetime.datetime.now()
        input_ids = self.tokenizer(
            "Today I believe we can finally", return_tensors="pt"
        )
        input_ids.to(self.device)
        outputs = self.model.generate(
            input_ids.input_ids, do_sample=False, max_length=input_length
        )
        res = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
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


s = zerorpc.Server(Model(), heartbeat=None)
s.bind("tcp://0.0.0.0:4242")
print("start running")
s.run()
