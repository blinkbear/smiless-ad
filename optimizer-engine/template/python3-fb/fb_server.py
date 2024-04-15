from function.suppress_stdout_stderr import suppress_stdout_stderr
import os
import zerorpc
import sys
from transformers import AutoTokenizer
import torch
import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class Model(object):
    def __init__(self):
        SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
        model = None
        model_path = os.path.join(SCRIPT_DIR, "models","finbert.pth")
        tokenizer_path = os.path.join(SCRIPT_DIR, "models", "tokenizer")
        backend = os.environ.get("BACKEND", "cpu")
        if backend == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        load_begin = datetime.datetime.now()
        model = torch.load(model_path)
        with suppress_stdout_stderr():
            model.eval()
            load_end = datetime.datetime.now()
            trans_begin = datetime.datetime.now()
            model.to(self.device)
            trans_end = datetime.datetime.now()
        self.model_load_time = (load_end - load_begin) / datetime.timedelta(
            microseconds=1
        )
        self.model_trans_time = (trans_end - trans_begin) / datetime.timedelta(
            microseconds=1
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, input_length):
        predict_start = datetime.datetime.now()
        sentence = ["test"] * input_length
        inputs = self.tokenizer(" ".join(sentence), return_tensors="pt")
        inputs.to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class_id = logits.argmax().item()
        predict_end = datetime.datetime.now()
        predict_time = (predict_end - predict_start) / datetime.timedelta(microseconds=1)
        result = {
            "ret": [self.model.config.id2label[predicted_class_id]],
            "predict_time": predict_time,
            "model_load_time": self.model_load_time,
            "model_trans_time": self.model_trans_time,
        }
        return result


s = zerorpc.Server(Model(), heartbeat=None)
s.bind("tcp://0.0.0.0:4242")
print("start running")
s.run()
