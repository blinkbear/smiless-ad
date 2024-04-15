import datetime
from function.suppress_stdout_stderr import suppress_stdout_stderr
from transformers import T5Tokenizer
import os
import zerorpc
import sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))


class Model(object):
    def __init__(self):
        SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
        model = None
        model_path = os.path.join(SCRIPT_DIR, "models","translation.pth")
        tokenizer_path = os.path.join(SCRIPT_DIR, "models", "tokenizer")
        backend = os.environ.get("BACKEND", "cpu")
        if backend == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        with suppress_stdout_stderr():
            tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
            load_begin = datetime.datetime.now()
            model = torch.load(model_path)
            self.text = self.read_text(os.path.join(SCRIPT_DIR, "story.txt"))
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

    # read text from story.txt
    def read_text(self, file_path):
        with open(file_path, "r") as f:
            text = f.read()
        return text

    def predict(self, input_length):
        predict_start = datetime.datetime.now()
        result = {}
        total_sentence = self.text[:input_length]
        prompt = f"translate English to French: {total_sentence}"
        with suppress_stdout_stderr():
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                self.device
            )
            outputs = self.model.generate(input_ids, max_length=8 * input_length).to(
                "cpu"
            )
            result["res"] = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result["model_load_time"] = self.model_load_time
            result["model_trans_time"] = self.model_trans_time
        predict_end = datetime.datetime.now()
        predict_time = (predict_end - predict_start) / datetime.timedelta(
            microseconds=1
        )
        result["predict_time"] = predict_time
        return result


s = zerorpc.Server(Model(), heartbeat=None)
s.bind("tcp://0.0.0.0:4242")
print("start running")
s.run()
