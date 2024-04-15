import datetime
import os
import torch
from transformers import DistilBertTokenizer

class Model(object):
    def __init__(self, container_path, model_path, backend):
        model_path, tokenizer_path = (
            os.path.join(container_path, model_path.split("#")[0][1:]),
            os.path.join(container_path, model_path.split("#")[1][1:]),
        )
        model = None
        if backend == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        load_begin = datetime.datetime.now()
        model = torch.load(model_path)
        model.eval()
        load_end = datetime.datetime.now()
        trans_begin = datetime.datetime.now()
        model.to(self.device)
        trans_end = datetime.datetime.now()
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        self.model = model
        self.tokenizer = tokenizer
        self.model_load_time = (load_end - load_begin) / datetime.timedelta(
            microseconds=1
        )
        self.model_trans_time = (trans_end - trans_begin) / datetime.timedelta(
            microseconds=1
        )

    def predict(self, return_result, **kwargs):
        input_length = 10
        if "input_length" in kwargs:
            input_length = kwargs["input_length"]
        predict_start = datetime.datetime.now()
        sample_sentence = ["test"] * input_length
        sample_sentence = " ".join(sample_sentence)
        inputs = self.tokenizer(sample_sentence, return_tensors="pt")
        inputs.to(self.device)
        with torch.no_grad():
            logits = self.model(**input).logits
        predicted_class_id = logits.argmax().item()
        predict_end = datetime.datetime.now()
        predict_time = (predict_end - predict_start) / datetime.timedelta(
            microseconds=1
        )
        result = {
            "ret": [self.model.config.id2label[predicted_class_id]],
            "predict_time": predict_time,
            "model_load_time": self.model_load_time,
            "model_trans_time": self.model_trans_time,
        }
        if not return_result:
            del result["ret"]
        return result
