import datetime
import json
import uuid
import requests
import os
from .invoke import register_invoker, InferenceInvoker

FUNCTION_NAME = "textgeneration"
SOURCEDIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
REQUEST_FILE = "request/textgeneration.json"
session = requests.Session()
session.mount(
    "https://", requests.adapters.HTTPAdapter(pool_connections=512, pool_maxsize=1024)
)
session.mount(
    "http://", requests.adapters.HTTPAdapter(pool_connections=512, pool_maxsize=1024)
)


@register_invoker
class TextGeneration(InferenceInvoker):
    def invoke(self, args):
        (
            url,
            namespace,
            batch_size,
            header,
            workflow,
            timeout,
            request_id,
        ) = args
        request_config = self.parseRequestFile()
        url = f"{url}/{FUNCTION_NAME}-{workflow}.{namespace}"
        for req in request_config:
            query_start = datetime.datetime.now()
            inputs = request_config[req]["inputs"]["demo"]
            for i in range(batch_size):
                request_config[req]["inputs"][f"{uuid.uuid1()}"] = inputs
            request_config[req]["request_id"] = request_id
            del request_config[req]["inputs"]["demo"]
            r = session.get(
                url,
                data=json.dumps(request_config[req]),
                allow_redirects=True,
                headers=header,
            )
            query_end = datetime.datetime.now()
            query_time = (query_end - query_start) / datetime.timedelta(microseconds=1)
            # query_time = float(r.headers.get("X-Execution-Seconds"))
            return {"query_time": query_time, "runtime": r.json()}

    def parseRequestFile(self):
        request_file = os.path.join(SOURCEDIR, REQUEST_FILE)
        with open(request_file, "r") as f:
            request_config = json.load(f)
        return request_config
