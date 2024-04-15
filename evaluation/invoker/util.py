import json
import os


def parseRequestFile(SOURCEDIR, REQUEST_FILE):
        request_file = os.path.join(SOURCEDIR, REQUEST_FILE)
        with open(request_file, "r") as f:
            request_config = json.load(f)
        return request_config
