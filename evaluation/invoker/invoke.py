import abc

INVOKER_FACTORY = {}

# register invoker by its name
def register_invoker(invoker):
    cls_name = invoker.__name__.lower()

    def register(invoker):
        INVOKER_FACTORY[cls_name] = invoker

    return register(invoker)


class InferenceInvoker(abc.ABC):
    @abc.abstractmethod
    def invoke(self, url,namespace, input_size=10):
        pass

    @abc.abstractmethod
    def parseRequestFile(self):
        pass
