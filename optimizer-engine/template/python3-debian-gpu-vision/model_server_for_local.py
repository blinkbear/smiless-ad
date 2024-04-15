import zerorpc
from function.models.model import Model
import os
import logging
from logging import handlers
class Logger(object):
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'crit':logging.CRITICAL
    }#日志级别关系映射

    def __init__(self,filename,level='info',when='D',backCount=3,fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)#设置日志格式
        self.logger.setLevel(self.level_relations.get(level))#设置日志级别
        sh = logging.StreamHandler()#往屏幕上输出
        sh.setFormatter(format_str) #设置屏幕上显示的格式
        th = handlers.TimedRotatingFileHandler(filename=filename,when=when,backupCount=backCount,encoding='utf-8')#往文件里写入#指定间隔时间自动生成文件的处理器
        #实例化TimedRotatingFileHandler
        #interval是时间间隔，backupCount是备份文件的个数，如果超过这个个数，就会自动删除，when是间隔的时间单位，单位有以下几种：
        # S 秒
        # M 分
        # H 小时、
        # D 天、
        # W 每星期（interval==0时代表星期一）
        # midnight 每天凌晨
        th.setFormatter(format_str)#设置文件里写入的格式
        self.logger.addHandler(sh) #把对象加到logger里
        self.logger.addHandler(th)

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
    log = Logger('all.log',level='debug')
    use_model_controller = os.environ.get("USE_MODEL_CONTROLLER", "False")
    backend = os.environ.get("BACKEND", "cpu")
    gpu_quantity = os.environ.get("gpu_quantity", 10)
    os.environ.setdefault("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", gpu_quantity)
    if use_model_controller == "False":
        try:
            s = zerorpc.Server(
                ModelForCPU("", "", backend, use_model_controller), heartbeat=None
            )
            log.logger.info("start bind")
            print("start bind")
            s.bind("tcp://0.0.0.0:4242")
            s.run()
        except Exception as e:
            log.logger.error(e)
            print(e)
    else:
        return


main()
