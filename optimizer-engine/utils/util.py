class Util():
    MIN_NUMBER = -1e9
    @staticmethod
    def resource_cost(cpu_resource, image_size):
        return {
            "cuda": image_size * 10,
            "cpu": cpu_resource * image_size,
        }