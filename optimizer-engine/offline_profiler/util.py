
MIN_NUMBER = -1e9
CPU_COST={
    # aws Ohio region
    1: 0.034 , # c6g.medium 1c2G10Gbps
    2: 0.068, # c6g.large 2c4G10Gbps
    4: 0.136, # c6g.xlarge 4c8G10Gbps
    8: 0.272, # c6g.2xlarge 8c16G10Gbps
    16: 0.544, # c6g.4xlarge 16c32G10Gbps
    32: 1.088, # c6g.8xlarge 32c64G25Gbps
}
GPU_COST={
    1: 0.306 # p3.2xlarge 8c61G10Gbps+V100
}
class Util():

    @staticmethod
    def resource_cost(cpu_resource):
        # cost_unit = {
        #     "cold_start_factor": 0.5,
        #     "cpu": 0.0084 / 2,
        #     "cuda": 0.01475,
        #     "memory": 0.0084 / 2 ,
        # }
        if cpu_resource in CPU_COST:
            cpu_cost_unit = CPU_COST[cpu_resource]
        else:
            cpu_cost_unit = CPU_COST[1]*cpu_resource
        
        return {
            "cuda": GPU_COST[1],
            "cpu": cpu_cost_unit,
            "memory": cpu_cost_unit
        }