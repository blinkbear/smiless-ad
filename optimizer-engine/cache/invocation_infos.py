from threading import RLock, Lock
import numpy as np
class InvocationInfos:
    def __init__(self):
        self.invocation_numbers = {}
        self.inter_arrival_time = {}
        self.read_invocation_number_lock = RLock()
        self.write_invocation_number_lock = Lock()
        self.read_inter_arrival_time_lock = RLock()
        self.write_inter_arrival_time_lock = Lock()

    def add_new_function(self, function_name):
        with self.write_inter_arrival_time_lock:
            self.inter_arrival_time[function_name] = []
        with self.write_invocation_number_lock:
            self.invocation_numbers[function_name] = []

    def init_function_invocation_number_from_list(
        self, function_name, invocation_numbers
    ):
        ## update invocation_number from local file
        with self.write_invocation_number_lock:
            self.invocation_numbers[function_name] = invocation_numbers

    def update_invocation_info(self, function_name, invocation_number):
        with self.write_invocation_number_lock:
            if function_name not in self.invocation_numbers:
                self.invocation_numbers[function_name] = [invocation_number]
            self.invocation_numbers[function_name].append(invocation_number)
        with self.write_inter_arrival_time_lock:
            if function_name not in self.inter_arrival_time:
                self.inter_arrival_time[function_name] = []
            if invocation_number == 0:
                if len(self.inter_arrival_time[function_name]) > 0:
                    self.inter_arrival_time[function_name][-1] = (
                        self.inter_arrival_time[function_name][-1] + 1
                    )
                else:
                    self.inter_arrival_time[function_name].append(0)
            else:
                self.inter_arrival_time[function_name][-1] = (
                    self.inter_arrival_time[function_name][-1] + 1
                )
                self.inter_arrival_time[function_name].append(0)

    def update_function_inter_arrival_time_from_list(self, workflow_entry, work_round):
        with self.write_inter_arrival_time_lock:
            if workflow_entry not in self.inter_arrival_time:
                self.inter_arrival_time[workflow_entry] = []
                self.inter_arrival_time[workflow_entry].append(work_round)
            else:
                last_round = np.sum(self.inter_arrival_time[workflow_entry])
                self.inter_arrival_time[workflow_entry].append(work_round - last_round)

    def get_function_invocation_number_from_list(self, function_name, round):
        with self.read_invocation_number_lock:
            if function_name not in self.invocation_numbers:
                return []
            return self.invocation_numbers[function_name][:round]

    def get_function_inter_arrival_time_from_list(self, function_name):
        with self.read_inter_arrival_time_lock:
            if function_name not in self.inter_arrival_time:
                return []
            return self.inter_arrival_time[function_name]

    def get_invocation_number(self, function_name):
        with self.read_invocation_number_lock:
            if function_name not in self.invocation_numbers:
                return []
            return self.invocation_numbers[function_name]
    def get_invocation_number_length(self, function_name):
        with self.read_invocation_number_lock:
            if function_name not in self.invocation_numbers:
                return 0
            return len(self.invocation_numbers[function_name]) 
        
        
    def get_inter_arrival_time(self, function_name):
        with self.read_inter_arrival_time_lock:
            if function_name not in self.inter_arrival_time:
                return []
            return self.inter_arrival_time[function_name]

    def get_all_inter_arrival_time(self):
        with self.read_inter_arrival_time_lock:
            return self.inter_arrival_time
    
    def get_all_invocation_number(self):
        with self.read_invocation_number_lock:
            return self.invocation_numbers
        
    def clear_inter_arrival_time(self, function_name):
        with self.write_inter_arrival_time_lock:
            self.inter_arrival_time[function_name] = []

    def clear_invocation_number(self, function_name):
        with self.write_invocation_number_lock:
            self.invocation_numbers[function_name] = []

    def clear_all(self):
        with self.write_inter_arrival_time_lock:
            self.inter_arrival_time = {}
        with self.write_invocation_number_lock:
            self.invocation_numbers = {}

    def remove_function(self, function_name):
        with self.write_inter_arrival_time_lock:
            del self.inter_arrival_time[function_name]
        with self.write_invocation_number_lock:
            del self.invocation_numbers[function_name]

