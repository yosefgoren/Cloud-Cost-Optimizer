# ============================= Settings ==============================================
N = 4

#number of components in each sample:
EACH_COMPONENT_COUNTS = [18]*N

#core algorithm parameters:
EACH_CANDIDATE_LIST_SIZE =              [10]*N + [5*(i**2) for i in range(1, N+1)] #make sure int here!
EACH_TIME_PER_REGION =   	        	[15.0]*N*2
EACH_EXPLOITATION_SCORE_PRICE_BIAS =    [0.5]*N*2
EACH_EXPLORATION_SCORE_DEPTH_BIAS =  	[1.0]*N*2
EACH_EXPLOITATION_BIAS =     	        [0.2*i for i in range(1, N+1)] + [0.8]*N

#hw resources distributions:
CPU_MEAN = 4
CPU_DIV = 3
CPU_CUTOFF_RANGE = (1, 32)
RAM_MEAN = 6
RAM_DIV = 4
RAM_CUTOFF_RANGE = (1, 128)
NET_MEAN = 2
NET_DIV = 1
NET_CUTOFF_RANGE = (1, 5)

#additional configurations:
FILTER_INSTANCES = ["a1", "t4g","i3","t3a","c5a.xlarge"]

# ============================= Implementation =========================================

import Fleet_Optimizer
import json
import numpy as np
import os
import shutil
import argparse

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Component:
    def __init__(self, cpu: int, ram: int, net: int):
        self.cpu = cpu
        self.ram = ram
        self.net = net

def to_json(collection, file_path: str)->None:
    with open(file_path, 'w') as file:
        json.dump(collection, file, indent=2)

def from_json(file_path: str):
    with open(file_path, 'r') as file:
        res = json.load(file)
    return res

class Sample:
    def __init__(self, comps: list):
        """comps is a list of items of type 'Component'"""
        self.comps = comps
    
    def generateInputJson(self, file_name: str):
        comp_dicts = [{
            "name"		:"component_"+str(idx),
            "vCPUs"		:comp.cpu,
            "memory"	:comp.ram,
            "network"	:comp.net
        } for idx, comp in enumerate(self.comps)]

        json_dict = {
            "selectedOs"		:"linux",
            "region"			:"all",
            "spot/onDemand"		:"onDemand",
            "AvailabilityZone"	:"all",
            "Architecture"		:"all",
            "filterInstances"	:FILTER_INSTANCES,
            "apps" 				:[{
                "app"		:"App",
                "share"		:True,
                "components":comp_dicts
            }]
        }
        to_json(json_dict, file_name)
        
        print("created sample input file: ", file_name)

class NormDistInt:
    def __init__(self, mean: int, div: int, cutoff_start: int, cutoff_end: int):
        if cutoff_start >= cutoff_end:
            raise Exception("NormDistInt error: cutoff range start should be samller than cutoff range end.")

        self.mean = mean
        self.div = div
        self.cutoff_start = cutoff_start
        self.cutoff_end = cutoff_end
        
    def __call__(self)->int:
        while True:
            res = int(np.random.normal(self.mean, self.div, size=(1,))[0])
            if self.cutoff_start < res and res < self.cutoff_end:
                return res

def create_sample(num_componants: int, cpu_dist: NormDistInt, ram_dist: NormDistInt, net_dist: NormDistInt)->Sample:
    comps = [Component(cpu_dist(), ram_dist(), net_dist()) for _ in range(num_componants)]
    return Sample(comps)

def make_experiment_dir(exp_dir_path: str):
    try:
        shutil.rmtree(exp_dir_path)
    except FileNotFoundError:
        pass
    except PermissionError:
        print("error: tried to create expriment dir at: '"+exp_dir_path+"', but got premission error.")
        exit(1)

    os.mkdir(exp_dir_path)
    for name in ["inputs", "outputs", "stats"]:
        os.mkdir(exp_dir_path+"/"+name)

def input_path_format(exp_dir_path: str, exp_name: str, sample_idx: int)->str:
    # return exp_dir_path+"/inputs/"+exp_name+"_"+str(sample_idx)+"_input.json"
    return exp_dir_path+"/inputs/"+str(sample_idx)+".json"

def output_path_format(exp_dir_path: str, exp_name: str, sample_idx: int)->str:
    # return exp_dir_path+"/outputs/"+exp_name+"_"+str(sample_idx)+"_output.json"
    return exp_dir_path+"/outputs/"+str(sample_idx)+".json"

def stats_path_format(exp_dir_path: str, exp_name: str, sample_idx: int)->str:
    # return exp_dir_path+"/stats/"+exp_name+"_"+str(sample_idx)+"_stats.sqlite3"
    return exp_dir_path+"/stats/"+str(sample_idx)+".sqlite3"

def generate_sample_inputs(exp_name: str, exp_dir_path: str, significance: int):
    cpu_dist = NormDistInt(CPU_MEAN, CPU_DIV, CPU_CUTOFF_RANGE[0], CPU_CUTOFF_RANGE[1])
    ram_dist = NormDistInt(RAM_MEAN, RAM_DIV, RAM_CUTOFF_RANGE[0], RAM_CUTOFF_RANGE[1])
    net_dist = NormDistInt(NET_MEAN, NET_DIV, NET_CUTOFF_RANGE[0], NET_CUTOFF_RANGE[1])
    
    samples = []
    for num_comps in EACH_COMPONENT_COUNTS:
        sample = create_sample(num_comps, cpu_dist, ram_dist, net_dist)
        samples += [sample]*significance

    make_experiment_dir(exp_dir_path)

    metadata_dict = {}
    for sample_idx, sample in enumerate(samples):	
        
        metadata_dict[sample_idx] = {
            "number_of_components":     len(sample.comps),   
            "algorithm_core_params":    {
                "candidate_list_size" : EACH_CANDIDATE_LIST_SIZE[sample_idx],
                "time_per_region" : EACH_TIME_PER_REGION[sample_idx],
                "exploitation_score_price_bias" : EACH_EXPLOITATION_SCORE_PRICE_BIAS[sample_idx],
                "exploration_score_depth_bias" : EACH_EXPLORATION_SCORE_DEPTH_BIAS[sample_idx],
                "exploitation_bias" : EACH_EXPLOITATION_BIAS[sample_idx]
            }
        }
        sample.generateInputJson(input_path_format(exp_dir_path, exp_name, sample_idx))
        
    to_json(metadata_dict, exp_dir_path+"/metadata.json")
    

def run_algorithm_on_samples(exp_name: str, exp_dir_path: str, verbosealg: bool, retry: int, bruteforce: bool):
    metadata_dict = {int(key):value for key, value in from_json(exp_dir_path+"/"+"metadata.json").items()}

    for sample_idx, sample_metadata in metadata_dict.items():
        algorithm_core_params = sample_metadata["algorithm_core_params"]
        sample_attempts_left = retry
        while True: #finish this loop when no exceptions happen
            try:
                Fleet_Optimizer.run_optimizer(
                    **algorithm_core_params,
                    input_file_name = input_path_format(exp_dir_path, exp_name, sample_idx),
                    output_file_name = output_path_format(exp_dir_path, exp_name, sample_idx),
                    stats_file_name = stats_path_format(exp_dir_path, exp_name, sample_idx),
                    verbose = verbosealg,
                    bruteforce = bruteforce
                )
            except Exception as e:
                print(f"{bcolors.WARNING}Error: Unknown exception occured:{bcolors.ENDC}")
                print(e)
                if sample_attempts_left <= 0:
                    print(f"{bcolors.WARNING}too many errors, dropping experiment.{bcolors.ENDC}")
                    exit(1)
                sample_attempts_left -= 1
                continue
            break

def main(experiment_name: str, run_generate: bool, run_algorithm: bool, verbosealg: bool, retry: int, bruteforce: bool, significance: int):
    experiment_dir_path = "./experiments/"+experiment_name
    if run_generate:
        generate_sample_inputs(experiment_name, experiment_dir_path, significance)
    if run_algorithm:
        run_algorithm_on_samples(experiment_name, experiment_dir_path, verbosealg, retry, bruteforce)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='The name of the experiment.')
    parser.add_argument('--nogen', action='store_true', help='run the experiment on all input samples in experiment directory, no new samples are generated.')
    parser.add_argument('--norun', action='store_true', help='create experiment samples and metadata without running the algorithm.')
    parser.add_argument('--verbosealg', action='store_true', help='if enabled - algorithm will print more details during runtime.')
    parser.add_argument('--retry', type=int, help='the number of times the experiment will retry to run the algorithm after unknown exception occurs.')
    parser.add_argument('--significance', type=int, help='each sample will be repeated this many times with the same generated input.')

    args = parser.parse_args()

    experiment_name = args.name
    if experiment_name is None:
        print("Please enter experiment name:")
        experiment_name = input()
    retry = 0 if args.retry is None else args.retry
    significance = 1 if args.significance is None else args.significance
    
    main(experiment_name, run_generate = not args.nogen, run_algorithm=not args.norun, verbosealg=args.verbosealg, retry=retry, bruteforce=args.retry, significance=significance)
