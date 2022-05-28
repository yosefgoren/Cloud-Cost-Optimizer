# ============================= Settings ==============================================
N = 4



COMPONENT_COUNTS = [18]*N
SIGNIFICANCE = [14]*N

#core algorithm parameters:
CANDIDATE_LIST_SIZE =              [5*(i**2) for i in range(1, N+1)] #make sure int here!
TIME_PER_REGION =   	        	[15.0]*N
EXPLOITATION_SCORE_PRICE_BIAS =    [0.5]*N
EXPLORATION_SCORE_DEPTH_BIAS =  	[1.0]*N
EXPLOITATION_BIAS =     	        [0.8]*N

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

# ============================= Imports =========================================

import Fleet_Optimizer
import json
import numpy as np
import os
import shutil
import argparse
import math

# ============================= File System Utilities =========================================

def to_json(collection, file_path: str)->None:
    with open(file_path, 'w') as file:
        json.dump(collection, file, indent=2)

def from_json(file_path: str):
    with open(file_path, 'r') as file:
        res = json.load(file)
    return res

def input_path_format(exp_dir_path: str, sample_idx: int)->str:
    return exp_dir_path+"/inputs/sample"+str(sample_idx)+".json"

def output_path_format(exp_dir_path: str, sample_idx: int, repetition: int)->str:
    return exp_dir_path+"/outputs/sample"+str(sample_idx)+"_repetition"+str(repetition)+".json"

def stats_path_format(exp_dir_path: str, sample_idx: int, repetition: int)->str:
    return exp_dir_path+"/stats/sample"+str(sample_idx)+"_repetition"+str(repetition)+".sqlite3"

def metadata_path_formtat(exp_dir_path: str)->str:
    return exp_dir_path+"/metadata.json"

def get_metadata_dict(experiment_dir_path: str)->dict:
    """returns a dictionary containing the medatada saved about the experiment."""
    return {int(key):value for key, value in from_json(metadata_path_formtat(experiment_dir_path)).items()}


# ============================= Implementation =========================================

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

    
def generate_sample_inputs(exp_dir_path: str):
    cpu_dist = NormDistInt(CPU_MEAN, CPU_DIV, CPU_CUTOFF_RANGE[0], CPU_CUTOFF_RANGE[1])
    ram_dist = NormDistInt(RAM_MEAN, RAM_DIV, RAM_CUTOFF_RANGE[0], RAM_CUTOFF_RANGE[1])
    net_dist = NormDistInt(NET_MEAN, NET_DIV, NET_CUTOFF_RANGE[0], NET_CUTOFF_RANGE[1])
    
    samples = [create_sample(num_comps, cpu_dist, ram_dist, net_dist) for num_comps in COMPONENT_COUNTS]
    make_experiment_dir(exp_dir_path)

    metadata_dict = {}
    for sample_idx, sample in enumerate(samples):
        metadata_dict[sample_idx] = {
            "number_of_components":     len(sample.comps),
            "significance": SIGNIFICANCE[sample_idx],
            "algorithm_core_params":    {
                "candidate_list_size" : CANDIDATE_LIST_SIZE[sample_idx],
                "time_per_region" : TIME_PER_REGION[sample_idx],
                "exploitation_score_price_bias" : EXPLOITATION_SCORE_PRICE_BIAS[sample_idx],
                "exploration_score_depth_bias" : EXPLORATION_SCORE_DEPTH_BIAS[sample_idx],
                "exploitation_bias" : EXPLOITATION_BIAS[sample_idx]
            }
        }
        sample.generateInputJson(input_path_format(exp_dir_path, sample_idx))
        
    to_json(metadata_dict, metadata_path_formtat(exp_dir_path))

def run_algorithm_on_samples(exp_dir_path: str, verbosealg: bool, retry: int, bruteforce: bool):
    for sample_idx, sample_metadata in get_metadata_dict(exp_dir_path).items():
        algorithm_core_params = sample_metadata["algorithm_core_params"]
        for repetition in range(sample_metadata["significance"]):
            sample_attempts_left = retry
            while True: # This loop will end when no exceptions happen, or retried too many times.
                try:
                    Fleet_Optimizer.run_optimizer(
                        **algorithm_core_params,
                        input_file_name = input_path_format(exp_dir_path, sample_idx),
                        output_file_name = output_path_format(exp_dir_path, sample_idx, repetition),
                        stats_file_name = stats_path_format(exp_dir_path, sample_idx, repetition),
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

def main(experiment_name: str, run_generate: bool, run_algorithm: bool, verbosealg: bool, retry: int, bruteforce: bool):
    experiment_dir_path = "./experiments/"+experiment_name
    if run_generate:
        generate_sample_inputs(experiment_dir_path)
    if run_algorithm:
        run_algorithm_on_samples(experiment_dir_path, verbosealg, retry, bruteforce)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help='The name of the experiment.')
    parser.add_argument('--nogen', action='store_true', help='run the experiment on all input samples in experiment directory, no new samples are generated.')
    parser.add_argument('--norun', action='store_true', help='create experiment samples and metadata without running the algorithm.')
    parser.add_argument('--verbosealg', action='store_true', help='if enabled - algorithm will print more details during runtime.')
    parser.add_argument('--retry', type=int, help='the number of times the experiment will retry to run the algorithm after unknown exception occurs.')
    parser.add_argument('--bruteforce', action='store_true', help='if selected - use bruteforce algorith instead.')

    args = parser.parse_args()

    experiment_name = args.name
    if experiment_name is None:
        print("Please enter experiment name:")
        experiment_name = input()
    retry = 0 if args.retry is None else args.retry
    significance = 1 if args.significance is None else args.significance
    
    main(experiment_name, run_generate = not args.nogen, run_algorithm=not args.norun, verbosealg=args.verbosealg, retry=retry, bruteforce=args.bruteforce)
