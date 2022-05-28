from audioop import mul
import Fleet_Optimizer
import json
import numpy as np
import os
import shutil
import sqlite3
from multiprocessing import Process

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

def metadata_path_format(exp_dir_path: str)->str:
    return exp_dir_path+"/metadata.json"

# ============================= Names =========================================

control_parameter_names = {
    "component_count",
    "time_per_region",
    "significance"
}
algorithm_parameter_names = {
    "candidate_list_size",
    "exploitation_score_price_bias",
    "exploration_score_depth_bias",
    "exploitation_bias"
}
component_resource_type_names = {
    "cpu",
    "ram",
    "net"
}

# ============================= Implementation =========================================

def yellow(msg: str)->str:
    return '\033[93m'+msg+'\033[0m'

def red(msg: str)->str:
    return '\033[91m'+msg+'\033[0m'

def green(msg: str)->str:
    return '\033[92m'+msg+'\033[0m'

def bool_prompt(msg: str):
    print(green(msg+" [Y/n]"))
    i = input()
    return i == "y" or i == "Y"

def fetal_error(msg: str):
    print(red("Fetal Error: "+msg))
    exit(1)

class Component:
    def __init__(self, cpu: int, ram: int, net: int):
        self.cpu = cpu
        self.ram = ram
        self.net = net

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

def get_query(db_path: str, query: str)->list:
    with sqlite3.connect(db_path) as conn:
        res = conn.execute(query)
    return res

def verify_dict_keys(d: dict, expected_keys: set):
        actual_keys = set(d.keys())
        if actual_keys != expected_keys:
            raise ValueError(f"Got wrong or missing dictionary keys:\n'{actual_keys=}'\n{expected_keys=}")

def dict_of_lists_to_list_of_dicts(dict_of_lists: dict)->list:
    """the function assumes all inner lists have the same lengths and there is at-least one list."""
    keys = list(dict_of_lists.keys())
    n = len(dict_of_lists[keys[0]])
    return [{key:dict_of_lists[key][idx] for key in keys} for idx in range(n)]

class Sample:
    @staticmethod
    def create(
            exp_dir_path: str, 
            sample_idx: int, 
            components: list, 
            control_parameters: dict,
            algorithm_parameters: dict
    ):
        verify_dict_keys(control_parameters, control_parameter_names)
        verify_dict_keys(algorithm_parameters, algorithm_parameter_names)

        #create the algorithm input file:
        comp_dicts = [{
            "name"		:"component_"+str(idx),
            "vCPUs"		:comp.cpu,
            "memory"	:comp.ram,
            "network"	:comp.net
        } for idx, comp in enumerate(components)]

        algorithm_input_json_dict = {
            "selectedOs"		:"linux",
            "region"			:"all",
            "spot/onDemand"		:"onDemand",
            "AvailabilityZone"	:"all",
            "Architecture"		:"all",
            "filterInstances"	:["a1", "t4g","i3","t3a","c5a.xlarge"],
            "apps" 				:[{
                "app"		:"App",
                "share"		:True,
                "components":comp_dicts
            }]
        }
        to_json(algorithm_input_json_dict, input_path_format(exp_dir_path, sample_idx))

        #create Sample object
        metadata = {
            "control_parameters" : control_parameters,
            "algorithm_parameters" : algorithm_parameters
        }
        return Sample(metadata, sample_idx, exp_dir_path)
    
    @staticmethod
    def load(
            exp_dir_path: str,
            sample_idx: int
    ):
        return Sample(from_json(metadata_path_format(exp_dir_path))[sample_idx], sample_idx, exp_dir_path)

    def __init__(self, metadata: dict, sample_idx: int, exp_dir_path: str):
        self.metadata = metadata
        self.sample_idx = sample_idx
        self.exp_dir_path = exp_dir_path

    def run(self, retry: int, verbosealg: bool, bruteforce: bool):
        algorithm_parameters = self.metadata["algorithm_parameters"]
        control_parameters = self.metadata["control_parameters"]
        for repetition in range(control_parameters["significance"]):
            sample_attempts_left = retry
            while True: # This loop will end when no exceptions happen, or retried too many times.
                try:
                    Fleet_Optimizer.run_optimizer(
                        **algorithm_parameters,
                        time_per_region = control_parameters["time_per_region"],
                        input_file_name = input_path_format(self.exp_dir_path, self.sample_idx),
                        output_file_name = output_path_format(self.exp_dir_path, self.sample_idx, repetition),
                        stats_file_name = stats_path_format(self.exp_dir_path, self.sample_idx, repetition),
                        verbose = verbosealg,
                        bruteforce = bruteforce #TODO: enable different algorithms...
                    )
                except Exception as e:
                    print(yellow("Error: Unknown exception occured:"))
                    print(e)
                    if sample_attempts_left <= 0:
                        print(yellow("too many errors, stopping run."))
                        raise e
                    sample_attempts_left -= 1
                    continue
                break
    def expected_runtime(self, num_regions: int)->float:
        return self.metadata["control_parameters"]["time_per_region"]*self.metadata["control_parameters"]["significance"]*float(num_regions)/3600

def run_sample(s: Sample, retry: int, verbosealg: bool, bruteforce: bool):
    s.run(retry, verbosealg, bruteforce)

class Experiment:
    default_experiments_root_dir = "./experiments"

    def __init__(self, experiment_name: str, experiments_root_dir: str, samples: list):
        """this method is for internal use only, 'Experiment' objects should be created with the static methods 'create, load'."""
        self.experiment_name = experiment_name
        self.exp_dir_path = experiments_root_dir+"/"+self.experiment_name
        self.samples = samples

    @staticmethod
    def create(
            experiment_name: str,
            control_parameter_lists: dict,
            algorithm_parameter_lists: dict,
            component_resource_distirubtions: dict,
            experiments_root_dir: str = default_experiments_root_dir,
            force: bool = False,
            unique_sample_inputs: bool = True 
    ):
        #verify input correctness:
        verify_dict_keys(control_parameter_lists, control_parameter_names)
        verify_dict_keys(algorithm_parameter_lists, algorithm_parameter_names)
        verify_dict_keys(component_resource_distirubtions, component_resource_type_names)

        lengths = [len(ls) for ls in (list(control_parameter_lists.values())+list(algorithm_parameter_lists.values()))]
        if any([n != lengths[0] for n in lengths]):
            raise ValueError("expected all parameter lists to have same lengths.")
        num_samples = lengths[0]

        #create experiment files and objects
        sample_hw_requirments = lambda : [component_resource_distirubtions[resource_name]() for resource_name in component_resource_type_names]
        exp_dir_path = experiments_root_dir+"/"+experiment_name
        if os.path.exists(exp_dir_path) and not force:
            if not bool_prompt(f"an experiment by the name '{experiment_name}'. override old experiment?"):
                exit(0)
        make_experiment_dir(exp_dir_path)

        control_parameter_dicts = dict_of_lists_to_list_of_dicts(control_parameter_lists)
        algorithm_parameter_dicts = dict_of_lists_to_list_of_dicts(algorithm_parameter_lists)
        samples = []

        component_set_generator = lambda : [Component(*(sample_hw_requirments())) for _ in range(max(control_parameter_lists["component_count"]))]
        if not unique_sample_inputs:
            static_component_set = component_set_generator()
            component_set_generator = lambda : static_component_set

        for sample_idx in range(num_samples):
            num_sample_components = control_parameter_lists["component_count"][sample_idx]
            sample_components = component_set_generator()[:num_sample_components]
            samples.append(Sample.create(
                    exp_dir_path, 
                    sample_idx, 
                    sample_components, 
                    control_parameter_dicts[sample_idx],
                    algorithm_parameter_dicts[sample_idx]
            ))

        to_json([sample.metadata for sample in samples], metadata_path_format(exp_dir_path))
        return Experiment(experiment_name, experiments_root_dir, samples)

    def calc_expected_time(self, num_cores: int, num_regions: int = 20):
        return sum([sample.expected_runtime(num_regions) for sample in self.samples])/float(num_cores)

    @staticmethod
    def load(
            experiment_name: str, 
            experiments_root_dir: str = default_experiments_root_dir
    ):
        exp_dir_path = experiments_root_dir+"/"+experiment_name
        exp_metadata = from_json(metadata_path_format(exp_dir_path))
        samples = [Sample(metadata, sample_idx, exp_dir_path) for sample_idx, metadata in enumerate(exp_metadata)]
        return Experiment(experiment_name, experiments_root_dir, samples)

    def run(self, 
            verbosealg: bool = False, 
            retry: int = 0, 
            bruteforce: bool = False, 
            force: bool = False,
            multiprocess: int = 1
    ):
        if (type(multiprocess) is not int) or (multiprocess <= 0):
            fetal_error("multiprocess argument must be a postive integer ('int').")
        if os.path.exists(output_path_format(self.exp_dir_path, 0, 0)) and not force:
            if not bool_prompt("this experiment has already been run. override results?"):
                return
        
        print(yellow(f"Running '{self.experiment_name}'. Expected runtime is {self.calc_expected_time(multiprocess, 20)} hours."))

        if multiprocess == 1:
            for sample in self.samples:
                sample.run(retry, verbosealg, bruteforce)
        else:
            ps = [Process(target=run_sample, args=(sample, retry, verbosealg, bruteforce)) for sample in self.samples]
            for p in ps:
                p.start()
            for p in ps:
                p.join()

    def get_num_samples(self)->int:
        return len(self.samples)

    def query_each_sample(self, query: str)->list:
        """this will return a list where each element is a list representing a sample in the test.
            each inner list is obtained by querying the sql-db of a sample with 'query'."""
        results = []
        for sample_idx, sample in enumerate(self.samples):
            for repetition in range(sample.metadata["control_parameters"]["significance"]):
                results.append(get_query(stats_path_format(self.exp_dir_path, sample_idx, repetition), query))
        return results
