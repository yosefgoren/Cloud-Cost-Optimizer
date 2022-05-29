from audioop import mul
from matplotlib import pyplot as plt
from requests import delete
import Fleet_Optimizer
import math
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

class Flags:
    ALL = None

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

def verify_dict_keys(d: dict, expected_keys: set):
        actual_keys = set(d.keys())
        if actual_keys != expected_keys:
            raise ValueError(f"Got wrong or missing dictionary keys:\n'{actual_keys=}'\n{expected_keys=}")

def dict_of_lists_to_list_of_dicts(dict_of_lists: dict)->list:
    """the function assumes all inner lists have the same lengths and there is at-least one list."""
    keys = list(dict_of_lists.keys())
    n = len(dict_of_lists[keys[0]])
    return [{key:dict_of_lists[key][idx] for key in keys} for idx in range(n)]

def invert_list(ls: list)->list:
    """turn list of lists inside out, function assumes all inner lists have same length.
        [[1,2,3], [4,5,6]] ---> [[1,4], [2,5], [3,6]]
        also, inner lists can also be tuples or sets insetead:
        [(1,2,3), (4,5,6)] ---> [[1,4], [2,5], [3,6]]"""
    return [[inner_list[idx] for inner_list in ls] for idx in range(len(ls[0]))]

def flatten_list_of_lists(ls: list)->list:
    res = []
    for inner_ls in ls:
        res += inner_ls
    return res

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
        
        self.component_count = self.metadata["control_parameters"]["component_count"]
        self.time_per_region = self.metadata["control_parameters"]["time_per_region"]
        self.segnificance = self.metadata["control_parameters"]["significance"]

    def run(self, retry: int, verbosealg: bool, bruteforce: bool):
        algorithm_parameters = self.metadata["algorithm_parameters"]
        control_parameters = self.metadata["control_parameters"]
        for repetition in range(control_parameters["significance"]):
            sample_attempts_left = retry
            def run_alg():
                Fleet_Optimizer.run_optimizer(
                    **algorithm_parameters,
                    time_per_region = control_parameters["time_per_region"],
                    input_file_name = input_path_format(self.exp_dir_path, self.sample_idx),
                    output_file_name = output_path_format(self.exp_dir_path, self.sample_idx, repetition),
                    stats_file_name = stats_path_format(self.exp_dir_path, self.sample_idx, repetition),
                    verbose = verbosealg,
                    bruteforce = bruteforce #TODO: enable different algorithms...
                )
            if retry > 0:
                while True: # This loop will end when no exceptions happen, or retried too many times.
                    try:
                        run_alg()
                    except Exception as e:
                        print(yellow("Error: Unknown exception occured:"))
                        print(e)
                        if sample_attempts_left <= 0:
                            print(yellow("too many errors, stopping run."))
                            raise e
                        sample_attempts_left -= 1
                        continue
                    break
            else:
                run_alg()

    def expected_runtime(self, num_regions: int)->float:
        return self.time_per_region*self.segnificance*float(num_regions)/3600.0
    
    def query_repetition(self, repetition: int, query: str)->list:
        db_path = stats_path_format(self.exp_dir_path, self.sample_idx, repetition)
        with sqlite3.connect(db_path) as conn:
            res = conn.execute(query)
        return list(res)

    def query_stats(self, query: str)->list:
        return [self.query_repetition(repetition, query) for repetition in range(self.segnificance)]

    def filter_entries_inf_price(entries: list)->list:
        """expects input to be a list of tuples where each tuple is (time, price)"""
        return [
            (time, price)
            for time, price in entries
            if (price is not math.inf and price is not np.inf)
        ]

    def get_times_prices(self, region: str, repetition: int = Flags.ALL)->tuple:
        """the first element of the result is a list of floats (times), the second is list of prices."""
        query = f"SELECT INSERT_TIME,BEST_PRICE FROM STATS WHERE REGION_SOLUTION = '{region}' ORDER BY INSERT_TIME;"
        
        entries = flatten_list_of_lists(self.query_stats(query)) if repetition is Flags.ALL else self.query_repetition(repetition, query)
        return invert_list(Sample.filter_entries_inf_price(entries))


def partition_tuples_list_by_field(entries: list, unique_field_idx: int):
    field_values = {entry[unique_field_idx] for entry in entries}
    return {value:[entry for entry in entries if entry[unique_field_idx] == value] for value in field_values}

class Stats:
    """stat fields are: 0:INSERT_TIME, 1:NODES_COUNT, 2:BEST_PRICE, 3:DEPTH_BEST, 4:ITERATION, 5:REGION_SOLUTION"""
    def __init__(self, content: list):
        self.content = content
        self.repetitions_aggregated = False

    def map_entries(self, mapping_func):
        if not self.repetitions_aggregated:
            self.content = [[[mapping_func(entry)
                        for entry in repetition_stats
                    ]
                    for repetition_stats in sample_stats
                ]
                for sample_stats in self.content
            ]
        else:
            self.content = [[mapping_func(entry)
                        for entry in sample_stats
                    ]
                for sample_stats in self.content
            ]
            
    def aggregate_field_partition(self, aggregation_func, unique_field_idx: int):
        """given an index of a field within the query tuple, aggregate together all sets of entries with the same value for that field."""
        if not self.repetitions_aggregated:
            self.content = [[[aggregation_func(stats)
                        for stats in partition_tuples_list_by_field(repetition_stats, unique_field_idx).values()
                    ]
                    for repetition_stats in sample_stats
                ]
                for sample_stats in self.content
            ]
        else:
            self.content = [[aggregation_func(stats)
                        for stats in partition_tuples_list_by_field(sample_stats, unique_field_idx).values()
                    ]
                for sample_stats in self.content
            ]


    def flatten_repetitions(self, aggregation_func = flatten_list_of_lists):
        self.content = [aggregation_func(sample_stats)
            for sample_stats in self.content
        ]
        self.repetitions_aggregated = True

def run_samples(sample_list: list, retry: int, verbosealg: bool, bruteforce: bool, pid: int):
    print(green(f"process {pid}, starting. got {len(sample_list)} samples."))
    for sample in sample_list:
        sample.run(retry, verbosealg, bruteforce)
        print(green(f"process {pid}, finished running sample {sample.sample_idx}."))

def describe_time(time_in_hours: float)->str:
    def trucated_str(num: float)->str:
        pre_dec_dot, post_dec_dot = str(num).split('.')
        return pre_dec_dot + '.' + post_dec_dot[:4]

    if time_in_hours < 1:
        return trucated_str(time_in_hours*60) + " minutes"
    if time_in_hours > 24:
        return trucated_str(time_in_hours/24.0) + " days"
    else:
        return trucated_str(time_in_hours) + " hours"
        
class Experiment:
    default_experiments_root_dir = "./experiments"

    def get_num_samples(self)->int:
        return len(self.samples)

    def query_each_sample(self, query: str)->list:
        return [sample.query_stats(query) for sample in self.samples]

    def get_stats(self, query: str)->Stats:
        return Stats(self.query_each_sample(query))

    def plot_time_price(self, interval_sample_ratio: float = 0.2):
        # get best price for each sample:
        best_prices = self.get_stats("SELECT MIN(BEST_PRICE) FROM STATS;")
        best_prices.flatten_repetitions()
        best_prices = [sample_stats[0][0] for sample_stats in best_prices.content]
        #'best_prices' should now we a list of floats. each one represents the best price seen in a sample.


        # find interval timing:
        times = self.get_stats("SELECT INSERT_TIME FROM STATS ORDER BY INSERT_TIME;")
        times.flatten_repetitions()
        times = [[time for (time,) in sample_stats] for sample_stats in times.content]
        #TODO: perhaps need to drop samples where times don't make sense?
        #'times' should transform as: [[(t1,), (t2,) ...], [...] ...] ---> [[t1, t2 ...], [...] ...]
        experiment_max_min_time = max([min(sample_stats) for sample_stats in times])
        #'experiment_max_min_time' is the amount untill the last sample repetition (aka run of the algorithm) has written it's first entry.
        experiment_max_time = max([max(sample_stats) for sample_stats in times])# each 'sample_stats' contains a list times of entries of the sample.
        total_time = experiment_max_time - experiment_max_min_time
        if total_time == 0:
            fetal_error("Error: difference between first and last entry times is zero")

        num_intervals = math.ceil(self.get_num_samples()*interval_sample_ratio)
        interval_len = float(total_time)/num_intervals
        interval_mids = [experiment_max_min_time+interval_len*(float(i)+0.5) for i in range(num_intervals)]
        
        def get_containing_interval_idx(time: float)->int:
            idx = math.floor(float(time-experiment_max_min_time)/total_time)
            return idx

        # get the best price for each interval for each sample:
        interval_prices_for_each_sample = []
        times_and_prices = self.get_stats("SELECT * FROM STATS ORDER BY INSERT_TIME;")
        times_and_prices.flatten_repetitions()
        times_and_prices.map_entries(lambda entry: (entry[0], entry[2]))

        for sample_times_prices in times_and_prices.content:
            sample_entries_by_interval = [[]]*num_intervals
            for time, price in sample_times_prices:
                sample_entries_by_interval[get_containing_interval_idx(time)].append((time, price))
            
            interval_prices = [
                min(interval_entries+[(None, math.inf)], key=lambda t: t[1])[1]
                for interval_entries in sample_entries_by_interval
            ]
            #'interval_prices' should now have one (best) price for each interval (from this sample).
            #if an interval is empty it will have infinite price and will be filtered later.
            interval_prices_for_each_sample.append(interval_prices)


        # normalize each sample according to 'best_prices':
        for sample_idx in range(self.get_num_samples()):
            interval_prices_for_each_sample[sample_idx] = [
                interval_price/float(best_prices[sample_idx])
                for interval_price in interval_prices_for_each_sample[sample_idx]
            ]
        interval_prices_for_each_interval = invert_list(interval_prices_for_each_sample)

        # aggregate all samples of each interval into average:
        average = lambda ls: sum(ls)/float(len(ls))
        interval_average_prices = [
            average(sample_price) 
            for sample_price in interval_prices_for_each_interval
        ]

        # remove all intervals with infinite price due to missing entries:
        times_and_average_prices = [
            (interval_mids[interval_idx], price)
            for interval_idx, price in enumerate(interval_average_prices)
            if price is not math.inf
        ]

        # plot results:
        x_axis = [time for time, price in times_and_average_prices]
        y_axis = [price for time, price in times_and_average_prices]
        plt.plot(x_axis, y_axis)
        plt.xlabel("time")
        plt.ylabel("average normalized price")
        plt.show()

    ALL = None
    def plot_sample_times_prices(self, 
            region: str,
            sample_idx: int = 0, 
            repetition: int = Flags.ALL
    ):
        plt.plot(*self.samples[sample_idx].get_times_prices(region, repetition))
        repetition_title = "all repetitions" if repetition == Flags.ALL else f"repeition:{repetition}"

        plt.title(f"times & prices ; {repetition_title}, sample:{sample_idx}, region:{region}")
        plt.xlabel("time")
        plt.xlabel("best price")
        plt.show()
    

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
        sample_hw_requirments = lambda : {resource_name:component_resource_distirubtions[resource_name]() for resource_name in component_resource_type_names}
        exp_dir_path = experiments_root_dir+"/"+experiment_name
        if os.path.exists(exp_dir_path) and not force:
            if not bool_prompt(f"an experiment by the name '{experiment_name}'. override old experiment?"):
                exit(0)
        make_experiment_dir(exp_dir_path)

        control_parameter_dicts = dict_of_lists_to_list_of_dicts(control_parameter_lists)
        algorithm_parameter_dicts = dict_of_lists_to_list_of_dicts(algorithm_parameter_lists)
        samples = []

        component_set_generator = lambda : [Component(**(sample_hw_requirments())) for _ in range(max(control_parameter_lists["component_count"]))]
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

    def print_expected_runtime(self, multiprocess: int):
        print(yellow(f" Expected runtime is {describe_time(self.calc_expected_time(multiprocess, 20)*1.1)}."))

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
        
        print(yellow(f"Running '{self.experiment_name}'."))
        self.print_expected_runtime(multiprocess)

        if multiprocess == 1:
            for sample in self.samples:
                sample.run(retry, verbosealg, bruteforce)
                print(green(f"finished running sample {sample.sample_idx}."))
        else:
            samples_each_process = [self.samples[pid:self.get_num_samples():multiprocess] for pid in range(multiprocess)]
            ps = [Process(
                    target=run_samples, args=(samples_each_process[pid], retry, verbosealg, bruteforce, pid)
                )
                for pid in range(multiprocess)
            ]
            for p in ps:
                p.start()
            for p in ps:
                p.join()
