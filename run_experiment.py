# ============================= Settings ==============================================
#component counts:
UNIQUE_SAMPLES = 3
SAMPLE_COUNT_STRIDE = 2
SAMPLES_PER_COMPONANT_COUNT = 4

#hw resources distributions:
CPU_MEAN = 4
CPU_DIV = 3
CPU_CUTOFF_RANGE = (1, 32)
RAM_MEAN = 8
RAM_DIV = 8
RAM_CUTOFF_RANGE = (1, 128)
NET_MEAN = 2
NET_DIV = 1
NET_CUTOFF_RANGE = (1, 5)

#algorithm parameters:
ALGORITHM_CORE_PARAMS = {
	"time_per_region": 		10.0,
	"candidate_list_size": 	10,
	"tation_score_bias": 	0.8,
	"depth_score_bias": 	1.0,
	"price_score_bias": 	0.5
}

# ============================= Implementation =========================================

from sys import argv
import Fleet_Optimizer
import json
import numpy as np
import os
import shutil


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
			"filterInstances"	:[],
			"apps" 				:[{
				"app"		:"App",
				"share"		:True,
				"components":comp_dicts
			}]
		}
		with open(file_name, 'w') as file:
			json.dump(json_dict, file, indent=2)
		
		print("created sample input file: ", file_name)

class NormDistInt:
	def __init__(self, mean: int, div: int, cutoff_start: int, cutoff_end: int):
		if cutoff_start >= cutoff_end:
			exception("NormDistInt error: cutoff range start should be samller than cutoff range end.")

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
	comps = [Component(cpu_dist(), ram_dist(), net_dist()) for i in range(num_componants)]
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

def main(argv: list):
	if len(argv) < 2:
		print("Missing experiment name argument.\nPlease enter experiment name:")
		experiment_name = input()
	else:
		experiment_name = argv[2]

	comp_counts = [i*SAMPLE_COUNT_STRIDE for i in range(1, UNIQUE_SAMPLES+1)]
	samples = []
	cpu_dist = NormDistInt(CPU_MEAN, CPU_DIV, CPU_CUTOFF_RANGE[0], CPU_CUTOFF_RANGE[1])
	ram_dist = NormDistInt(RAM_MEAN, RAM_DIV, RAM_CUTOFF_RANGE[0], RAM_CUTOFF_RANGE[1])
	net_dist = NormDistInt(NET_MEAN, NET_DIV, NET_CUTOFF_RANGE[0], NET_CUTOFF_RANGE[1])

	for num_comps in comp_counts:
		samples += [create_sample(num_comps, cpu_dist, ram_dist, net_dist) 
			for i in range(SAMPLES_PER_COMPONANT_COUNT)]
	
	experiment_dir_prefix = "./experiments/"+experiment_name+"/"
	make_experiment_dir(experiment_dir_prefix)
	
	input_dir_prefix = experiment_dir_prefix+"inputs/"
	output_dir_prefix = experiment_dir_prefix+"outputs/"
	stats_dir_prefix = experiment_dir_prefix+"stats/"

	metadata_dict = {}
	for sample_idx, sample in enumerate(samples):	
		sample_name = experiment_name+"_"+str(sample_idx)
		
		metadata_dict[sample_idx] = ALGORITHM_CORE_PARAMS

		input_file_full_name = input_dir_prefix + sample_name + "_input.json"
		output_file_full_name = output_dir_prefix + sample_name + "_output.json"
		stats_file_full_name = stats_dir_prefix + sample_name + "_stats.sqlite3"
	
		sample.generateInputJson(input_file_full_name)
		
		Fleet_Optimizer.run_optimizer(
			ALGORITHM_CORE_PARAMS["time_per_region"],
			ALGORITHM_CORE_PARAMS["candidate_list_size"],
			ALGORITHM_CORE_PARAMS["tation_score_bias"],
			ALGORITHM_CORE_PARAMS["depth_score_bias"],
			ALGORITHM_CORE_PARAMS["price_score_bias"],
			input_file_name=input_file_full_name,
			output_file_name=output_file_full_name,
			stats_file_name=stats_file_full_name
		)

	metadata_file_name = experiment_dir_prefix+"/"+experiment_name+"_metadata.json"
	with open(metadata_file_name, "w") as file:
		json.dump(metadata_dict, file, indent=2)

if __name__ == "__main__":
	main(argv)
