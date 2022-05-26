from sys import argv
import Fleet_Optimizer.run_optimizer

UNIQUE_SAMPLES = 20
SAMPLE_COUNT_STRIDE = 2
SAMPLES_PER_COMPONANT_COUNT = 4

CPU_MEAN = 4
CPU_DIV = 3
CPU_CUTOFF_RANGE = (0, 32)

RAM_MEAN = 8
RAM_DIV = 8
RAM_CUTOFF_RANGE = (0, 128)

NET_MEAN = 2
NET_DIV = 1
NET_CUTOFF_RANGE = (0, 5)

TIME_PER_REGION = 10
CANDIDATE_LIST_SIZE = 10
TATION_SCORE_BIAS = 0.8
DEPTH_SCORE_BIAS = 1
PRICE_SCORE_BIAS = 0.5

class Sample:
	class Instance:
		def __init__(self, cpu: int, ram: int, net: int):
			self.cpu = cpu
			self.ram = ram
			self.net = net
	def __init__(self, instances: list):
		"""instances is a list of items of type 'Instance'"""
		self.instaces = instances
	
	def generateInputJson(self, file_name: str):
		pass

class NormDistInt:
	def __init__(self, mean: int, div: int, cutoff_range: tuple):
		pass
	def __call__(self)->int:
		pass

def create_sample(num_componants: int, cpu_dist: NormDistInt, ram_dist: NormDistInt, net_dist: NormDistInt)->Sample:
	pass

def main(argv: list):
	if len(argv) < 1:
		print("missing experiment name. exit.")
		exit(1)
	experiment_name = argv[2]

	comb_counts = [i*SAMPLE_COUNT_STRIDE for i in range(UNIQUE_SAMPLES)]
	samples = []
	cpu_dist = NormDistInt(CPU_MEAN, CPU_DIV, CPU_CUTOFF_RANGE)
	ram_dist = NormDistInt(RAM_MEAN, RAM_DIV, RAM_CUTOFF_RANGE)
	net_dist = NormDistInt(NET_MEAN, NET_DIV, NET_CUTOFF_RANGE)

	for comb_count in comb_counts:
		samples += [create_sample(comb_count, i, cpu_dist, ram_dist, net_dist) 
			for i in range(SAMPLES_PER_COMPONANT_COUNT)]
	
	for sample_idx, sample in enumerate(samples):
		sample_name = experiment_name+"_"+str(sample_idx)
		input_file_name = sample_name + "_input.json"
		output_file_name = sample_name + "_output.json"
		stats_file_name = sample_name + "_stats.sqlite3"
		#TODO: make sure all files are in right directory and not just in root dir.
		sample.generateInputJson(input_file_name)
		Fleet_Optimizer.run_optimizer(
			TIME_PER_REGION,
			CANDIDATE_LIST_SIZE,
			TATION_SCORE_BIAS,
			DEPTH_SCORE_BIAS,
			PRICE_SCORE_BIAS,
			input_file_name=input_file_name,
			output_file_name=output_file_name,
			stats_file_name=stats_file_name
		)

if __name__ == "__main__":
	main(argv)