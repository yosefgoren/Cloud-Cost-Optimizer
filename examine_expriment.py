from run_experiment import get_metadata_dict, stats_path_format
import sqlite3

def query_as_list(db_path: str, query: str)->list:
	with sqlite3.connect(db_path) as conn:
		res = conn.execute(query)
	return list(res)

class ExperimentInfo:
	def __init__(self, experiment_directory_path: str):
		self.exp_dir_path = experiment_directory_path
		self.metadata = get_metadata_dict(self.exp_dir_path)
	
	def get_num_samples(self)->int:
		return len(self.metadata)

	def qury_each_sample(self, query_each: str)->list:
		"""this will return a list where each element is a list representing a sample in the test.
			each inner list is obtained by querying the sql-db of a sample with the query 'query_each'"""
		results = []
		for sample_idx, _ in self.metadata:
			results.append(query_as_list(stats_path_format(self.exp_dir_path, sample_idx), query_each))
		return results


def main():
	query = "SELECT INSERT_TIME, (BEST_PRICE) FROM STATS ORDER BY INSERT_TIME;"

if __name__ == "__main__":
	main()
	