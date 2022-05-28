from Experiment import Experiment
from Distributions import NormDistInt
# import argparse

import matplotlib.pyplot as plt

def CE_exploitation_bias_A()->Experiment:
    N = 5
    return Experiment.create(
        experiment_name="exploitation_bias_A",
        control_parameter_lists = {
            "component_count":  [16]*N,
            "time_per_region":  [13.0]*N,
            "significance":     [11]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [0.25* i for i in range(N)]
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        }
    )

def CE_exploitation_bias_B()->Experiment:
    N = 20
    return Experiment.create(
        experiment_name="exploitation_bias_B",
        control_parameter_lists = {
            "component_count":  [16]*N,
            "time_per_region":  [13.0]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [float(i)/(N-1) for i in range(N)]
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        }
    )

def CE_exploitation_bias_C()->Experiment:
    N = 20
    return Experiment.create(
        experiment_name="exploitation_bias_C",
        control_parameter_lists = {
            "component_count":  [20]*N,
            "time_per_region":  [13.0]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [float(i)/(N-1) for i in range(N)]
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        unique_sample_inputs = False
    )

def CE_candidate_size_A()->Experiment:
    N = 20
    return Experiment.create(
        experiment_name="candidate_size_A",
        control_parameter_lists = {
            "component_count":  [20]*N,
            "time_per_region":  [13.0]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [float(i)/(N-1) for i in range(N)]
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        unique_sample_inputs = False
    )

if __name__ == "__main__":
    e = CE_exploitation_bias_C()
    e.run(multiprocess=6)
    
    # e = Experiment.load("exploitation_bias_C")

    res = [list(x)[0][0] for x in e.query_each_sample("SELECT MAX(BEST_PRICE) FROM STATS;")]
    sig = 1
    samples_mean_prices = [sum(res[i*sig: (i+1)*sig])/float(sig) for i in range(e.get_num_samples())]

    x_bias_axis = [float(i)/(19) for i in range(20)]
    plt.plot(x_bias_axis, samples_mean_prices)
    plt.show()


# def main(experiment_name: str, run_generate: bool, run_algorithm: bool, verbosealg: bool, retry: int, bruteforce: bool):
#     experiment_dir_path = "./experiments/"+experiment_name
#     if run_generate:
#         generate_sample_inputs(experiment_dir_path)
#     if run_algorithm:
#         run_algorithm_on_samples(experiment_dir_path, verbosealg, retry, bruteforce)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--name', type=str, help='The name of the experiment.')
#     parser.add_argument('--nogen', action='store_true', help='run the experiment on all input samples in experiment directory, no new samples are generated.')
#     parser.add_argument('--norun', action='store_true', help='create experiment samples and metadata without running the algorithm.')
#     parser.add_argument('--verbosealg', action='store_true', help='if enabled - algorithm will print more details during runtime.')
#     parser.add_argument('--retry', type=int, help='the number of times the experiment will retry to run the algorithm after unknown exception occurs.')
#     parser.add_argument('--bruteforce', action='store_true', help='if selected - use bruteforce algorith instead.')

#     args = parser.parse_args()

#     experiment_name = args.name
#     if experiment_name is None:
#         print("Please enter experiment name:")
#         experiment_name = input()
#     retry = 0 if args.retry is None else args.retry
    
#     main(experiment_name, run_generate = not args.nogen, run_algorithm=not args.norun, verbosealg=args.verbosealg, retry=retry, bruteforce=args.bruteforce)
