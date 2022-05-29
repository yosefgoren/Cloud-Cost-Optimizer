from audioop import mul
from Experiment import Experiment
from Distributions import NormDistInt
# import argparse

import matplotlib.pyplot as plt

def exploitation_bias_A()->Experiment:
    N = 5
    return Experiment.create(
        experiment_name="exploitation_bias_A",
        control_parameter_lists = {
            "component_count":  [5]*N,
            "time_per_region":  [1.0]*N,
            "significance":     [1]*N
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

def exploitation_bias_B()->Experiment:
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

def exploitation_bias_C()->Experiment:
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

def time_price_A()->Experiment:
    N = 360
    return Experiment.create(
        experiment_name="time_price_A",
        control_parameter_lists = {
            "component_count":  [15]*N,
            "time_per_region":  [10.0]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [0.8]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        unique_sample_inputs = True
    )

def time_price()->Experiment:
    N = 4
    return Experiment.create(
        experiment_name="time_price_C",
        control_parameter_lists = {
            "component_count":  [18]*N,
            "time_per_region":  [20.0]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [0.8]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        unique_sample_inputs = True
    )

import sys
if __name__ == "__main__":
    # eb = time_price()
    # eb.run(multiprocess=4)
    eb = Experiment.load("time_price_B")

    for sample_idx in range(eb.get_num_samples()):
        eb.plot_sample_times_prices("ap-south-1", sample_idx)
    # eb.plot_time_price()
