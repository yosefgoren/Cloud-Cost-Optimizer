from numpy import exp
from Experiment import *
from Distributions import NormDistInt
from comb_optimizer import DevelopMode, GetNextMode, GetStartNodeMode
import matplotlib.pyplot as plt

def serach_algorithm_parameters_format(develop_mode, proportion, get_next, get_start):
    return {
            "develop_mode":                             develop_mode,
            "proportion_amount_node_sons_to_develop":   proportion,
            "get_next_mode":                            get_next,
            "get_starting_node_mode":                   get_start
    } 

def reset_algorithm_parameters_format(candidate_list_size, exploitation_score_price_bias, exploration_score_depth_bias, exploitation_bias):
    return {
            "candidate_list_size":              candidate_list_size,   
            "exploitation_score_price_bias":    exploitation_score_price_bias,
            "exploration_score_depth_bias":     exploration_score_depth_bias,
            "exploitation_bias":                exploitation_bias 
    }

def GenericCreator(
        experiment_name: str,
        search_algorithm_parameters: dict,
        reset_algorithm_parameters: dict,
        bruteforce: bool,
        experiments_root_dir: str,
        N: int, 
        C: int, 
        T: int,
        regions: list
)->Experiment:
    return Experiment.create(
        experiment_name=experiment_name,
        control_parameter_lists = {
            "component_count":  [C]*N,
            "time_per_region":  [float(T)]*N,
            "significance":     [1]*N
        },
        search_algorithm_parameter_lists = {key:[value]*N for key, value in search_algorithm_parameters.items()},
        reset_algorithm_parameter_lists = {key:[value]*N for key, value in reset_algorithm_parameters.items()},
        bruteforce = bruteforce,
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        force=False,
        region = regions,
        experiments_root_dir=experiments_root_dir
    )

# variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"
if __name__ == "__main__":
    NCT = [4, 5, 3]
    
    sp = serach_algorithm_parameters_format(DevelopMode.ALL, 1, GetNextMode.STOCHASTIC_ANNEALING, GetStartNodeMode.RESET_SELECTOR)
    ap = reset_algorithm_parameters_format(16, 0, 0, 0)
    expname = "genexp"
    def basicCreator(experiment_root_dir, N, C, T, regions):
        return GenericCreator(
                experiment_name=expname, 
                search_algorithm_parameters=sp, 
                reset_algorithm_parameters=ap, 
                experiments_root_dir=experiment_root_dir, 
                bruteforce=False, 
                N=N, 
                C=C, 
                T=T, 
                regions=regions
        )

    creator = basicCreator

    s = Series.create([creator], "s1", *NCT, ["us-east-1"])
    s.run()
    s.plot()
