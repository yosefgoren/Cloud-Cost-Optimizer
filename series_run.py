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
        regions: list,
        use_existing_inputs=False
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
        experiments_root_dir=experiments_root_dir,
        use_existing_inputs=use_existing_inputs
    )

def gen_exp(aps,sps,names,experiment_root_dir, N, C, T, regions):
    creators = []
    first_name = False
    for ap,sp,name in zip(aps,sps,names):
        creator = GenericCreator(
                experiment_name=name,
                search_algorithm_parameters=sp,
                reset_algorithm_parameters=ap,
                experiments_root_dir=experiment_root_dir,
                bruteforce=False,
                N=N,
                C=C,
                T=T,
                regions=regions,
                use_existing_inputs = first_name
        )
        if not first_name:
            first_name = name
        creators += [creator]
    return creators




def restart_algs():

    aps = [reset_algorithm_parameters_format(64, 0.5, 0.2, 0.5)]
    sps = [serach_algorithm_parameters_format(DevelopMode.ALL, 1, GetNextMode.STOCHASTIC_ANNEALING,
                                                GetStartNodeMode.RANDOM)]
    names = ["Random_Reset"]

    aps += [reset_algorithm_parameters_format(64, 0.5, 0.2, 0.5)]
    sps += [serach_algorithm_parameters_format(DevelopMode.ALL, 1, GetNextMode.GREEDY,
                                                GetStartNodeMode.RESET_SELECTOR)]
    names += ["Greedy"]

    aps += [reset_algorithm_parameters_format(64, 0.5, 0.2, 0.5)]
    sps += [serach_algorithm_parameters_format(DevelopMode.ALL, 1, GetNextMode.GREEDY,
                                                GetStartNodeMode.ROOT)]
    names += ["Root"]


    return aps ,sps , names

def develop_proportion():
    proportion_val = [i/10 for i in range(1,10)]
    proportion_val += [0.001,0.005]

    aps = [reset_algorithm_parameters_format(64, 0.5, 0.2, 0.5)] * len(proportion_val)
    sps = []
    names = []

    for proportion in proportion_val:
        sps += [serach_algorithm_parameters_format(DevelopMode.PROPORTIONAL, proportion, GetNextMode.STOCHASTIC_ANNEALING,
                                                GetStartNodeMode.RESET_SELECTOR)]
        names += [f"develop_proportion_{proportion*100}"]


    return aps ,sps , names

# variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"

def comp_num_09_trail():
    aps_dp , sps_dp , names_dp = develop_proportion()
    aps_rs , sps_rs , names_rs = restart_algs()

    aps = aps_dp + aps_rs
    sps = sps_dp + sps_rs
    names = names_dp + names_rs

    #add brute force , infinity time

    return aps ,sps , names

def comp_num_20_trail():
    aps_dp, sps_dp, names_dp = develop_proportion()

    aps_rs, sps_rs, names_rs =  restart_algs()

    aps = aps_dp + aps_rs
    sps = sps_dp + sps_rs
    names = names_dp + names_rs

    # add brute force

    return aps, sps, names

    
if __name__ == "__main__":
    NCT = [60, 9, 6]
    aps, sps, names = comp_num_09_trail()
    # print(f"{aps=}")
    # print()
    # print()
    # print(f"{sps=}")
    # print()
    # print()
    # print(f"{names=}")
    # print()
    # print()
    exps_09 = gen_exp(aps,sps,names,"../experiments/09_comp", *NCT, "all")
    s_09 = Series(exps_09)
    # s_09.run(multiprocess=3, retry=3)

    NCT = [60, 20, 6]
    aps, sps, names = comp_num_20_trail()
    exps_20 = gen_exp(aps,sps,names,"../experiments/20_comp", *NCT, "all")
    s_20 = Series(exps_20)
    # s_20.run(multiprocess=3, retry=3)

    # s.plot()
