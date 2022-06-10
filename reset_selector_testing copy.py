from Experiment import *
from Distributions import NormDistInt
from comb_optimizer import DevelopMode, GetNextMode, GetStartNodeMode
import numpy as np
import matplotlib.pyplot as plt


def RSE_Generic(N: int, C: int, T: int, root_dir: str, GSNM: GetStartNodeMode, tation_bias: float, name: str)->Experiment:
    print(f"created experiment named:\n{name}\n")
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [C]*N,
            "time_per_region":  [float(T)]*N,
            "significance":     [1]*N
        },
        search_algorithm_parameter_lists = {
            "develop_mode":                             [DevelopMode.ALL]*N,
            "proportion_amount_node_sons_to_develop":   [1]*N,
            "get_next_mode":                            [GetNextMode.STOCHASTIC_ANNEALING]*N,
            "get_starting_node_mode":                   [GSNM]*N
        },
        reset_algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [0.5]*N,
            "exploitation_bias":                [tation_bias]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        force=True,
        region = "us-east-1",
        experiments_root_dir=root_dir
    )

def RSE_root(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.ROOT, 0, "RSE_root_restart")

def RSE_random(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.RANDOM, 0, "RSE_random_restart")

def RSE_selector1_0(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.RESET_SELECTOR, 1.0, "RSE_reset_sel_1_0")

def RSE_selector0_3(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.RESET_SELECTOR, 0.3, "RSE_reset_sel_0_3")

def RSE_selector0_7(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.RESET_SELECTOR, 0.7, "RSE_reset_sel_0_7")

def RSE_selector0_0(*args)->Experiment:
    return RSE_Generic(*args, GetStartNodeMode.RESET_SELECTOR, 0.0, "RSE_reset_sel_0_0")

RSEs = {
    "RSE_root_restart" : RSE_root,
    "RSE_random_restart" : RSE_random,
    "RSE_reset_sel_1_0" : RSE_selector1_0,
    "RSE_reset_sel_0_3" : RSE_selector0_3,
    "RSE_reset_sel_0_7" : RSE_selector0_7,
    "RSE_reset_sel_0_0" : RSE_selector0_0
}

def run_RSEs(N: int, root_dir: str):
    C = 10
    T = 10
    mp = 6

    for RSE_gen in RSEs.values():
        e = RSE_gen(N, C, T, root_dir)
        e.run(multiprocess=mp, retry=2)

def get_cross_section_plots(N: int, chunk: int, series_name: str):
    for i in range(N//chunk):
        first_sample, last_sample = (chunk*i, chunk*(i+1))
        sampels = [i for i in range(first_sample, last_sample)]
        for name in RSEs.keys():
            e = Experiment.load(name, experiments_root_dir=root_dir)
            times, prices = e.get_plot_curves(normalize=False, x_variable="INSERT_TIME", y_variable="BEST_PRICE", sample_indices=sampels)[0]
            plt.plot(times, prices, label=name)
        figname = f"RSE; samples {first_sample}-{last_sample}"
        plt.title(figname)
        plt.xlabel("time")
        plt.ylabel("best price")
        plt.legend()
        plt.savefig("../plots/RSE/"+series_name+"/"+figname)
        plt.clf()

# variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"
if __name__ == "__main__":
    series_name = "RSE-N200-C10"
    root_dir = "./experiments/"+series_name
    N = 800
    #run:
    # run_RSEs(N, root_dir)
    #plot:
    # get_cross_section_plots(N, N//2, series_name)

    for name in RSEs.keys():
        e = Experiment.load(name, experiments_root_dir=root_dir)
        times, prices = e.get_plot_curves(normalize=False, x_variable="INSERT_TIME", y_variable="BEST_PRICE")[0]
        plt.plot(times, prices, label=name)
    plt.title(f"RSE; all samples")
    plt.xlabel("time")
    plt.ylabel("best price")
    plt.legend()
    plt.show()