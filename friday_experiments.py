from numpy import exp
from Experiment import *
from Distributions import NormDistInt
from comb_optimizer import DevelopMode, GetNextMode, GetStartNodeMode
import matplotlib.pyplot as plt
import warnings

regions = "all"
# regions = ["us-west-1"]

def RS_Generic(N: int, C: int, T: int, root_dir: str, getnext: GetNextMode, getstart: GetStartNodeMode, name: str)->Experiment:
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
            "get_next_mode":                            [getnext]*N,
            "get_starting_node_mode":                   [getstart]*N
        },
        reset_algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [.5]*N,
            "exploration_score_depth_bias":     [.5]*N,
            "exploitation_bias":                [.5]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        force=False,
        region = regions,
        experiments_root_dir=root_dir
    )

def RS_Greedy(*args)->Experiment:
    return RS_Generic(*args, GetNextMode.GREEDY, GetStartNodeMode.ROOT, "RS_Greedy")
def RS_Root(*args)->Experiment:
    return RS_Generic(*args, GetNextMode.STOCHASTIC_ANNEALING, GetStartNodeMode.ROOT, "RS_Root")
def RS_Random(*args)->Experiment:
    return RS_Generic(*args, GetNextMode.STOCHASTIC_ANNEALING, GetStartNodeMode.RANDOM, "RS_Random")
def get_RS(run_not_load: bool, mp: int, *NCT):
    RSs = {
        "RS_Greedy":RS_Greedy,
        "RS_Root":RS_Root,
        "RS_Random":RS_Random
    }
    experiments = []
    for name, gen in RSs.items():
        if run_not_load:
            e = gen(*NCT, "./experiments/RS")
            e.run(multiprocess=mp, retry=3)
        else:
            e = Experiment.load(name, experiments_root_dir="./experiments/RS")
        experiments.append(e)
    return experiments

def DP_Generic(N: int, C: int, T: int, root_dir: str, proportion: float, name: str)->Experiment:
    print(f"created experiment named:\n{name}\n")
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [C]*N,
            "time_per_region":  [float(T)]*N,
            "significance":     [1]*N
        },
        search_algorithm_parameter_lists = {
            "develop_mode":                             [DevelopMode.PROPORTIONAL]*N,
            "proportion_amount_node_sons_to_develop":   [proportion]*N,
            "get_next_mode":                            [GetNextMode.STOCHASTIC_ANNEALING]*N,
            "get_starting_node_mode":                   [GetStartNodeMode.RESET_SELECTOR]*N
        },
        reset_algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [.5]*N,
            "exploration_score_depth_bias":     [.5]*N,
            "exploitation_bias":                [.5]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        force=False,
        region = regions,
        experiments_root_dir=root_dir
    )

def DP_005(*args)->Experiment:
    return DP_Generic(*args, .05, "DP_005")
def DP_015(*args)->Experiment:
    return DP_Generic(*args, .15, "DP_015")
def DP_025(*args)->Experiment:
    return DP_Generic(*args, .25, "DP_025")
def DP_035(*args)->Experiment:
    return DP_Generic(*args, .35, "DP_035")
def get_DP(run_not_load: bool, mp: int, *NCT):
    DPs = {
        "DP_005":DP_005,
        "DP_015":DP_015,  
        "DP_025":DP_025,  
        "DP_035":DP_035  
    }
    experiments = []
    for name, gen in DPs.items():
        if run_not_load:
            e = gen(*NCT, "./experiments/DP")
            e.run(multiprocess=mp, retry=3)
        else:
            e = Experiment.load(name, "./experiments/DP")
        experiments.append(e)
    return experiments

def IB_Generic(N: int, C: int, T: int, root_dir: str, tation_bias: float, price_bias: float, depth_bias: float, name: str)->Experiment:
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
            "get_starting_node_mode":                   [GetStartNodeMode.RESET_SELECTOR]*N
        },
        reset_algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [price_bias]*N,
            "exploration_score_depth_bias":     [depth_bias]*N,
            "exploitation_bias":                [tation_bias]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        force=False,
        region = regions,
        experiments_root_dir=root_dir
    )

def IB_max_dist(*args)->Experiment:
    return IB_Generic(*args, 0, 1, 0, "IB_max_dist")
def IB_max_depth(*args)->Experiment:
    return IB_Generic(*args, 0, 1, 1, "IB_max_depth")
def IB_max_price(*args)->Experiment:
    return IB_Generic(*args, 1, 1, 0, "IB_max_price")
def IB_max_panelty(*args)->Experiment:
    return IB_Generic(*args, 1, 0, 0, "IB_max_panelty")
def IB_T05_P00_D00(*args)->Experiment:
    return IB_Generic(*args, .5, 0, 0, "IB_T05_P00_D00")
def get_IB_A(run_not_load: bool, mp: int, *NCT):
    IB_As = {
        "IB_max_dist":IB_max_dist,
        "IB_max_depth":IB_max_depth,
        "IB_max_price":IB_max_price,
        "IB_max_panelty":IB_max_panelty,
        "IB_T05_P00_D00":IB_T05_P00_D00
    }
    experiments = []
    root_dir = "./experiments/IB_A"
    for name, gen in IB_As.items():
        if run_not_load:
            e = gen(*NCT, root_dir)
            e.run(multiprocess=mp, retry=3)
        else:
            e = Experiment.load(name, experiments_root_dir=root_dir)
        experiments.append(e)
    return experiments

def IB_T05_P00_D10(*args)->Experiment:
    return IB_Generic(*args, .5, 0.0, 1.0, "IB_T05_P00_D10")
def IB_T05_P00_D05(*args)->Experiment:
    return IB_Generic(*args, .5, 0.0, 0.5, "IB_T05_P00_D05")
def IB_T05_P05_D10(*args)->Experiment:
    return IB_Generic(*args, .5, 0.5, 1.0, "IB_T05_P05_D10")
def IB_T05_P05_D00(*args)->Experiment:
    return IB_Generic(*args, .5, 0.5, 0.0, "IB_T05_P05_D00")
def IB_T05_P10_D10(*args)->Experiment:
    return IB_Generic(*args, .5, 1.0, 1.0, "IB_T05_P10_D10")
def IB_T05_P10_D05(*args)->Experiment:
    return IB_Generic(*args, .5, 1.0, 0.5, "IB_T05_P10_D05")
def IB_T05_P10_D00(*args)->Experiment:
    return IB_Generic(*args, .5, 1.0, 0.0, "IB_T05_P10_D00")
def get_IB_B(run_not_load: bool, mp: int, *NCT):
    IB_Bs = {
        "IB_T05_P00_D10":IB_T05_P00_D10,
        "IB_T05_P00_D05":IB_T05_P00_D05,
        "IB_T05_P05_D10":IB_T05_P05_D10,
        "IB_T05_P05_D00":IB_T05_P05_D00,
        "IB_T05_P10_D10":IB_T05_P10_D10,
        "IB_T05_P10_D05":IB_T05_P10_D05,
        "IB_T05_P10_D00":IB_T05_P10_D00
    }
    experiments = []
    root_dir = "./experiments/IB_B"
    for name, gen in IB_Bs.items():
        if run_not_load:
            e = gen(*NCT, root_dir)
            e.run(multiprocess=mp, retry=3)
        else:
            e = Experiment.load(name, experiments_root_dir=root_dir)
        experiments.append(e)
    return experiments

ALLSTARS = {
    "RS_Random":RS_Random,
    "RS_Greedy":RS_Greedy,
    "RS_Root":RS_Root,
    "DP_025":DP_025,
    "IB_max_dist":IB_max_dist,
    "IB_max_panelty":IB_max_panelty,
    "IB_T05_P10_D05":IB_T05_P10_D05,
    "IB_T05_P10_D00":IB_T05_P10_D00,
}
def get_AS_Big(run_not_load: bool, mp: int):
    NCT = [25, 20, 30]
    # NCT = [5, 5, 2]
    experiments = []
    root_dir = "./experiments/AS_Big"
    for name, gen in ALLSTARS.items():
        if run_not_load:
            e = gen(*NCT, root_dir)
            e.run(multiprocess=mp, retry=3)
        else:
            e = Experiment.load(name, experiments_root_dir=root_dir)
        experiments.append(e)
    return experiments

def bruteforce(N: int, C: int, T: int, root_dir: str, name: str)->Experiment:
    print(f"created experiment named:\n{name}\n")
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [C]*N,
            "time_per_region":  [None]*N,
            "significance":     [1]*N
        },search_algorithm_parameter_lists = {
            "develop_mode":                             [None]*N,
            "proportion_amount_node_sons_to_develop":   [None]*N,
            "get_next_mode":                            [None]*N,
            "get_starting_node_mode":                   [None]*N
        },
        reset_algorithm_parameter_lists = {
            "candidate_list_size":              [None]*N,
            "exploitation_score_price_bias":    [None]*N,
            "exploration_score_depth_bias":     [None]*N,
            "exploitation_bias":                [None]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        },
        force=False,
        region = regions,
        experiments_root_dir=root_dir
    )

def get_AS_Small(run_not_load: bool, mp: int):
    NCT = [25, 8, 5]
    # NCT = [5, 5, 2]
    experiments = []
    root_dir = "./experiments/AS_Small"
    # regular experiments:
    for name, gen in ALLSTARS.items():
        if run_not_load:
            e = gen(*NCT, root_dir)
            e.run(multiprocess=mp, retry=3)
        else:
            e = Experiment.load(name, experiments_root_dir=root_dir)
        experiments.append(e)
    
    #bruteforce:
    if run_not_load:
        e = bruteforce(*NCT, root_dir, "bruteforce")
        e.run(bruteforce=True)
    else:
        e = Experiment.load("bruteforce", experiments_root_dir=root_dir)
    experiments.append(e)
    return experiments

# variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"
if __name__ == "__main__":
    args = [False, 5] #[run/load, mp]
    #run/load: 
    # exps = get_AS_Small(*args)
    # exps = get_AS_Big(*args)
    exps = get_RS(*args)
    exps.append(Experiment.load("IB_max_price", "./experiments/IB_A"))

    #plot:
    for e in exps:
        times, prices = e.get_plot_curves(
                normalize=False, 
                x_variable="INSERT_TIME", 
                y_variable="BEST_PRICE"
        )[0]
        name = e.experiment_name
        if name == "IB_max_price":
            name = "reset selector (max price bias)"            
        plt.plot(times, prices, label=name)
    plt.xlabel("time")
    plt.ylabel("best price")
    plt.legend()
    plt.show()