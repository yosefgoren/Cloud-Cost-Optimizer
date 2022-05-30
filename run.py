from Experiment import *
from Distributions import NormDistInt

def create_tation_bias_experiment(N: int, C: int, T: int, Bias: float)->Experiment:
    name = f"time-price_N-{N}_C-{C}_T-{T}_Bias-{Bias}"
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [C]*N,
            "time_per_region":  [float(T)]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [Bias]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(4, 3, 1, 32),
            "ram": NormDistInt(6, 4 ,1, 128),
            "net": NormDistInt(2, 1, 1, 5)
        }
    )

def one_big_sample_experiment()->Experiment:
    N = 1
    name = f"one_big_sample_experiment"
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [100]*N,
            "time_per_region":  [3600.0*5/20]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [0.8]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(3, 3, 1, 32),
            "ram": NormDistInt(4, 4 ,1, 128),
            "net": NormDistInt(1, 1, 1, 4)
        },
        unique_sample_inputs = True
    )


def create_time_price_experiment(N: int, C: int, T: int)->Experiment:
    name = f"time-price_N-{N}_C-{C}_T-{T}"
    return Experiment.create(
        experiment_name=name,
        control_parameter_lists = {
            "component_count":  [C]*N,
            "time_per_region":  [float(T)]*N,
            "significance":     [1]*N
        },
        algorithm_parameter_lists = {
            "candidate_list_size":              [64]*N,   
            "exploitation_score_price_bias":    [0.5]*N,
            "exploration_score_depth_bias":     [1.0]*N,
            "exploitation_bias":                [0.8]*N
        },
        component_resource_distirubtions = {
            "cpu": NormDistInt(3, 3, 1, 32),
            "ram": NormDistInt(4, 4 ,1, 128),
            "net": NormDistInt(1, 1, 1, 4)
        },
        unique_sample_inputs = True
    )

if __name__ == "__main__":
    e = one_big_sample_experiment()
    e.run(retry=3)
    
    # experiment_name = "time-price_N-100_C-10_T-15"
    # experiment_name = "time-price_N-300_C-18_T-20"
    # e = Experiment.load(experiment_name)
    # print(e.get_regions_list()[:3])
    
    # for bias in [0.25*i for i in range(5)]:
    #     e = create_tation_bias_experiment(300, 18, 20, bias)
    # e = create_tation_bias_experiment(200, 18, 20, 0)
    
    # e.print_expected_runtime(6)
    # if not bool_prompt("sure you want to run?"):
    #     exit()
    # # e.run(multiprocess=6, retry=5)
    


    # variablses are: "INSERT_TIME", "NODES_COUNT", "ITERATION", "DEPTH_BEST", "BEST_PRICE"
    # e.plot("INSERT_TIME", "NODES_COUNT", normalize=False)
    # e.plot("ITERATION", "NODES_COUNT", normalize=False)

    # e.plot("INSERT_TIME", "DEPTH_BEST", normalize=False)
    # e.plot("NODES_COUNT", "DEPTH_BEST", normalize=False)
    # e.plot("ITERATION", "DEPTH_BEST", normalize=False)

    # e.plot("INSERT_TIME", "BEST_PRICE", normalize=False)
    # e.plot("NODES_COUNT", "BEST_PRICE", normalize=True)
    # e.plot("ITERATION", "BEST_PRICE", normalize=True)
    