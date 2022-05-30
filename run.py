from Experiment import *
from Distributions import NormDistInt

def create_tation_bias_experiment()->Experiment:
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

def create_time_price_experiment(N: int, C: int, T: int)->Experiment:
    return Experiment.create(
        experiment_name=f"time-price_N-{N}_C-{C}_T-{T}",
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
    # e = Experiment.load("time-price_N-300_C-18_T-20")
    e = create_time_price_experiment(100, 10, 15)
    e.print_expected_runtime(6)
    if not bool_prompt("sure you want to run?"):
        exit()
    e.run(multiprocess=6, retry=5)
    
    # for region in e.get_regions_list()[:-1]:
    #     e.plot_times_prices(region, normalize=True)
    