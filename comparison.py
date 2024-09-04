import argparse
import pandas as pd
import numpy as np
import random

def read_arguments():
    parser = argparse.ArgumentParser(description="Compare algorithms.")
    parser.add_argument("--out_dir", type=str, default="data", help="Output folder.")
    parser.add_argument("--random_search_file", type=str, help="Random search csv file path.")
    parser.add_argument("--heuristic_file", type=str, help="Deterministic heuristic csv file path.")
    parser.add_argument("--optuna_file", type=str, help="Optuna csv file path.")
    parser.add_argument("--genetic_algorithm_file", type=str, help="Grammatical algorithm csv file path.")
    parser.add_argument("--aco_file", type=str, help="Ant colony optimization csv file path.")
    parser.add_argument("--rl_algorithm_file", type=str, help="Reinforcement learning csv file path.")
    parser.add_argument("--grammatical_evolution_file", type=str, help="Grammatical evolution csv file path.")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset.")
    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    # random search
    df_random = pd.read_csv(args["random_search_file"])
    random_search_makespan = df_random["y"].min()
    print(f"random_search_makespan: {random_search_makespan}")

    # heuristic
    df_heuristic = pd.read_csv(args["heuristic_file"])
    heuristic_makespan = df_heuristic["y"].min()
    print(f"heuristic_makespan: {heuristic_makespan}")

    # optuna
    df_optuna = pd.read_csv(args["optuna_file"])
    optuna_makespan = df_optuna["y"].min()
    print(f"optuna_makespan: {optuna_makespan}")

    # genetic algorithm
    df_ga = pd.read_csv(args["genetic_algorithm_file"])
    ga_makespan = df_ga["worst_fitness"].min()
    print(f"ga_makespan: {ga_makespan}")
    
    # ant colony optimization
    df_aco = pd.read_csv(args["aco_file"])
    aco_makespan = df_aco["worst_fitness"].min()
    print(f"aco_makespan: {aco_makespan}")

    # reinforcement learning
    df_rl = pd.read_csv(args["rl_algorithm_file"])
    rl_makespan = df_rl["y"].min()
    print(f"rl_makespan: {rl_makespan}")

    # grammatical evolution
    df_ge = pd.read_csv(args["grammatical_evolution_file"])
    ge_makespan = df_ge["worst_fitness"].min()
    print(f"ge_makespan: {ge_makespan}")

    tabella_latex = f"""{args['dataset_name']} & ${random_search_makespan}$ & ${heuristic_makespan}$ & ${optuna_makespan}$ & ${ga_makespan}$ & ${aco_makespan}$ & ${rl_makespan}$ & ${ge_makespan}$ \\\\"""
    outputFile = open(f"{args['out_dir']}/tabella.txt", 'w')
    outputFile.write(tabella_latex)
    