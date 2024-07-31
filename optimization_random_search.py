from typing import List, Dict, Tuple, Set
from alpypeopt import AnyLogicModel
from itertools import permutations
import optuna
import matplotlib.pyplot as plt
import csv
import argparse
import pandas as pd
import random

def read_arguments():
    parser = argparse.ArgumentParser(description="Adige hybrid flow shop scheduling optimizer.")

    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--input_csv", type=str, help="File path of the CSV where the optimization output has been stored.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials.")

    parser.add_argument("--m", type=int, default=1, help="Resources of type M.")
    parser.add_argument("--e", type=int, default=1, help="Resources of type E.")
    parser.add_argument("--r", type=int, default=1, help="Resources of type R.")
    parser.add_argument("--dataset", type=str, default="data/d.csv", help="File path with the dataset.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])

    # history of evaluations
    history_x:List[List[int]] = []
    history_y:List[int] = []

    if args["input_csv"]:
        csv_file_path = args["input_csv"]
        df = pd.read_csv(csv_file_path)
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(df["y"])), df["y"], marker='o', linestyle='-', linewidth=1, markersize=4, label='makespan')
        plt.xlabel('Trial')
        plt.ylabel('Revenue')
        plt.title('Fitness Trend')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        adige_model = AnyLogicModel(
            env_config={
                'run_exported_model': False
            }
        )

        # init design variable setup
        adige_setup = adige_model.get_jvm().adige.AdigeSetup()

        # read dataset jobs
        input_jobs: List[int] = list()
        input_df = pd.read_csv(args["dataset"])
        input_jobs = input_df["order_id"].tolist()

        def simulation(x: List[int],
                       totalAvailableM: int,
                       totalAvailableE: int,
                       totalAvailableR: int,
                       datasetFilePath: str,
                       reset=True):
            """
            x : list of jobs in descending priority order
            totalAvailableM : number of resources of type M available
            totalAvailableE : number of resources of type E available
            totalAvailableR : number of resources of type R available
            datasetFilePath : file path of the csv file storing the dataset
            """
            # set available resources
            adige_setup.setTotalAvailableM(totalAvailableM)
            adige_setup.setTotalAvailableE(totalAvailableE)
            adige_setup.setTotalAvailableR(totalAvailableR)

            # set dataset file path
            adige_setup.setDatasetFilePath(datasetFilePath)

            # assign priorities
            for job_position, job_id in enumerate(x):
                adige_setup.setPriority(job_id, len(x)-job_position)
            
            # pass input setup and run model until end
            adige_model.setup_and_run(adige_setup)
            
            # extract model output or simulation result
            model_output = adige_model.get_model_output()
            if reset:
                # reset simulation model to be ready for next iteration
                adige_model.reset()
            
            return model_output.getMakespan()

        # run the optimizer
        best_y = float("inf")
        best_x = []
        for _ in range(args["n_trials"]):
            x = input_jobs.copy()
            rng.shuffle(x)
            y = simulation(x=x,
                           totalAvailableM=args["m"],
                           totalAvailableE=args["e"],
                           totalAvailableR=args["r"],
                           datasetFilePath=args["dataset"])
            if y<best_y:
                best_y = y
                best_x = x.copy()
            history_x.append(best_x.copy())
            history_y.append(int(y))

        # print result
        print(f"Solution is {best_x} for a value of {best_y}")

        # run simulation with optimal result to use UI to explore results in AnyLogic
        simulation(x=best_x,
                   totalAvailableM=args["m"],
                   totalAvailableE=args["e"],
                   totalAvailableR=args["r"],
                   datasetFilePath=args["dataset"],
                   reset=False)
        
        # close model
        adige_model.close()

        # store fitness trend history in csv output file
        csv_file_path = "history_random.csv"
        csv_file = open(csv_file_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y"])
        for x, y in zip(history_x, history_y):
            csv_writer.writerow([str(x), y])
        csv_file.close()
        print(f"Data successfully written to history_random.csv")