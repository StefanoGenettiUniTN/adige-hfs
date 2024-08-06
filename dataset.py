import argparse
import pandas as pd
import numpy as np
import random

def read_arguments():
    parser = argparse.ArgumentParser(description="Dataset creator.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--out_dir", type=str, default="data", help="Folder where the dataset has to be stored.")
    parser.add_argument("--n", type=int, default=20, help="Number of jobs.")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of dataset instances.")
    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])
    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])
    
    # populate list of orders
    output_folder = args["out_dir"]

    for r in range(args["n_runs"]):    
        order_id = np.arange(args["n"])

        #machine_type = np.random.choice(["lt7", "lt8"], args["n"])
        machine_type = np.random.choice(["lt7",
                                        "lt7_p",
                                        "lt7_ins",
                                        "lt7_p_ins",
                                        "lt8",
                                        "lt8_p",
                                        "lt8_ula",
                                        "lt8_p_ula",
                                        "lt8_12",
                                        "lt8_p_12",
                                        "lt8_12_ula",
                                        "lt8_p_12_ula",], args["n"])
        '''
        machine_type = np.random.choice(["lt7",
                                        "lt7_p",
                                        "lt7_ins",
                                        "lt7_p_ins",], args["n"])
        '''
        '''
        machine_type = np.random.choice(["lt8",
                                        "lt8_p",
                                        "lt8_ula",
                                        "lt8_p_ula",
                                        "lt8_12",
                                        "lt8_p_12",
                                        "lt8_12_ula",
                                        "lt8_p_12_ula",], args["n"])
        '''
        date_basement_arrival = np.random.randint(1, 20, args["n"])
        #date_basement_arrival = np.full(args["n"], 1)

        date_electrical_panel_arrival = np.random.randint(1, 20, args["n"])
        #date_electrical_panel_arrival = np.full(args["n"], 1)

        '''
        date_basement_arrival = []
        date_electrical_panel_arrival = []
        g1 = {"low": 1, "up": 73}
        g2 = {"low": 73, "up": 146}
        g3 = {"low": 146, "up": 219}
        g4 = {"low": 219, "up": 292}
        g5 = {"low": 292, "up": 356}
        for _ in range(args["n"]):
            # each machine assigned uniformly at ranom to one of five groups
            group = rng.randint(1,5)
            if group==1:
                date_basement_arrival.append(rng.randint(g1["low"], g1["up"]))
                date_electrical_panel_arrival.append(rng.randint(g1["low"], g1["up"]))
            elif group==2:
                date_basement_arrival.append(rng.randint(g2["low"], g2["up"]))
                date_electrical_panel_arrival.append(rng.randint(g2["low"], g2["up"]))
            elif group==3:
                date_basement_arrival.append(rng.randint(g3["low"], g3["up"]))
                date_electrical_panel_arrival.append(rng.randint(g3["low"], g3["up"]))
            elif group==4:
                date_basement_arrival.append(rng.randint(g4["low"], g4["up"]))
                date_electrical_panel_arrival.append(rng.randint(g4["low"], g4["up"]))
            elif group==5:
                date_basement_arrival.append(rng.randint(g5["low"], g5["up"]))
                date_electrical_panel_arrival.append(rng.randint(g5["low"], g5["up"]))
        date_basement_arrival = np.array(date_basement_arrival)
        date_electrical_panel_arrival = np.array(date_electrical_panel_arrival)
        '''

        date_delivery = date_basement_arrival + np.random.randint(20, 50, args["n"])
        #date_delivery = np.full(args["n"], 1)

        df = pd.DataFrame({
            'order_id': order_id,
            'machine_type': machine_type,
            'date_basement_arrival': date_basement_arrival,
            'date_electrical_panel_arrival': date_electrical_panel_arrival,
            'date_delivery': date_delivery
        })
        output_file_path = f"{output_folder}/d04/d04_{r}.csv"
        df.to_csv(output_file_path, index=False)

    