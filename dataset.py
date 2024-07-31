import argparse
import pandas as pd
import numpy as np

def read_arguments():
    parser = argparse.ArgumentParser(description="Dataset creator.")
    parser.add_argument("--output_file_path", type=str, default="data/d.csv", help="File path of the CSV where the dataset has to be stored.")
    parser.add_argument("--n", type=int, default=20, help="Number of jobs.")
    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    
    # populate list of orders
    output_file_path = args["output_file_path"]
    
    order_id = np.arange(args["n"])

    machine_type = np.random.choice(["lt7", "lt8"], args["n"])

    date_basement_arrival = np.random.randint(1, 20, args["n"])
    #date_basement_arrival = np.full(args["n"], 1)

    date_electrical_panel_arrival = np.random.randint(1, 20, args["n"])
    #date_electrical_panel_arrival = np.full(args["n"], 1)

    date_delivery = date_basement_arrival + np.random.randint(20, 50, args["n"])
    #date_delivery = np.full(args["n"], 1)

    df = pd.DataFrame({
        'order_id': order_id,
        'machine_type': machine_type,
        'date_basement_arrival': date_basement_arrival,
        'date_electrical_panel_arrival': date_electrical_panel_arrival,
        'date_delivery': date_delivery
    })
    df.to_csv(output_file_path, index=False)

    