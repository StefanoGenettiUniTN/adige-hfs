import json
import argparse
import time
from datetime import datetime


def read_arguments():
    parser = argparse.ArgumentParser(description="Update digital twin.")
    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()

    while True:
        print("BUONGIORNO AMORE")
        print("TI AMO")
        time.sleep(2)






        """
        # get the current time
        current_time = datetime.now().isoformat()

        # create a dictionary to store the time
        time_data = {
            "current_time": current_time
        }

        # save the time to a JSON file
        with open("time.json", "w") as json_file:
            json.dump(time_data, json_file, indent=4)

        print("Current time saved to time.json")
        
        # wait for 50 seconds before the next update
        time.sleep(50)
        """