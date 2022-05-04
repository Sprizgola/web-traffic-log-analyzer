import os.path
import re
import logging
import argparse

import pandas as pd
import numpy as np

from datetime import datetime
from ast import literal_eval

from internal_lib.data_processing import parse_raw_log_data, extract_features


parser = argparse.ArgumentParser(description="Parse raw log file.")
parser.add_argument("--input_path", type=str,
                    help="Input path containing the raw log file")

parser.add_argument("--output_dir", type=str,
                    help="Output directory in which the parsed data will be saved.")

args = parser.parse_args()

INPUT_PATH = args.input_path
OUTPUT_DIR = args.output_dir
# INPUT_PATH = "raw_data/access_log.txt"
# OUTPUT_DIR = "."


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# Bot UA regex: https://gist.github.com/dvlop/fca36213ad6237891609e1e038a3bbc1
with open("user_agents.txt", "r") as f:
    ua_list = literal_eval(f.read())
    ua_list = [x.lower() for x in ua_list]

# Bot-Crawlers UA: https://user-agents.net/
with open("crawlers.json", "r") as f:
    crawlers = literal_eval(f.read())
    crawlers = [x.lower() for x in crawlers]


regex = "|".join([f"{x}" for x in ua_list])
regex += "|" + "|".join([f"{re.escape(x)}" for x in crawlers])
regex = re.compile(regex, re.IGNORECASE)


def str_contain(s: str):
    global regex

    if pd.isna(s):
        return False

    return bool(regex.search(s))


vec_function = np.vectorize(str_contain)

if __name__ == "__main__":

    start_time = datetime.now()

    idx = int(datetime.now().timestamp())

    # Starting parse procedure
    # 1. Parse raw log file
    parse_raw_log_data(input_path=INPUT_PATH,
                       output_path=os.path.join(OUTPUT_DIR, f"data_processed_{idx}.csv"), verbose=True)

    # 2. Extract features
    data = pd.read_csv(os.path.join(OUTPUT_DIR, f"data_processed_{idx}.csv"))
    # 2.1 Remove the headers of the chunks
    data = data[~(data["ip"] == "ip")]
    df_features = extract_features(data)

    # 3. Apply labels to dataframe
    user_agents = df_features["user_agent"].unique()

    labels = vec_function(user_agents)
    map_labels = {ua: label for label, ua in zip(labels, user_agents)}
    df_features["is_bot"] = df_features["user_agent"].map(map_labels)

    # 3. Store parsed dataframe
    df_features.to_csv(f"./parsed_data/parsed_data_{idx}.csv", index=True, index_label="session_id")

    logging.info(f"Execution time: {datetime.now() - start_time}")
