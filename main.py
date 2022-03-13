import re

import numpy as np
import pandas as pd
import logging

from datetime import datetime, timedelta
from internal_lib.data_processing import parse_raw_log_data, extract_features


INPUT_PATH = "data/log.txt"
OUTPUT_PATH = "data/data_processed.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

start_time = datetime.now()

# parse_raw_log_data(input_path=INPUT_PATH, output_path=OUTPUT_PATH, verbose=True)

df = pd.read_csv(OUTPUT_PATH)
df["status"] = df["status"].astype(str)

df.loc[df["size"] == "-", "size"] = 0
df.loc[:, "size"] = df["size"].astype(int)
df.loc[df["referer"] == "-", "referer"] = None
df.loc[df["user_agent"] == "-", "user_agent"] = None
df.loc[:, "timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%b/%Y:%H:%M:%S %z", utc=True)
df["ref_date"] = df["timestamp"].apply(lambda x: x.date())

# Devo raggruppare i dati in sessioni
df["req_duration"] = df.groupby(["ip", "user_agent"])["timestamp"].shift(-1) - df.groupby(["ip", "user_agent"])["timestamp"].shift(0)
df.loc[df["req_duration"] > timedelta(minutes=30), "req_duration"] = timedelta(minutes=30)
# I valori 'nan' son dovuti alle sessioni 'nuove'; setto quindi come valore 30*60 (30m) a queste sessioni
df["req_duration"] = df["req_duration"].apply(lambda x: x.total_seconds() if not pd.isna(x) else 30*60)

df["new_session"] = df.groupby(["ip", "user_agent"])["timestamp"].diff()
df["new_session"] = np.where((df["new_session"] > timedelta(minutes=30)) | (df["new_session"].isnull()), 1, 0)


COLUMNS_TO_GROUP = ["ip", "session_id", "user_agent"]

# TODO: AGGIUNGERE USER AGENT AL GROUP BY
df["session_id"] = df.sort_values(["ip", "user_agent"])["new_session"].cumsum()


df_features = extract_features(df)



print(f"Esecuzione terminata in: {datetime.now() - start_time}")
