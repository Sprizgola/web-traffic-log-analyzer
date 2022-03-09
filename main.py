import numpy as np
import pandas as pd
import logging

from datetime import datetime, timedelta
from internal_lib.data_processing import parse_raw_log_data

INPUT_PATH = "data/asd"
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
df["req_duration"] = df.groupby("ip")["timestamp"].shift(-1) - df.groupby("ip")["timestamp"].shift(0)
df.loc[df["req_duration"] > timedelta(minutes=30), "req_duration"] = timedelta(minutes=30)
# I valori 'nan' son dovuti alle sessioni 'nuove'; setto quindi come valore 30*60 (30m) a queste sessioni
df["req_duration"] = df["req_duration"].apply(lambda x: x.total_seconds() if not pd.isna(x) else 30*60)

df["new_session"] = df.groupby("ip")["timestamp"].diff()
df["new_session"] = np.where((df["new_session"] > timedelta(minutes=30)) | (df["new_session"].isnull()), 1, 0)

df["session_id"] = df.sort_values(["ip", "timestamp"])["new_session"].cumsum()

# Calcolo i valori aggregati
df_feature = df.groupby(["ip", "session_id"]).agg(
    session_duration=("timestamp", np.ptp), request_count=("request", np.count_nonzero),
    mean_req=("req_duration", np.mean), total_req_size=("size", np.sum))

df_feature["session_duration"] = df_feature["session_duration"].apply(lambda x: x.total_seconds())
df_feature.loc[df_feature["session_duration"] == 0] = 30*60

# La funzione 'count' considera solo i valori non 'nan'
empty_referer = df.groupby(["ip", "session_id"])["timestamp"].count() - df.groupby(["ip", "session_id"])["referer"].count()
# Percentuale di referer nulli nella sessione
df_feature["pc_referer"] = empty_referer / df.groupby(["ip", "session_id"])["timestamp"].count() * 100

# Percentuale di richieste con errori 4xx
df["errors_4xx"] = df["status"].apply(lambda x: True if "40" in x else False)
n_errors_4xx = df.groupby(["ip", "session_id"])["errors_4xx"].sum()
df_feature["pc_error_4xx"] = n_errors_4xx / df.groupby(["ip", "session_id"])["timestamp"].count() * 100

# Percentuale di richieste che contengono HEAD
df["head_req"] = df["request"].apply(lambda x: True if "HEAD" in x else False)
n_head_req = df.groupby(["ip", "session_id"])["head_req"].sum()
df_feature["pc_head_req"] = n_head_req / df.groupby(["ip", "session_id"])["timestamp"].count() * 100


print(f"Esecuzione terminata in: {datetime.now() - start_time}")
