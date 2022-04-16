import re

import numpy as np
import pandas as pd
import logging

from datetime import datetime, timedelta
from internal_lib.data_processing import find_sessions, extract_features, parse_raw_log_data

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans


INPUT_PATH = "data/log_full.txt"
OUTPUT_PATH = "data/data_processed.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

start_time = datetime.now()

#
# parse_raw_log_data(input_path=INPUT_PATH, verbose=True)
#
# data = pd.read_csv("processed_data/processed_log_1648998818.csv")
# data = data[data["timestamp"] != "timestamp"]
#
# df_features, df_labels = extract_features(data)
# idx = int(datetime.now().timestamp())
#
# df_features.to_csv(f"./parsed_data/parsed_data_{idx}.csv", index="session_id")
# df_labels.to_csv(f"./parsed_data/labels_{idx}.csv", index="session_id")

idx = 1
df = pd.read_csv(f"./parsed_data/parsed_data_{idx}.csv", index_col="session_id")

# Describe dataframe

for col in df.columns:
    nonzero = df[df[col] > 0][col].count()
    zeros = df[df[col] == 0][col].count()
    print(f"Col: {col} - Non zero: {nonzero/df.shape[0]*100:.2f} - Zero-value: {zeros/df.shape[0]*100:.2f}")


# column_to_drop = ["conditions_views", "pc_head_req", ]
# Effettuo il 'cap' degli outlier considerando la colonna "session_duration"
# Limite superiore: 1h -> 3600 s
upper_lim = 60 * 60

df.loc[df['session_duration'] >= upper_lim, "session_duration"] = upper_lim

# Effettuo lo scaling delle variabili non categoriche
numerical_features = ['session_duration', 'requests_count', 'mean_request', 'total_size',
                      'pc_referer', 'pc_error_4xx', 'pc_head_req', 'pg_img_ratio',
                      'page_views', 'pc_page_ref_empty', 'login_actions', 'internal_search',
                      'add_to_cart', 'has_source', 'product_views', 'conditions_views', 'homepage_views']

data = df[numerical_features].fillna(0).values
cat_col = df.has_source.values.reshape(data.shape[0], 1)

scaler = StandardScaler()
scaler.fit(data)

X_scaled = scaler.transform(data)
X_standard = np.concatenate([X_scaled, cat_col], axis=1)

minmax = MinMaxScaler()
X_norm = minmax.fit_transform(np.concatenate([data, cat_col], axis=1))

# X NOT SCALED
X = df[[x for x in df.columns if x != "session_id"]].values

ssd = list()

K = range(1, 15)

for data, data_type in zip([X_norm], ["normalized"]):
    for k in K:
        logging.info(f"K = {k}")
        km = KMeans(n_clusters=k, init='k-means++',  random_state=42, verbose=1, tol=0.0000000001, max_iter=1000)
        km = km.fit(data)
        ssd.append(km.inertia_)

    plt.plot([x for x in K], ssd, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title(f'Elbow Method For Optimal k - {data_type}')
    plt.show()
    ssd = list()

print(f"Esecuzione terminata in: {datetime.now() - start_time}")

