import re

import numpy as np
import pandas as pd
import logging

from datetime import datetime, timedelta
from internal_lib.data_processing import find_sessions, extract_features, parse_raw_log_data

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


INPUT_PATH = "data/log.txt"
OUTPUT_PATH = "data/data_processed.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

start_time = datetime.now()


# data = parse_raw_log_data(input_path=INPUT_PATH, verbose=True)
#
# df_features = extract_features(data)

df = pd.read_csv("parsed_data.txt")

# Effettuo l'encoding delle colonne categoriche
X_cat = pd.get_dummies(df["has_source"], prefix="cat", drop_first=True).values
# df.drop("has_source", axis=1, inplace=True)
#
# df = pd.concat([df, cat_df], axis=1)

# Effettuo lo scaling delle variabili non categoriche

print()

numerical_features = ['session_duration', 'requests_count', 'mean_request', 'total_size', 'pc_referer',
                      'pc_error_4xx', 'pc_head_req', 'pg_img_ratio', 'page_views', 'pc_page_ref_empty', 'login_actions',
                      'internal_search', 'add_to_cart', 'product_views', 'conditions_views', 'homepage_views']

data = df[numerical_features].values

scaler = StandardScaler()
scaler.fit(data)

X_scaled = scaler.transform(data)
X_cat_scaled = scaler.fit_transform(X_cat)
X = np.concatenate([X_scaled, X_cat_scaled], axis=1)

# X NOT SCALED
X = df[[x for x in df.columns if x != "session_id"]].values

import matplotlib.pyplot as plt

# km = KMeans(
#     n_clusters=4, init='k-means++',
#     random_state=42, verbose=2, tol=0.0000001, max_iter=1000)

ssd = list()

K = range(1, 15)
for k in range(1, 8):
    km = KMeans(n_clusters=k, init='random',  random_state=42, verbose=1, tol=0.0000000001, max_iter=1000)
    km = km.fit(X)
    ssd.append(km.inertia_)

plt.plot([x for x in range(1, 8)], ssd, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

print(f"Esecuzione terminata in: {datetime.now() - start_time}")


print(asd)