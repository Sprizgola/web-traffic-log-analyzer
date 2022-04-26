import json
import re
import multiprocessing
import numpy as np
import pandas as pd
import logging

from datetime import datetime, timedelta
from ast import literal_eval
from internal_lib.data_processing import find_sessions, extract_features, parse_raw_log_data
from internal_lib.utils import print_df_cluster_info

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

with open("user_agents.txt", "r") as f:
    ua_list = literal_eval(f.read())
    ua_list = [x.lower() for x in ua_list]

regex = "|".join([f"{x}" for x in ua_list])
regex = re.compile(regex, re.IGNORECASE)


def str_contain(s: str):
    global regex

    if pd.isna(s):
        return False

    return bool(regex.search(s))


vec_function = np.vectorize(str_contain)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    idx = 1650918231

    # idx = int(datetime.now().timestamp())
    #
    # INPUT_PATH = "raw_data/log_full.txt"
    # OUTPUT_PATH = f"processed_data/data_processed_{idx}.csv"
    #
    # logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    #
    start_time = datetime.now()
    #
    # # Starting parse procedure
    # # 1. Parse raw log file
    # parse_raw_log_data(input_path=INPUT_PATH, output_path=OUTPUT_PATH, verbose=True)
    #
    # # 2. Extract features
    # data = pd.read_csv(OUTPUT_PATH)
    # data = data[data["timestamp"] != "timestamp"]
    #
    # df_features, df_labels = extract_features(data)
    #
    # 3. Apply labels to dataframe
    # user_agents = df_labels["user_agent"].unique()
    #
    # labels = vec_function(user_agents)
    # map_labels = {ua: label for label, ua in zip(labels, user_agents)}
    # df_labels["is_bot"] = df_labels["user_agent"].map(map_labels)

    # # 3. Store parsed dataframe
    # df_features.to_csv(f"./parsed_data/parsed_data_{idx}.csv", index="session_id")
    # df_labels.to_csv(f"./parsed_data/labels_{idx}.csv", index="session_id")

    df = pd.read_csv(f"./parsed_data/parsed_data_{idx}.csv")
    df.fillna(0, inplace=True)
    df.set_index("Unnamed: 0", inplace=True)

    df_labels = pd.read_csv(f"./parsed_data/labels_{idx}.csv")
    df_labels.drop("Unnamed: 0", axis=1, inplace=True)
    df_labels = df_labels.groupby(["session_id"]).first()

    df_cluster = df.merge(df_labels, left_index=True, right_index=True)
    df_cluster["user_agent"].fillna("-", inplace=True)
    df_cluster["user_agent"] = df_cluster["user_agent"].str.lower()

    map_is_bot = df_labels["is_bot"].to_dict()
    # Describe dataframe
    columns_to_keep = list()

    # for col in df.columns:
    #     nonzero = df[df[col] > 0][col].count()
    #     zeros = df[df[col] == 0][col].count()
    #     print(f"Col: {col} - Non zero: {nonzero/df.shape[0]*100:.2f} - Zero-value: {zeros/df.shape[0]*100:.2f}")
    #     if nonzero/df.shape[0]*100 > 10:
    #         columns_to_keep.append(col)
    #
    # df = df[columns_to_keep]

    # Effettuo il cap degli outliers
    # for col in df.columns:
    #     if col in ["session_duration", "requests_count", "mean_request", "total_size", "page_views"]:
    #         threshold = df[col].quantile(.99)
    #
    #         df.loc[df[col] > threshold, col] = threshold

    # Eseguo lo split train/test
    X = df.values
    y = df_cluster["is_bot"].values.reshape(df_cluster.shape[0], -1)

    scaler = StandardScaler()
    scaler.fit(X)

    X_scaled = scaler.transform(X)

    session_ids = df.index.to_numpy().reshape(df.shape[0], -1)
    # Stack the session_ids in order to keep track of the labels

    X_scaled = np.concatenate([session_ids, X_scaled], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42, stratify=y)

    clusters_range = [5, 50, 761]

    for n_cluster in clusters_range:

        km = KMeans(n_clusters=n_cluster, init='k-means++', verbose=0, tol=0.0000000001, max_iter=100, random_state=42)

        km.fit(X_train[:, 1:])

        y_pred = km.labels_

        n_classes = len(np.unique(y_pred))
        n_samples = X_train.shape[0]

        logging.info("Inizio procedura di mapping")

        map_labels = dict()

        tmp_df = pd.DataFrame(
            data=np.concatenate([X_train[:, :1].astype(int), y_pred.reshape(y_pred.shape[0], 1)], axis=1),
            columns=["session_id", "labels"])
        tmp_df["is_bot"] = tmp_df["session_id"].map(map_is_bot)

        for i in range(n_classes):

            sub_df = tmp_df[tmp_df.labels == i]

            samples = sub_df.shape[0]
            n_bot = sub_df["is_bot"].sum()
            not_bot = samples - n_bot

            if n_bot > not_bot:
                map_labels[i] = "bot"
            else:
                map_labels[i] = "human"
            continue

        tmp_df["labels"] = tmp_df["labels"].map(map_labels)
        print_df_cluster_info(tmp_df)
        print("*******************************")
    print(f"Esecuzione terminata in: {datetime.now() - start_time}")


















