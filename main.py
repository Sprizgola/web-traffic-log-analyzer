import json
import re
import multiprocessing
import numpy as np
import pandas as pd
import logging

from datetime import datetime, timedelta
from ast import literal_eval
from internal_lib.data_processing import find_sessions, extract_features, parse_raw_log_data

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

with open("user_agents.txt", "r") as f:
    ua_list = literal_eval(f.read())
    ua_list = [x.lower() for x in ua_list]

regex = "|".join([f"{x}" for x in ua_list])
regex = re.compile(regex)


def str_contain(s: str):
    global regex

    return bool(regex.search(s))


if __name__ == "__main__":
    multiprocessing.freeze_support()

    INPUT_PATH = "data/log_full.txt"
    OUTPUT_PATH = "data/data_processed.csv"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

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
    df.fillna(0, inplace=True)

    df_labels = pd.read_csv(f"./parsed_data/labels_{idx}.csv", index_col="session_id")
    df_labels.drop("Unnamed: 0", axis=1, inplace=True)

    is_bot = pd.read_csv("is_bot.csv", index_col="session_id")

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
    for col in df.columns:
        if col in ["session_duration", "requests_count", "mean_request", "total_size", "page_views"]:
            threshold = df[col].quantile(.99)

            df.loc[df[col] > threshold, col] = threshold

    data = df.values

    scaler = StandardScaler()
    scaler.fit(data)

    X_scaled = scaler.transform(data)

    clusters_range = [5, 50, 761]
    for n_cluster in clusters_range:

        km = KMeans(n_clusters=n_cluster, init='k-means++', verbose=0, tol=0.0000000001, max_iter=100, random_state=42)

        km.fit(X_scaled)

        df["labels"] = km.labels_

        df = df.sample(frac=1)

        # data = df[:50000]

        # sns.set_style("whitegrid")
        # sns.pairplot(data, hue="labels")
        # plt.show()

        # Concateno al dataframe gli ip/user_agents
        df_cluster = df.merge(df_labels, left_index=True, right_index=True)
        df_cluster["user_agent"].fillna("-", inplace=True)
        df_cluster["user_agent"] = df_cluster["user_agent"].str.lower()
        #
        n_classes = len(df_cluster["labels"].unique())
        df_size = df_cluster.shape[0]

        # user_agents = df_cluster["user_agent"].values.tolist()
        # print("Inizio multiprocessing")
        # with multiprocessing.Pool(multiprocessing.cpu_count() - 1 or 1) as pool:
        #     result = pool.map(str_contain, user_agents)

        df_cluster = pd.concat([df_cluster.sort_index(), is_bot.sort_index()], axis=1)

        logging.info(f"Cluster: {n_cluster}")

        for i in range(n_classes):
            sub_df = df_cluster[df_cluster.labels == i]

            samples = sub_df.shape[0]
            n_bot = sub_df["is_bot"].sum()
            not_bot = samples - n_bot

            logging.info(f"Label: {i}\nPerc. samples: {samples/df_size*100:.2f}\n"
                         f"Perc. bot: {n_bot/samples*100:.2f}\n"
                         f"Perc. users: {not_bot/samples*100:.2f}")

            for col in sub_df.columns:
                if col not in ["session_duration", "requests_count", "mean_request", "total_size", "pc_referer", "pc_page_ref_empty"]:
                    continue
                logging.info(f"Col: {col} - Mean: {sub_df[col].mean():.2f} - Min: {sub_df[col].min():.2f} - Max: {sub_df[col].max():.2f}")

        logging.info("\n******************************************************")

    print(f"Esecuzione terminata in: {datetime.now() - start_time}")


















