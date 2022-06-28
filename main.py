import argparse
import logging
import os.path

import pandas as pd
import numpy as np

from datetime import datetime

from internal_lib.data_processing import cap_outliers, drop_single_value_columns, assign_labels
from internal_lib.utils import print_df_cluster_info, compute_and_plot_silhouette
from internal_lib.data_visualization import plot_confusion_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

FEATURES_LIST = ["session_duration", "requests_count", "mean_request", "total_size", "pc_referer",
                 "pc_error_4xx", "pc_head_req", "pg_img_ratio", "page_views", "pc_page_ref_empty",
                 "login_actions", "internal_search", "add_to_cart", "has_source", "product_views",
                 "conditions_views", "homepage_views"]

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str,
                    help="Input path containing the processed log file")
parser.add_argument("--plot_output_path", type=str, default=None,
                    help="Output path to save the plots")

args = parser.parse_args()
DATA_PATH = args.data_path
PLOT_OUTPUT_PATH = args.plot_output_path


if __name__ == "__main__":

    start_time = datetime.now()

    df = pd.read_csv(DATA_PATH)

    # Drop the rows having 'user_agent' == None
    df = df[df["user_agent"].notna()]

    df.set_index("session_id", inplace=True)

    df["user_agent"].fillna("-", inplace=True)
    df["user_agent"] = df["user_agent"].str.lower()

    map_is_bot = df["is_bot"].to_dict()

    columns_to_keep = list()

    logging.info("Drop the columns containing only one unique value")
    drop_single_value_columns(df)

    logging.info("Apply cap the outliers")
    cap_outliers(df=df, q=.95, columns_to_cap=["session_duration", "requests_count", "mean_request", "product_views",
                                               "total_size", "page_views", "homepage_views", "internal_search"])

    X = df[FEATURES_LIST].values
    y = df["is_bot"].values.reshape(df.shape[0], -1)

    session_ids = df.index.to_numpy().reshape(df.shape[0], -1)

    logging.info("Scaling data")
    # Eseguo lo split train/test
    scaler = StandardScaler()
    scaler.fit(X)

    X_scaled = scaler.transform(X)

    # Stack the session_ids in order to keep track of the labels
    X_scaled = np.concatenate([session_ids, X_scaled], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    clusters_range = [5, 50, 700, 1000]

    logging.info("Start training")

    for n_cluster in clusters_range:
        logging.info(f"Number of clusters: {n_cluster}")

        km = KMeans(n_clusters=n_cluster, init='k-means++', verbose=0, tol=0.0000000001, random_state=42)

        km.fit(X_train[:, 1:])

        km_labels = km.labels_

        n_classes = km.n_clusters

        # Create the train dataframe
        train_df = pd.DataFrame(
            data=np.concatenate([X_train[:, :1].astype(int), km_labels.reshape(km_labels.shape[0], 1)], axis=1),
            columns=["session_id", "predicted_labels"])
        train_df["is_bot"] = train_df["session_id"].map(map_is_bot)

        # Assign the predict labels
        train_df = assign_labels(train_df, n_classes)
        print_df_cluster_info(train_df)

        # is_bot: 'bot' if True else 'human'
        train_df.loc[train_df["is_bot"] == True, "true_label"] = "bot"
        train_df.loc[train_df["is_bot"] == False, "true_label"] = "human"

        # Compute the confusion matrix
        y_train_true = train_df["true_label"].values
        y_train_pred = train_df["predicted_labels"].values

        savefig_path = os.path.join(PLOT_OUTPUT_PATH, f"plots/cm_{n_classes}_train.png") \
            if PLOT_OUTPUT_PATH is not None else None
        plot_confusion_matrix(y_train_true, y_train_pred, savefig_path)
        logging.info(f"Adjusted Rand score: {adjusted_rand_score(y_train_true, y_train_pred)}")
        logging.info("Predict data")

        y_pred = km.predict(X_test[:, 1:])

        pred_df = pd.DataFrame(
            data=np.concatenate([X_test[:, :1].astype(int), y_pred.reshape(y_pred.shape[0], 1)], axis=1),
            columns=["session_id", "predicted_labels"])

        pred_df["is_bot"] = pred_df["session_id"].map(map_is_bot)
        
        pred_df = pred_df.join(df[FEATURES_LIST], on="session_id")
        
        pred_df.loc[pred_df["is_bot"] == True, "true_label"] = "bot"
        pred_df.loc[pred_df["is_bot"] == False, "true_label"] = "human"

        pred_df = assign_labels(pred_df, n_classes)

        print_df_cluster_info(pred_df)

        # Compute the confusion matrix
        y_test_true = pred_df["true_label"].values
        y_test_pred = pred_df["predicted_labels"].values

        savefig_path = os.path.join(PLOT_OUTPUT_PATH, f"plots/cm_{n_classes}_predict.png")\
            if PLOT_OUTPUT_PATH is not None else None
        plot_confusion_matrix(y_test_true, y_test_pred, savefig_path)

        logging.info(f"Adjusted Rand score: {adjusted_rand_score(y_test_true, y_test_pred)}")

        print(f"Homogeneity score: {homogeneity_score(y_test_true, y_test_pred)}")
        print(f"Completeness score: {completeness_score(y_test_true, y_test_pred)}")
        print(f"V score: {v_measure_score(y_test_true, y_test_pred)}")

    logging.info(f"Execution time: {datetime.now() - start_time}")

