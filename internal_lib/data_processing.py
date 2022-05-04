import re
import logging

import numpy as np
import pandas
import pandas as pd

from datetime import timedelta, datetime


def parse_raw_log_data(input_path: str, output_path: str, verbose: bool = True):
    """
    This function parse a raw Apache access-log text file into a Pandas dataframe, with each field being represented
    by a column.
    More info on Apache access-log format here:
    :param input_path: str -> path to the raw log file
    :param output_path: str -> path to save the file
    :param verbose:
    :return:
    """

    data_gen = pd.read_csv(
        input_path,
        sep=r'^(\S*).*\[(.*)\]\s"(\S*)\s(\S*)\s([^"]*)"\s(\S*)\s(\S*)\s"([^"]*)"\s"([^"]*)"',
        engine='python',
        na_values='-',
        header=None, index_col=False,
        usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        names=['ip', 'timestamp', 'http_method', 'request', 'http_version', 'status', 'size', 'referer', 'user_agent'],
        dtype={"ip": str, "timestamp": pd.Timestamp, "http_method": str, "request": str,
               "status": str, "size": str, "referer": str, "user_agent": str},
        verbose=verbose,
        chunksize=500000
    )

    for data in data_gen:
        data = data[~data["ip"].isna()]

        data.to_csv(output_path, mode="a", index=False, header=True)


def find_sessions(df: pd.DataFrame):
    """
    This function allow to find the session for each (ip, user_agent).
    The session is computed by grouping each request by ip and user_agent with the condition that the time between
    two adjacent requests cannot exceed 30 minutes.

    :param df: pandas.DataFrame containing the parsed log
    :return: previous dataframe with the session_id label
    """
    logging.info("Inizializzo le sessioni")

    df.drop_duplicates(inplace=True)

    df.sort_values(by=["ip", "user_agent", "timestamp"], inplace=True)

    # df["status"].fillna(0, inplace=True)
    df["size"].fillna(0, inplace=True)

    df.loc[:, "timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%b/%Y:%H:%M:%S %z", utc=True)
    df["ref_date"] = df["timestamp"].dt.date

    df["req_duration"] = df.groupby(["ip", "user_agent"])["timestamp"].diff()

    # Per le sessioni con una sola richiesta assegno la durata pari a 0m; per quelle con più richieste la durata
    # sara' data dalla differenza tra la prima e l'ultima richiesta effettuata nella sessione

    df["new_session"] = np.where((df["req_duration"] > timedelta(minutes=30)) | (df["req_duration"].isnull()), 1, 0)
    df["session_id"] = df.sort_values(["ip", "user_agent"])["new_session"].cumsum()

    # Recupero i secondi
    df["req_duration"] = df["req_duration"] / np.timedelta64(1, "s")
    # Shifto di una riga la durata delle richieste calcolata precedentemente con 'diff'. Assegno poi 0 come valore
    # di default per una richiesta isolata o per un ultima richiesta effettuata
    df["req_duration"] = df.groupby("session_id")["req_duration"].shift(-1)
    df["req_duration"].fillna(0, inplace=True)

    logging.info(f"Ricavate {df.session_id.unique().size} sessioni.")

    return df


def extract_features(df: pd.DataFrame) -> (pd.DataFrame, np.array):
    """
    This function allow to retrieve the features from a dataframe containing the parsed log.
    :param df: pandas.DataFrame containing the parsed log
    :return: (pandas.DataFrame, pandas.DataFrame)
            - Dataframe containing the extracted features
            - Dataframe containing the user agent associated to each session_id
    """
    df = find_sessions(df)

    logging.info("Estraggo le features")

    features_name = ["session_duration", "requests_count", "mean_request", "total_size", "pc_referer", "pc_error_4xx",
                     "pc_head_req", "pg_img_ratio", "page_views", "pc_page_ref_empty", "login_actions",
                     "internal_search", "add_to_cart", "has_source", "product_views", "conditions_views", "homepage_views"]

    empty_matrix = np.empty((df["session_id"].nunique(), len(features_name)))
    empty_matrix[:] = np.nan

    df_feature = pd.DataFrame(index=df.session_id.unique(), data=empty_matrix, columns=features_name)
    logging.info("Session duration")
    # Session duration
    df_feature["session_duration"] = df.groupby(["session_id"])["timestamp"].agg(["min", "max"]).diff(axis=1)["max"]

    logging.info("Requests count")
    # Request count
    df_feature["requests_count"] = df.groupby(["session_id"])["request"].count()

    logging.info("Mean request duration")
    # Mean request duration
    df_feature["mean_request"] = df.groupby(["session_id"])["req_duration"].mean()

    logging.info("Total request data size")
    # Total request size (byte)
    df.loc[df["size"] == "-", "size"] = 0
    df["size"] = df["size"].astype(float)
    df_feature["total_size"] = df.groupby(["session_id"])["size"].sum()

    df_feature["session_duration"] = df_feature["session_duration"] / np.timedelta64(1, "s")

    logging.info("Empty referer requests percent")
    # La funzione 'count' considera solo i valori non 'nan'
    df_feature["pc_referer"] = df.groupby(["session_id"])["timestamp"].count() - df.groupby(["session_id"])["referer"].count()
    # Percentuale di referer nulli nella sessione
    df_feature["pc_referer"] /= df.groupby(["session_id"])["timestamp"].count()
    df_feature["pc_referer"] *= 100

    logging.info("4XX error codes percent")
    # Percentuale di richieste con errori 4xx
    df["status"] = df["status"].astype("str")
    df["errors_4xx"] = df["status"].str.contains("4[0-9]{2}")
    n_errors_4xx = df.groupby(["session_id"])["errors_4xx"].sum()
    df_feature["pc_error_4xx"] = n_errors_4xx / df.groupby(["session_id"])["timestamp"].count() * 100

    logging.info("HEAD request percent")
    # Percentuale di richieste che contengono HEAD
    df["head_req"] = df["http_method"] == "HEAD"
    n_head_req = df.groupby(["session_id"])["head_req"].sum()
    df_feature["pc_head_req"] = n_head_req / df.groupby(["session_id"])["timestamp"].count() * 100

    logging.info("Image/page ratio")
    img_tag = [".jpg", ".gif", ".png", ".avif", ".apng", ".svg", ".webp"]
    regex = "|".join([f"\{x}" for x in img_tag])

    # Richieste che contengono le immagini
    df["has_img"] = df["request"].str.contains(regex)
    n_img_req = df.groupby(["session_id"])["has_img"].sum()
    df_feature["pg_img_ratio"] = n_img_req / df.groupby(["session_id"])["timestamp"].count()

    logging.info("Page views")
    # Page views -> HTML
    df["page_views"] = df["request"].str.contains(".html|.htm", na=False)
    df_feature["page_views"] = df.groupby(["session_id"])["page_views"].sum()

    logging.info("Page with empty referer percent")
    # Percentuale di richieste sulle pagine date da UA nulli
    df_feature["pc_page_ref_empty"] = df[df["page_views"]].referer.isnull().groupby([df["session_id"]]).sum()
    df_feature["pc_page_ref_empty"] /= df_feature["page_views"]
    df_feature["pc_page_ref_empty"] *= 100
    # Effettuo il fill na per le pagine che non hanno referer
    df_feature["pc_page_ref_empty"].fillna(0, inplace=True)

    logging.info("E-Commerce features")

    """
    PARTE RELATIVA ALLE FEATURE SULL'E-COMMERCE
    """

    cart_actions = ["aggiungi_al_carrello.php"]
    login_operations = ["action=login"]
    internal_search_engine = ["search.html\?q", "search-by-bike.html\?"]

    logging.info("Login actions")
    # Number of login operations (including “Register success” and “Login success”)
    df["login_actions"] = df["request"].str.contains("|".join(login_operations))
    df_feature["login_actions"] = df.groupby("session_id")["login_actions"].sum()

    logging.info("Internal search actions")
    # Number of searches using the internal search engine
    df["internal_search"] = df["request"].str.contains("|".join(internal_search_engine))
    df_feature["internal_search"] = df.groupby("session_id")["internal_search"].sum()

    logging.info("Cart actions")
    # Number of operations of adding a product to the shopping cart
    df["add_to_cart"] = df["request"].str.contains("|".join(cart_actions))
    df_feature["add_to_cart"] = df.groupby("session_id")["add_to_cart"].sum()

    logging.info("Request with 'source' session")
    # Check https://support.google.com/analytics/answer/1033863#zippy=%2Cin-this-article
    # Whether a “source” of the session is specified
    df["has_source"] = df["request"].str.contains("utm_?")
    df_feature["has_source"] = df.groupby("session_id")["has_source"].sum() > 0
    df_feature["has_source"] = df_feature["has_source"].astype(int)

    logging.info("Product views")
    # Number of views of product description pages -> currency=[a-z A-Z]{3}&size=[0-9 A-Z]{1,3}
    df["product_views"] = df["request"].str.contains(r"currency=[a-z A-Z]{3}&size=[0-9 A-Z]{1,3}")
    df_feature["product_views"] = df.groupby("session_id")["product_views"].sum()

    logging.info("Condition shipping page read")
    # Number of views of the page with shipping terms and conditions
    df["conditions_views"] = df["request"].str.contains("conditions-of-sale.html")
    df_feature["conditions_views"] = df.groupby("session_id")["conditions_views"].sum()

    logging.info("Homepage views count")
    # Number of views of the website’s home page
    # Because of the different languages, the homepage can be 'it/', 'en/' etc.
    homepage_regex = r"^\/[a-z]{2}\/$"
    df["homepage_views"] = df["request"].str.contains(homepage_regex)
    df_feature["homepage_views"] = df.groupby("session_id")["homepage_views"].sum()

    return df_feature.join(df[["session_id", "user_agent"]].groupby(["session_id"]).first())


def assign_labels(df: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    """
    Function that assign a label ['bot' or 'human'] to the class that has the greater number
    of occurrence of bot/human
    :param df: pandas DataFrame
    :param n_classes: number of classes inside the DataFrame
    :return: pandas DataFrame
    """
    map_labels = dict()

    for i in range(n_classes):

        sub_df = df[df.predicted_labels == i]

        samples = sub_df.shape[0]
        bot = sub_df["is_bot"].sum()
        human = samples - bot

        if bot > human:
            map_labels[i] = "bot"
        else:
            map_labels[i] = "human"
        continue

    df["predicted_labels"] = df["predicted_labels"].map(map_labels)

    return df


def drop_single_value_columns(df: pd.DataFrame):
    """
    Function that drop a list of columns that contains only a single value
    :param df: pandas Dataframe
    :return: None
    """
    columns_to_drop = list()

    for col in df.columns:
        unique_val = df[col].nunique()
        if unique_val == 1:
            columns_to_drop.append(col)

    df.drop(columns_to_drop, axis=1, inplace=True)


def cap_outliers(df: pandas.DataFrame, q: float, columns_to_cap: list) -> pd.DataFrame:
    """
    Function that apply a value based on a quantile threshold for all the columns found in 'columns_to_cap'
    :param df: pandas Dataframe
    :param q: int -> quantile
    :param columns_to_cap:
    :return: pandas Dataframe
    """

    for col in df.columns:
        if col in columns_to_cap:
            threshold = df[col].quantile(q)

            df.loc[df[col] > threshold, col] = threshold

    return df
