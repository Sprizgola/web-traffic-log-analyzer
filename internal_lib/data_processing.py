import re
import logging

import numpy as np
import pandas as pd

from datetime import timedelta, datetime


def parse_raw_log_data(input_path: str, verbose: bool = True):

    idx = int(datetime.now().timestamp())

    data_gen = pd.read_csv(
        input_path,
        sep=r'^(\S*).*\[(.*)\]\s"(\S*)\s(\S*)\s([^"]*)"\s(\S*)\s(\S*)\s"([^"]*)"\s"([^"]*)"',
        engine='python',
        na_values='-',
        header=None, index_col=False,
        usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        names=['ip', 'timestamp', 'http_method', 'request', 'http_version', 'status', 'size', 'referer', 'user_agent'],
        dtype={"ip": str, "timestamp": pd.Timestamp, "http_method": str, "request": str,
               "status": str, "size": float, "referer": str, "user_agent": str},
        verbose=verbose,
        chunksize=500000
    )

    for data in data_gen:
        data = data[~data["ip"].isna()]
        data.to_csv(f"./processed_data/processed_log_{idx}.csv", mode="a", index=False)


def find_sessions(df: pd.DataFrame):

    logging.info("Inizializzo le sessioni")

    df.sort_values(by=["ip", "user_agent", "timestamp"], inplace=True)

    df["status"].fillna(0, inplace=True)
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


def extract_features(df: pd.DataFrame) -> pd.DataFrame:

    df = find_sessions(df)

    logging.info("Estraggo le features")

    features_name = ["session_duration", "requests_count", "mean_request", "total_size", "pc_referer", "pc_error_4xx",
                     "pc_head_req", "pg_img_ratio", "page_views", "pc_page_ref_empty", "login_actions",
                     "internal_search", "add_to_cart", "has_source", "product_views", "conditions_views", "homepage_views"]

    empty_matrix = np.empty((df["session_id"].unique().size, len(features_name)))
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
    df_feature["total_size"] = df.groupby(["session_id"])["size"].sum()

    df_feature["session_duration"] = df_feature["session_duration"] / np.timedelta64(1, "s")
    # df_feature.loc[df_feature["session_duration"] == 0, "session_duration"] = 30*60

    logging.info("Empty referer requests percent")
    # La funzione 'count' considera solo i valori non 'nan'
    df_feature["pc_referer"] = df.groupby(["session_id"])["timestamp"].count() - df.groupby(["session_id"])["referer"].count()
    # Percentuale di referer nulli nella sessione
    df_feature["pc_referer"] /= df.groupby(["session_id"])["timestamp"].count()
    df_feature["pc_referer"] *= 100

    logging.info("4XX error codes percent")
    # Percentuale di richieste con errori 4xx
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
    df["page_views"] = df["request"].str.contains(".html|.htm")
    df_feature["page_views"] = df.groupby(["session_id"])["page_views"].sum()

    logging.info("Page with empty referer percent")
    # Percentuale di richieste sulle pagine date da UA nulli
    df_feature["pc_page_ref_empty"] = df[df["page_views"]].referer.isnull().groupby([df["session_id"]]).sum()
    df_feature["pc_page_ref_empty"] /= df_feature["page_views"]
    df_feature["pc_page_ref_empty"] *= 100

    logging.info("E-Commerce features")
    """
    PARTE RELATIVA ALLE FEATURE SULL'E-COMMERCE
    """

    cart_actions = ["aggiungi_al_carrello.php"]
    login_operations = ["action=login"]
    internal_search_engine = ["search.html\?q", "search-by-bike.html\?"]

    # TODO: Number of views of pages informing about the store and the trading company -> Non presente
    # TODO: Number of views of pages with entertainment contents -> Non presente
    # TODO: Number of other page views -> Da chiarire
    # TODO: Whether the session ended with a purchase

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

    # TODO: Whether the session ended with a purchase
    # braintree.php?lang = it & payment = yes

    logging.info("Product views")
    # Number of views of product description pages -> currency=[a-z A-Z]{3}&size=[0-9 A-Z]{1,3}
    df["product_views"] = df["request"].str.contains("currency=[a-z A-Z]{3}&size=[0-9 A-Z]{1,3}")
    df_feature["product_views"] = df.groupby("session_id")["product_views"].sum()

    logging.info("Condition shipping page read")
    # Number of views of the page with shipping terms and conditions
    df["conditions_views"] = df["request"].str.contains("conditions-of-sale.html")
    df_feature["conditions_views"] = df.groupby("session_id")["conditions_views"].sum()

    logging.info("Homepage views count")
    # Number of views of the website’s home page
    homepage_regex = "^\/[a-z]{2}\/$"
    df["homepage_views"] = df["request"].str.contains(homepage_regex)
    df_feature["homepage_views"] = df.groupby("session_id")["homepage_views"].sum()

    return df_feature

