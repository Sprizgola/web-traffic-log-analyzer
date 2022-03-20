import re
import logging

import numpy as np
import pandas as pd

from datetime import timedelta


# def parse_raw_log_data(input_path: str, output_path: str, verbose: bool = True) -> None:
#
#     raw_data = pd.read_csv(input_path, index_col=False, on_bad_lines="warn").values
#
#     regc = re.compile(
#         '^(\S*).*\[(.*)\]\s"(\S*)\s(\S*)\s([^"]*)"\s(\S*)\s(\S*)\s"([^"]*)"\s"([^"]*)"')
#
#     dict_list = list()
#     for idx, row in enumerate(raw_data):
#
#         if verbose:
#             logging.info(f"Row: {idx}/{len(raw_data)}")
#
#         tmp_dict = dict()
#         m = regc.match(row[0])
#         if m is None:
#             continue
#         tmp_dict["ip"] = m.group(1)
#         tmp_dict["timestamp"] = m.group(2)
#         tmp_dict["http_method"] = m.group(3) if m.group(3) != "-" else None
#         tmp_dict["request"] = m.group(4) if m.group(4) != "-" else None
#         tmp_dict["http_version"] = m.group(5) if m.group(5) != "-" else None
#         tmp_dict["status"] = m.group(6) if m.group(6) != "-" else None
#         tmp_dict["size"] = m.group(7) if m.group(7) != "-" else None
#         tmp_dict["referer"] = m.group(8) if m.group(8) != "-" else None
#         tmp_dict["user_agent"] = m.group(9) if m.group(9) != "-" else None
#
#         dict_list.append(tmp_dict)
#
#     pd.DataFrame(dict_list).to_csv(output_path, index=False)


def parse_raw_log_data(input_path: str, verbose: bool = True):
    data = pd.read_csv(
        input_path,
        sep=r'^(\S*).*\[(.*)\]\s"(\S*)\s(\S*)\s([^"]*)"\s(\S*)\s(\S*)\s"([^"]*)"\s"([^"]*)"',
        engine='python',
        na_values='-',
        header=None, index_col=False,
        usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9],
        names=['ip', 'timestamp', 'http_method', 'request', 'http_version', 'status', 'size', 'referer', 'user_agent'],
        dtype={"ip": str, "timestamp": pd.Timestamp, "http_method": str, "request": str,
               "status": str, "size": float, "referer": str, "user_agent": str},
        verbose=verbose
    )

    data = data[~data["ip"].isna()]

    return data


def find_sessions(df: pd.DataFrame):

    logging.info("Inizializzo le sessioni")

    df["status"].fillna(0, inplace=True)
    df["size"].fillna(0, inplace=True)

    df.loc[:, "timestamp"] = pd.to_datetime(df["timestamp"], format="%d/%b/%Y:%H:%M:%S %z", utc=True)
    df["ref_date"] = df["timestamp"].apply(lambda x: x.date())

    # Devo raggruppare i dati in sessioni
    df["req_duration"] = df.groupby(["ip", "user_agent"])["timestamp"].shift(-1) - \
                         df.groupby(["ip", "user_agent"])["timestamp"].shift(0)

    df.loc[df["req_duration"] > timedelta(minutes=30), "req_duration"] = timedelta(minutes=30)

    # FIXME: Verificare se assegnare 0 o 30
    df["req_duration"] = df["req_duration"].apply(lambda x: x.total_seconds() if not pd.isna(x) else 0)

    df["new_session"] = df.groupby(["ip", "user_agent"])["timestamp"].diff()
    df["new_session"] = np.where((df["new_session"] > timedelta(minutes=30)) | (df["new_session"].isnull()), 1, 0)

    df["session_id"] = df.sort_values(["ip", "user_agent"])["new_session"].cumsum()

    logging.info(f"Ricavate {df.index.size} sessioni.")

    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:

    df = find_sessions(df)

    logging.info("Estraggo le features")

    logging.info("Session duration")
    # Session duration
    df_feature = df.groupby(["session_id"])["timestamp"].agg(session_duration=np.ptp)

    logging.info("Requests count")
    # Request count
    df_feature["requests_count"] = df.groupby(["session_id"])["request"].count()

    logging.info("Mean request duration")
    # Mean request duration
    df_feature["mean_request"] = df.groupby(["session_id"])["req_duration"].mean()

    logging.info("Total request data size")
    # Total request size (byte)
    df_feature["total_size"] = df.groupby(["session_id"])["size"].sum()

    df_feature["session_duration"] = df_feature["session_duration"].apply(lambda x: x.total_seconds())
    # df_feature.loc[df_feature["session_duration"] == 0, "session_duration"] = 30*60

    logging.info("Empty referer requests percent")
    # La funzione 'count' considera solo i valori non 'nan'
    empty_referer = df.groupby(["session_id"])["timestamp"].count() - df.groupby(["session_id"])["referer"].count()
    # Percentuale di referer nulli nella sessione
    df_feature["pc_referer"] = empty_referer / df.groupby(["session_id"])["timestamp"].count() * 100

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
    # Page views
    df_feature["page_views"] = df.groupby(["session_id"])["request"].nunique(dropna=False)

    logging.info("Page with empty referer percent")
    # Percentuale di richieste sulle pagine date da UA nulli
    df_feature["pc_page_ref_empty"] = df.groupby(["session_id", "request"])["referer"].count().reset_index().groupby("session_id").sum()
    df_feature["pc_page_ref_empty"] = ((df_feature["page_views"] - df_feature["pc_page_ref_empty"]) / df_feature["page_views"]) * 100

    logging.info("E-Commerce features")
    """
    PARTE RELATIVA ALLE FEATURE SULL'E-COMMERCE
    """

    cart_actions = ["aggiungi_al_carrello.php"]
    login_operations = ["action=login", "msg=login-error"]
    internal_search_engine = ["search.html\?q", "search-by-bike.html\?"]
    source = ["utm_source"]
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
    df_feature["internal_search"] = df.groupby("session_id")["internal_search"].sum().fillna(0)

    logging.info("Cart actions")
    # Number of operations of adding a product to the shopping cart
    df["add_to_cart"] = df["request"].str.contains("|".join(cart_actions))
    df_feature["add_to_cart"] = df.groupby("session_id")["add_to_cart"].sum().fillna(0)

    logging.info("Request with 'source' session")
    # Whether a “source” of the session is specified
    df["has_source"] = df["request"].str.contains("|".join(source))
    df_feature["has_source"] = df.groupby("session_id")["has_source"].sum() > 0

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

