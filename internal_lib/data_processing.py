import re
import logging
import numpy as np
import pandas as pd


def parse_raw_log_data(input_path: str, output_path: str, verbose: bool = True) -> None:

    raw_data = pd.read_csv(input_path, index_col=False, on_bad_lines="warn").values

    regc = re.compile(
        '^(\S*).*\[(.*)\]\s"(\S*)\s(\S*)\s([^"]*)"\s(\S*)\s(\S*)\s"([^"]*)"\s"([^"]*)"')

    dict_list = list()
    for idx, row in enumerate(raw_data):

        if verbose:
            logging.info(f"Row: {idx}/{len(raw_data)}")

        tmp_dict = dict()
        m = regc.match(row[0])
        if m is None:
            continue
        tmp_dict["ip"] = m.group(1)
        tmp_dict["timestamp"] = m.group(2)
        tmp_dict["http_method"] = m.group(3)
        tmp_dict["request"] = m.group(4)
        tmp_dict["http_version"] = m.group(5)
        tmp_dict["status"] = m.group(6)
        tmp_dict["size"] = m.group(7)
        tmp_dict["referer"] = m.group(8)
        tmp_dict["user_agent"] = m.group(9)

        dict_list.append(tmp_dict)

    pd.DataFrame(dict_list).to_csv(output_path, index=False)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:

    # Calcolo i valori aggregati
    df_feature = df.groupby(["session_id"]).agg(
        session_duration=("timestamp", np.ptp), request_count=("request", "count"),
        mean_req=("req_duration", np.mean), total_req_size=("size", "sum"))

    df_feature["session_duration"] = df_feature["session_duration"].apply(lambda x: x.total_seconds())
    df_feature.loc[df_feature["session_duration"] == 0, "session_duration"] = 30*60

    # La funzione 'count' considera solo i valori non 'nan'
    empty_referer = df.groupby(["session_id"])["timestamp"].count() - df.groupby(["session_id"])["referer"].count()
    # Percentuale di referer nulli nella sessione
    df_feature["pc_referer"] = empty_referer / df.groupby(["session_id"])["timestamp"].count() * 100

    # Percentuale di richieste con errori 4xx
    df["errors_4xx"] = df["status"].str.contains("4[0-9]{2}")
    n_errors_4xx = df.groupby(["session_id"])["errors_4xx"].sum()
    df_feature["pc_error_4xx"] = n_errors_4xx / df.groupby(["session_id"])["timestamp"].count() * 100

    # Percentuale di richieste che contengono HEAD
    df["head_req"] = df["request"].apply(lambda x: True if "HEAD" in x else False)
    n_head_req = df.groupby(["session_id"])["head_req"].sum()
    df_feature["pc_head_req"] = n_head_req / df.groupby(["session_id"])["timestamp"].count() * 100

    img_tag = [".jpg", ".gif", ".png", ".avif", ".apng", ".svg", ".webp"]
    regex = "|".join([f"\{x}" for x in img_tag])

    # Richieste che contengono le immagini
    df["has_img"] = df["request"].str.contains(regex)
    n_img_req = df.groupby(["session_id"])["has_img"].sum()
    df_feature["pg_img_ratio"] = n_img_req / df.groupby(["session_id"])["timestamp"].count()

    # Page views
    df_feature["page_views"] = df.groupby(["session_id"])["request"].nunique(dropna=False)

    # Percentuale di richieste sulle pagine date da UA nulli
    df_feature["pc_page_ref_empty"] = df.groupby(["session_id", "request"])["referer"].count().reset_index().groupby("session_id").sum()
    df_feature["pc_page_ref_empty"] = ((df_feature["page_views"] - df_feature["pc_page_ref_empty"]) / df_feature["page_views"]) * 100

    """
    PARTE RELATIVA ALLE FEATURE SULL'E-COMMERCE
    """

    carrello = ["cart.html", "carrello.html", "warenkorb.html", "panier.html", "cesta.html"]
    cart_actions = ["aggiungi_al_carrello.php"]
    login_operations = ["action=login", "msg=login-error"]
    internal_search_engine = ["search.html\?q", "search-by-bike.html\?"]
    source = ["utm_source"]
    checkout = ["checkout.php"]

    # TODO: Number of views of pages informing about the store and the trading company -> Non presente
    # TODO: Number of views of pages with entertainment contents -> Non presente
    # TODO: Number of other page views -> Da chiarire

    # Number of login operations (including “Register success” and “Login success”)
    df["login_actions"] = df["request"].str.contains("|".join(login_operations))
    df_feature["login_actions"] = df.groupby("session_id")["login_actions"].sum()

    # Number of searches using the internal search engine
    df["internal_search"] = df["request"].str.contains("|".join(internal_search_engine))
    df_feature["internal_search"] = df.groupby("session_id")["internal_search"].sum().fillna(0)

    # Number of operations of adding a product to the shopping cart
    df["add_to_cart"] = df["request"].str.contains("|".join(cart_actions))
    df_feature["add_to_cart"] = df.groupby("session_id")["add_to_cart"].sum().fillna(0)

    # Whether a “source” of the session is specified
    df["has_source"] = df["request"].str.contains("|".join(source))
    df_feature["has_source"] = df.groupby("session_id")["has_source"].sum() > 0

    # TODO: Whether the session ended with a purchase
    # braintree.php?lang = it & payment = yes

    # Number of views of product description pages -> currency=[a-z A-Z]{3}&size=[0-9 A-Z]{1,3}
    df["product_views"] = df["request"].str.contains("currency=[a-z A-Z]{3}&size=[0-9 A-Z]{1,3}")
    df_feature["product_views"] = df.groupby("session_id")["product_views"].sum()

    # Number of views of the page with shipping terms and conditions
    df["conditions_views"] = df["request"].str.contains("conditions-of-sale.html")
    df_feature["conditions_views"] = df.groupby("session_id")["conditions_views"].sum()

    # Number of views of the website’s home page
    homepage_regex = "^\/[a-z]{2}\/$"
    df["homepage_views"] = df["request"].str.contains(homepage_regex)
    df_feature["homepage_views"] = df.groupby("session_id")["homepage_views"].sum()

    return df_feature


def extract_marketing_features(df: pd.DataFrame) -> pd.DataFrame:
    carrello = ["cart.html", "carrello.html", "warenkorb.html", "panier.html", "cesta.html"]
    cart_actions = ["aggiungi_al_carrello.php"]
    login_operations = ["action=login", "msg=login-error"]
    internal_search_engine = ["search.html\?q", "search-by-bike.html\?"]
    source = ["utm_source"]
    checkout = ["checkout.php"]

    # TODO: Number of views of pages informing about the store and the trading company -> Non presente
    # TODO: Number of views of pages with entertainment contents -> Non presente
    # TODO: Number of other page views -> Da chiarire

    # Number of login operations (including “Register success” and “Login success”)
    df["login_actions"] = df["request"].str.contains("|".join(login_operations))
    df_feature = df.groupby("session_id")["login_actions"].sum().reset_index()

    # Number of searches using the internal search engine
    df["internal_search"] = df["request"].str.contains("|".join(internal_search_engine))
    df_feature["internal_search"] = df.groupby("session_id")["internal_search"].sum().fillna(0)

    # Number of operations of adding a product to the shopping cart
    df["add_to_cart"] = df["request"].str.contains("|".join(cart_actions))
    df_feature["add_to_cart"] = df.groupby("session_id")["add_to_cart"].sum().fillna(0)

    # Whether a “source” of the session is specified
    df["has_source"] = df["request"].str.contains("|".join(source))
    df_feature["has_source"] = df.groupby("session_id")["has_source"].sum() > 0

    # TODO: Whether the session ended with a purchase
    # braintree.php?lang = it & payment = yes

    # Number of views of product description pages -> currency=[a-z A-Z]{3}&size=[0-9 A-Z]{1,3}
    df["product_views"] = df["request"].str.contains("currency=[a-z A-Z]{3}&size=[0-9 A-Z]{1,3}")
    df_feature["product_views"] = df.groupby("session_id")["product_views"].sum()

    # Number of views of the page with shipping terms and conditions
    df["conditions_views"] = df["request"].str.contains("conditions-of-sale.html")
    df_feature["conditions_views"] = df.groupby("session_id")["conditions_views"].sum()

    # Number of views of the website’s home page
    homepage_regex = "^\/[a-z]{2}\/$"
    df["homepage_views"] = df["request"].str.contains(homepage_regex)
    df_feature["homepage_views"] = df.groupby("session_id")["homepage_views"].sum()

    return df_feature


