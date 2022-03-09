import re
import logging
import numpy as np
import pandas as pd


def parse_raw_log_data(input_path: str, output_path: str, verbose: bool = True) -> None:

    raw_data = pd.read_csv(input_path, index_col=False).values

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

