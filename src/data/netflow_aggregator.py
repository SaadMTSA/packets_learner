import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import click
from scipy.stats import entropy
from pathlib import Path
import ray
import logging as LOGGER

LOGGER.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=LOGGER.INFO)

NETFLOW_DIR = "/data/saed/network/data/netflow_files_csv/"
NETFLOW_SPAM_FILES = [
    os.path.join(NETFLOW_DIR, "capture201108%s.binetflow" % i)
    for i in ["10", "11", "15-2", "17", "15-3"]
]
NETFLOW_DDOS_FILES = [
    os.path.join(NETFLOW_DIR, "capture201108%s.binetflow" % i)
    for i in ["15", "18", "18-2"]
]
NETFLOW_IRS_FILES = [
    os.path.join(NETFLOW_DIR, "capture201108%s.binetflow" % i)
    for i in ["10", "11", "12", "15", "17", "18", "18-2"]
]



@click.command()
@click.argument("scenario")
@click.argument("window_size", type=click.FLOAT)
@click.argument("output_file")
def write_netflow_agg_scenario(scenario, window_size, output_file):
    """
    Aggregates netflow data based upon a scenario and a window size.
    """
    output_file = Path(output_file)
    if output_file.suffix == ".csv":
        LOGGER.info(f"Generating aggregated features for {scenario} scenario ...")
        features_df = generate_netflow_agg_scenario(scenario, window_size)
        LOGGER.info(f"Writing aggregated features to {output_file} ...")
        features_df.to_csv(output_file, index=False)
    elif output_file.suffix == ".parquet":
        LOGGER.info(f"Generating aggregated features for {scenario} scenario ...")
        features_df = generate_netflow_agg_scenario(scenario, window_size)
        LOGGER.info(f"Writing aggregated features to {output_file} ...")
        feature_df.to_parquet(
            output_file,
            index=False,
            engine="pyarrow",
            version="2.0",
            compression="ZSTD",
        )
    else:
        raise TypeError(f"Unsupported file type {output_file.suffix}")
    LOGGER.info("Successfully done!")


def generate_netflow_agg_scenario(scenario, window_size):
    LOGGER.info(f"Reading {scenario} files ...")
    mapper = {
        "spam": NETFLOW_SPAM_FILES,
        "ddos": NETFLOW_DDOS_FILES,
        "irs": NETFLOW_IRS_FILES,
    }
    if scenario not in mapper:
        raise ValueError(f"Unknown type '{scenario}'.")
    features_df = []
    for i in mapper[scenario]:
        LOGGER.info(f"Reading, Processing, and Aggregating {i} ...")
        features_df.append(
            _aggregate_netflow(pd.read_csv(i, parse_dates=[0]), window_size)
        )
    LOGGER.info(f"Concatenating results ...")
    features_df = pd.concat(features_df)
    return features_df


def _classify_ip(value):
    if ":" in value:
        return 3
    cls = int(value.split(".")[0])
    if cls < 127:
        return 0
    elif cls < 192:
        return 1
    elif cls < 223:
        return 2
    else:
        return 3


def _aggregate_netflow(netflow, window_size):
    LOGGER.info("Cleaning data ...")
    netflow["StartTime"] = (
        netflow.StartTime - netflow.StartTime.min()
    ).dt.total_seconds()
    netflow["StartWindow"] = netflow.StartTime // window_size
    netflow["EndWindow"] = (netflow.StartTime + netflow.Dur) // window_size
    netflow["FirstDur"] = netflow.apply(
        lambda row: row["Dur"]
        if row["StartWindow"] == row["EndWindow"]
        else (row["StartWindow"] + 1) * window_size - row["StartTime"],
        axis=1,
    )
    netflow["LastDur"] = netflow.apply(
        lambda row: row["Dur"]
        if row["StartWindow"] == row["EndWindow"]
        else row["EndWindow"] * window_size - row["StartTime"] + row["Dur"],
        axis=1,
    )
    netflow["Label"] = netflow.Label.apply(
        lambda x: -1 if "background" in x.lower() else 1 if "botnet" in x.lower() else 0
    )
    netflow["SrcIPClass"] = netflow.SrcAddr.apply(_classify_ip)
    netflow["DstIPClass"] = netflow.DstAddr.apply(_classify_ip)
    netflow["Sport"] = netflow.Sport.apply(lambda x: int(x, 0) if x == "nan" else 0)
    netflow["Dport"] = netflow.Dport.apply(lambda x: int(x, 0) if x == "nan" else 0)
    firstWindow = int(netflow.StartWindow.min())
    lastWindow = int(netflow.EndWindow.max())
    LOGGER.info("Aggregating data ...")
    features_calculated = []
    in_parallel = 40
    results_ids = []
    for window in tqdm(range(firstWindow, lastWindow + 1), unit="window"):
        results_ids.append(
            _aggregate_window(
                netflow[
                    (netflow.StartWindow <= window) & (netflow.EndWindow >= window)
                ],
                window,
                window_size,
            )
        )
    return pd.DataFrame(features_calculated)


def _aggregate_window(flow, window, window_size):
    features = {}
    flow.loc[:, "isLast"] = flow.EndWindow == window
    flow.loc[:, "CurrDur"] = flow.apply(
        lambda x: x["FirstDur"]
        if x["StartWindow"] == window
        else x["LastDur"]
        if x["isLast"]
        else window_size,
        axis=1,
    )
    features["n_conn"] = len(flow)
    features["n_normal"] = len(flow[flow.Label == 0])
    features["n_background"] = len(flow[flow.Label == -1])
    features["m_duration"] = flow.CurrDur.mean()

    src_class_cnt = flow.SrcIPClass.value_counts()
    features["n_src_ip_a"] = src_class_cnt[0] if 0 in src_class_cnt else 0
    features["n_src_ip_b"] = src_class_cnt[1] if 1 in src_class_cnt else 0
    features["n_src_ip_c"] = src_class_cnt[2] if 2 in src_class_cnt else 0
    features["n_src_ip_na"] = src_class_cnt[3] if 3 in src_class_cnt else 0

    dst_class_cnt = flow.DstIPClass.value_counts()
    features["n_dst_ip_a"] = dst_class_cnt[0] if 0 in dst_class_cnt else 0
    features["n_dst_ip_b"] = dst_class_cnt[1] if 1 in dst_class_cnt else 0
    features["n_dst_ip_c"] = dst_class_cnt[2] if 2 in dst_class_cnt else 0
    features["n_dst_ip_na"] = dst_class_cnt[3] if 3 in dst_class_cnt else 0

    src_port_mask = flow.Sport >= 1024
    src_port_cnt = src_port_mask.value_counts()
    features["n_src_port>=1024"] = (
        src_port_cnt[True] if True in src_port_cnt.index else 0
    )
    features["n_src_port<1024"] = (
        src_port_cnt[False] if False in src_port_cnt.index else 0
    )
    dst_port_mask = flow.Dport >= 1024
    dst_port_cnt = dst_port_mask.value_counts()
    features["n_dst_port>=1024"] = (
        dst_port_cnt[True] if True in dst_port_cnt.index else 0
    )
    features["n_dst_port<1024"] = (
        dst_port_cnt[False] if False in dst_port_cnt.index else 0
    )

    proto_cnt = flow.Proto.value_counts()
    features["n_icmp"] = proto_cnt["icmp"] if "icmp" in proto_cnt.index else 0
    features["n_tcp"] = proto_cnt["tcp"] if "tcp" in proto_cnt.index else 0
    features["n_udp"] = proto_cnt["udp"] if "udp" in proto_cnt.index else 0

    features["s_packets"] = entropy(flow.TotPkts[flow.isLast].value_counts())
    features["s_srcbytes"] = entropy(flow.SrcBytes[flow.isLast].value_counts())
    features["s_totbytes"] = entropy(flow.TotBytes[flow.isLast].value_counts())
    features["s_state"] = entropy(
        flow.State.value_counts()
    )  ## is that a data leak? can it be known at the beginning? if not, remove it for alive connections
    features["s_duration"] = entropy(flow.Dur[flow.isLast].value_counts())
    features["sd_packets"] = flow.TotPkts[flow.isLast].std()
    features["sd_srcbytes"] = flow.SrcBytes[flow.isLast].std()
    features["sd_totbytes"] = flow.TotBytes[flow.isLast].std()
    features["sd_duration"] = flow.Dur[flow.isLast].std()

    features["s_srcip"] = entropy(flow.SrcAddr.value_counts())
    features["s_dstip"] = entropy(flow.DstAddr.value_counts())

    features["s_src_ip_a"] = entropy(flow.SrcAddr[flow.SrcIPClass == 0].value_counts())
    features["s_src_ip_b"] = entropy(flow.SrcAddr[flow.SrcIPClass == 1].value_counts())
    features["s_src_ip_c"] = entropy(flow.SrcAddr[flow.SrcIPClass == 2].value_counts())
    features["s_src_ip_d"] = entropy(flow.SrcAddr[flow.SrcIPClass == 3].value_counts())

    features["s_dst_ip_a"] = entropy(flow.DstAddr[flow.DstIPClass == 0].value_counts())
    features["s_dst_ip_b"] = entropy(flow.DstAddr[flow.DstIPClass == 1].value_counts())
    features["s_dst_ip_c"] = entropy(flow.DstAddr[flow.DstIPClass == 2].value_counts())
    features["s_dst_ip_d"] = entropy(flow.DstAddr[flow.DstIPClass == 3].value_counts())

    features["s_src_port>=1024"] = entropy(flow.Sport[src_port_mask].value_counts())
    features["s_src_port<1024"] = entropy(flow.Sport[~src_port_mask].value_counts())
    features["s_dst_port>=1024"] = entropy(flow.Dport[dst_port_mask].value_counts())
    features["s_dst_port<1024"] = entropy(flow.Dport[~dst_port_mask].value_counts())
    return features