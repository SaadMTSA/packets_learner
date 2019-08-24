import pandas as pd
import numpy as np
import os
import click
from pathlib import Path
from tqdm import tqdm

PCAP_DIR = "/data/saed/network/data/pcap_files_csv/"
PCAP_SPAM_FILES = [
    os.path.join(PCAP_DIR, "capture201108%s.parquet" % i)
    for i in ["10", "11", "15-2", "17", "15-3"]
]
PCAP_DDOS_FILES = [
    os.path.join(PCAP_DIR, "capture201108%s.parquet" % i) for i in ["15", "18", "18-2"]
]
PCAP_IRS_FILES = [
    os.path.join(PCAP_DIR, "capture201108%s.parquet" % i)
    for i in ["10", "11", "12", "15", "17", "18", "18-2"]
]

MALICIOUS = set(
    [
        "147.32.84.165",  # 15
        "147.32.84.165",
        "147.32.84.191",
        "147.32.84.192",  # 18-2
        "147.32.84.165",
        "147.32.84.191",
        "147.32.84.192",
        "147.32.84.193",
        "147.32.84.204",
        "147.32.84.205",
        "147.32.84.206",
        "147.32.84.207",
        "147.32.84.208",
        "147.32.84.209",
    ]
)  # 18

NORMAL = set(
    [
        "147.32.84.170",
        "147.32.84.134",
        "147.32.84.164",
        "147.32.87.36",
        "147.32.80.9",
        "147.32.87.11",  # 15
        "147.32.84.170",
        "147.32.84.134",
        "147.32.84.164",
        "147.32.87.36",
        "147.32.80.9",
        "147.32.87.11",  # 18-2
        "147.32.84.170",
        "147.32.84.134",
        "147.32.84.164",
        "147.32.87.36",
        "147.32.80.9",
        "147.32.87.11",
    ]
)  # 18


@click.command()
@click.argument("scenario")
@click.argument("window_size", type=click.FLOAT)
@click.argument("output_file")
def write_pcap_agg_scenario(scenario, window_size, output_file):
    """
    Aggregates pcap data based upon a scenario and a window size.
    """
    output_file = Path(output_file)
    if output_file.suffix == ".csv":
        features_df = generate_pcap_agg_scenario(scenario, window_size)
        features_df.to_csv(output_file, index=False)
    elif output_file.suffix == ".parquet":
        features_df = generate_pcap_agg_scenario(scenario, window_size)
        feature_df.to_parquet(
            output_file,
            index=False,
            engine="pyarrow",
            version="2.0",
            compression="ZSTD",
        )
    raise TypeError(f"Unsupported file type {output_file.suffix}")


def generate_pcap_agg_scenario(scenario, window_size):
    mapper = {"spam": PCAP_SPAM_FILES, "ddos": PCAP_DDOS_FILES, "irs": PCAP_IRS_FILES}
    if scenario not in mapper:
        raise ValueError(f"Unknown type '{scenario}'.")

    features_df = [
        _aggregate_pcap(pd.read_parquet(i), window_size)
        for i in mapper[scenario]
    ]

    features_df = pd.concat(features_df)
    return features_df


def _aggregate_pcap(packets, window_size):
    packets["time"] = packets.time - packets.time.min()
    packets["start_window"] = packets.time // window_size
    first_window = int(packets["start_window"].min())
    last_window = int(packets["start_window"].max())
    results = []
    for window in tqdm(range(first_window, last_window + 1)):
        window_df = packets[packets["start_window"] == window]
        results.append(_aggregate_pcap_packets(window_df))

    return results


def _aggregate_pcap_packets(packets):
    fs = {}
    fs["n_packets"] = len(packets)
    fs["n_abnormal"] = len([0 for i in packets.itertuples(index=False) if i[14] in MALICIOUS])
    fs["n_normal"] = len([0 for i in packets.itertuples(index=False) if i[14] in NORMAL])
    fs["n_background"] = fs["n_packets"] - fs["n_abnormal"] - fs["n_normal"]

    ip_src = [[], [], [], []]
    for i in packets.itertuples(index=False):
        ip = int(i[14].split(".")[0])
        if ip <= 126:
            ip_src[0].append(i[14])
        elif ip <= 191:
            ip_src[1].append(i[14])
        elif ip <= 223:
            ip_src[2].append(i[14])
        else:
            ip_src[3].append(i[14])
    fs["n_src_ipclass_a"] = len(ip_src[0])
    fs["n_src_ipclass_b"] = len(ip_src[1])
    fs["n_src_ipclass_c"] = len(ip_src[2])
    fs["n_src_ipclass_na"] = len(ip_src[3])

    ip_dst = [[], [], [], []]
    for i in packets.itertuples(index=False):
        ip = int(i[15].split(".")[0])
        if ip <= 126:
            ip_dst[0].append(i[15])
        elif ip <= 191:
            ip_dst[1].append(i[15])
        elif ip <= 223:
            ip_dst[2].append(i[15])
        else:
            ip_dst[3].append(i[15])
    fs["n_dst_ipclass_a"] = len(ip_dst[0])
    fs["n_dst_ipclass_b"] = len(ip_dst[1])
    fs["n_dst_ipclass_c"] = len(ip_dst[2])
    fs["n_dst_ipclass_na"] = len(ip_dst[3])

    port_src = [[], []]
    for i in packets.itertuples(index=False):
        src_port = int(i[17])
        prot = int(i[12])
        if prot == 1:
            continue
        if src_port >= 1024:
            port_src[0].append(src_port)
        else:
            port_src[1].append(src_port)
    fs["n_src_port>=1024"] = len(port_src[0])
    fs["n_src_port<1024"] = len(port_src[1])

    port_dst = [[], []]
    for i in packets.itertuples(index=False):
        dst_port = int(i[18])
        prot = int(i[12])
        if prot == 1:
            continue
        if dst_port >= 1024:
            port_dst[0].append(dst_port)
        else:
            port_dst[1].append(dst_port)
    fs["n_dst_port>=1024"] = len(port_dst[0])
    fs["n_dst_port<1024"] = len(port_dst[1])

    fs["n_tcp"] = len([0 for i in packets.itertuples(index=False) if int(i[12]) == 6])
    fs["n_udp"] = len([0 for i in packets.itertuples(index=False) if int(i[12]) == 17])
    fs["n_icmp"] = len([0 for i in packets.itertuples(index=False) if int(i[12]) == 1])

    src_ips = pd.Series(np.array([j for i in ip_src for j in i]))
    fs["ent_src_ip"] = entropy(src_ips.value_counts().values)

    dst_ips = pd.Series(np.array([j for i in ip_dst for j in i]))
    fs["ent_dst_ip"] = entropy(dst_ips.value_counts().values)

    src_ips = pd.Series(ip_src[0])
    fs["ent_src_ip_a"] = entropy(src_ips.value_counts().values)

    src_ips = pd.Series(ip_src[1])
    fs["ent_src_ip_b"] = entropy(src_ips.value_counts().values)

    src_ips = pd.Series(ip_src[2])
    fs["ent_src_ip_c"] = entropy(src_ips.value_counts().values)

    src_ips = pd.Series(ip_src[3])
    fs["ent_src_ip_na"] = entropy(src_ips.value_counts().values)

    dst_ips = pd.Series(ip_dst[0])
    fs["ent_dst_ip_a"] = entropy(dst_ips.value_counts().values)

    dst_ips = pd.Series(ip_dst[1])
    fs["ent_dst_ip_b"] = entropy(dst_ips.value_counts().values)

    dst_ips = pd.Series(ip_dst[2])
    fs["ent_dst_ip_c"] = entropy(dst_ips.value_counts().values)

    dst_ips = pd.Series(ip_dst[3])
    fs["ent_dst_ip_na"] = entropy(dst_ips.value_counts().values)

    src_ports = pd.Series(np.array([j for i in port_src for j in i]))
    fs["ent_src_port"] = entropy(src_ports.value_counts())

    dst_ports = pd.Series(np.array([j for i in port_dst for j in i]))
    fs["ent_dst_port"] = entropy(dst_ports.value_counts())

    src_ports = pd.Series(port_src[0])
    fs["ent_src_port>=1024"] = entropy(src_ports.value_counts())

    src_ports = pd.Series(port_src[1])
    fs["ent_src_port<1024"] = entropy(src_ports.value_counts())

    dst_ports = pd.Series(port_dst[0])
    fs["ent_dst_port>=1024"] = entropy(dst_ports.value_counts())

    dst_ports = pd.Series(port_dst[1])
    fs["ent_dst_port<1024"] = entropy(dst_ports.value_counts())

    return fs