import threading
from scapy.all import *
import os
import sys
import pandas as pd
import csv
import time
from tqdm import tqdm
import click
import numpy as np

header = [
    "time",
    "ethernet.dst",
    "ethernet.src",
    "ethernet.type",
    "ip.version",
    "ip.ihl",
    "ip.tos",
    "ip.len",
    "ip.id",
    "ip.flags",
    "ip.frag",
    "ip.ttl",
    "ip.proto",
    "ip.chksum",
    "ip.src",
    "ip.dst",
    "ip.options",
    "udp.sport",
    "udp.dport",
    "udp.len",
    "udp.chksum",
    "tcp.sport",
    "tcp.dport",
    "tcp.seq",
    "tcp.ack",
    "tcp.dataofs",
    "tcp.reserved",
    "tcp.flags",
    "tcp.window",
    "tcp.chksum",
    "tcp.urgptr",
    "tcp.options",
    "icmp.type",
    "icmp.code",
    "icmp.chksum",
    "icmp.reserved",
    "icmp.length",
    "icmp.nexthopmtu",
    "icmp.ip.version",
    "icmp.ip.ihl",
    "icmp.ip.tos",
    "icmp.ip.len",
    "icmp.ip.id",
    "icmp.ip.flags",
    "icmp.ip.frag",
    "icmp.ip.ttl",
    "icmp.ip.proto",
    "icmp.ip.chksum",
    "icmp.ip.src",
    "icmp.ip.dst",
    "icmp.ip.options",
]


def process_udp(p):
    res = [
        p.time,
        p["Ethernet"].dst,
        p["Ethernet"].src,
        p["Ethernet"].type,
        p["IP"].version,
        p["IP"].ihl,
        p["IP"].tos,
        p["IP"].len,
        p["IP"].id,
        p["IP"].flags,
        p["IP"].frag,
        p["IP"].ttl,
        "udp",
        p["IP"].chksum,
        p["IP"].src,
        p["IP"].dst,
        p["IP"].options,
    ]
    try:
        res.extend(
            [
                p["UDP"].sport,
                p["UDP"].dport,
                p["UDP"].len,
                p["UDP"].chksum,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ]
        )
    except:
        res.extend([None] * 34)

    return res


def process_tcp(p):
    res = [
        p.time,
        p["Ethernet"].dst,
        p["Ethernet"].src,
        p["Ethernet"].type,
        p["IP"].version,
        p["IP"].ihl,
        p["IP"].tos,
        p["IP"].len,
        p["IP"].id,
        p["IP"].flags,
        p["IP"].frag,
        p["IP"].ttl,
        "tcp",
        p["IP"].chksum,
        p["IP"].src,
        p["IP"].dst,
        p["IP"].options,
        None,
        None,
        None,
        None,
        p["TCP"].sport,
        p["TCP"].dport,
        p["TCP"].seq,
        p["TCP"].ack,
        p["TCP"].dataofs,
        p["TCP"].reserved,
        p["TCP"].flags,
        p["TCP"].window,
        p["TCP"].chksum,
        p["TCP"].urgptr,
        p["TCP"].options,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    return res


def process_icmp(p):
    res = [
        p.time,
        p["Ethernet"].dst,
        p["Ethernet"].src,
        p["Ethernet"].type,
        p["IP"].version,
        p["IP"].ihl,
        p["IP"].tos,
        p["IP"].len,
        p["IP"].id,
        p["IP"].flags,
        p["IP"].frag,
        p["IP"].ttl,
        "icmp",
        p["IP"].chksum,
        p["IP"].src,
        p["IP"].dst,
        p["IP"].options,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    try:
        res += [
            p["ICMP"].type,
            p["ICMP"].code,
            p["ICMP"].chksum,
            p["ICMP"].reserved,
            p["ICMP"].length,
            p["ICMP"].nexthopmtu,
            p["IP in ICMP"].version,
            p["IP in ICMP"].ihl,
            p["IP in ICMP"].tos,
            p["IP in ICMP"].len,
            p["IP in ICMP"].id,
            p["IP in ICMP"].flags,
            p["IP in ICMP"].frag,
            p["IP in ICMP"].ttl,
            p["IP in ICMP"].proto,
            p["IP in ICMP"].chksum,
            p["IP in ICMP"].src,
            p["IP in ICMP"].dst,
            p["IP in ICMP"].options,
        ]
    except IndexError:
        res += [None] * 15
    return res


def process_packet(packet):
    try:
        proto = packet["IP"].proto
        if proto == 6:
            res = process_tcp(packet)
        elif proto == 17:
            res = process_udp(packet)
        elif proto == 1:
            res = process_icmp(packet)

        return ",".join([str(i) for i in res]) + "\n"
    except:
        return None

@click.command()
@click.argument("in_pcap", type=click.Path(exists=True))
@click.argument("out_pcap")
@click.option("--write_each", default=1000)
def process_pcap_to_csv(in_pcap, out_pcap, write_each):
    """
    Converts pcap data to tabular csv data that can be aggregated later on.
    """
    print("Reading " + in_pcap)
    with open(out_pcap, "w") as file:
        file.writelines(",".join(header) + "\n")
        cnt = 0
        to_write = [None] * count
        for p in tqdm(PcapReader(in_pcap), unit="packets"):
            to_write[cnt] = process_packet(p)
            if to_write[cnt] is None:
                continue
            cnt += 1
            if cnt % count == 0:
                file.writelines(to_write)
                to_write = [None] * count
                cnt = 0
    print("Done " + in_pcap)