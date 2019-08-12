import pandas as pd
import numpy as np
import click
from glob import glob
import logging as LOGGER

from data import create_directory

LOGGER.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=LOGGER.INFO)

def get_files(glob_):
    res = glob(glob_)
    if res:
        return res
    raise ValueError(f"No matches for {glob_}")

def read_multiple_csvs(files):
    dfs = []
    for i in files:
        LOGGER.info(f"Reading {i} ...")
        dfs.append(pd.read_csv(i, index_col=0))
    LOGGER.info("Concatenating metrics ...")
    return pd.concat(dfs)

@click.command()
@click.argument('glob')
@click.argument('output_directory')
def compare_evaluations(glob, output_directory):
    LOGGER.info("Reading metric files ...")
    metrics = read_multiple_csvs(get_files(glob))
    LOGGER.info("Preparing output directory ...")
    output_directory = create_directory(output_directory)
    output_file = output_directory / 'combined_metrics.csv'
    metrics.to_csv(output_file)
    LOGGER.info(f"Output saved to {output_file}")
