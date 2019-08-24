import pandas as pd
import numpy as np
import click
import logging as LOGGER
import seaborn as sns
import matplotlib.pyplot as plt

from glob import glob
from matplotlib.backends.backend_pdf import PdfPages

from src.data.data import create_directory

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
        dfs.append(pd.read_csv(i))
    LOGGER.info("Concatenating metrics ...")
    return pd.concat(dfs)

def setup_metrics_table(metrics):
    metrics = metrics.copy()
    metrics['network'] = metrics.model.apply(lambda x: x.split('_')[1])
    metrics['seq_len'] = metrics.model.apply(lambda x: int(x.split('_')[2].split('seq-len')[0]))
    metrics['steps'] = metrics.model.apply(lambda x: int(x.split('_')[3].split('steps')[0]))
    metrics.sort_values(['network', 'seq_len', 'steps'])
    return metrics

def _plot_nn(metrics, model, values, pdf):
    data = metrics[metrics['model'].str.contains(model)]
    if len(data) > 0:
        for val in values:
            neg_log_loss = data[['seq_len', 'steps', val]].pivot(index='seq_len', columns='steps', values=val)
            plt.title(f"{model.upper()} - {val.upper()}")
            sns.heatmap(neg_log_loss)
            pdf.savefig()
            plt.close()
        
def plot_nn_comps(metrics, pdf):
    _plot_nn(metrics, 'gru', ['log_loss', 'f1', 'precision', 'recall', 'accuracy', 'average_precision'], pdf)
    _plot_nn(metrics, 'lstm', ['log_loss', 'f1', 'precision', 'recall', 'accuracy', 'average_precision'], pdf)
    _plot_nn(metrics, 'rnn', ['log_loss', 'f1', 'precision', 'recall', 'accuracy', 'average_precision'], pdf)
    _plot_nn(metrics, 'cstm1', ['log_loss', 'f1', 'precision', 'recall', 'accuracy', 'average_precision'], pdf)
    
def _plot_ml(metrics, model, values, pdf):
    raise NotImplementedError()
    
def plot_ml_comps(metrics, pdf):
    raise NotImplementedError()
    
@click.command()
@click.argument('glob')
@click.argument('output_directory')
@click.option('--kind', type=click.Choice(['dl', 'ml']), default='dl')
def compare_evaluations(glob, output_directory, kind):
    output_directory = create_directory(output_directory)
    
    LOGGER.info("Reading metric files ...")
    metrics = read_multiple_csvs(get_files(glob))
    
    LOGGER.info("Preparing metrics table ...")
    metrics = setup_metrics_table(metrics)
    
    LOGGER.info("Plotting comparative plots ...")
    with PdfPages(output_directory / 'comparative_plots.pdf') as pdf:
        if kind == 'dl':
            plot_nn_comps(metrics, pdf)
        else:
            plot_ml_comps(metrics, pdf)
    
    LOGGER.info("Preparing output directory ...")
    output_file = output_directory / 'combined_metrics.csv'
    metrics.to_csv(output_file, index=False)
    LOGGER.info(f"Output saved to {output_file}")
