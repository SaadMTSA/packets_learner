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

def setup_metrics_table(metrics, kind):
    metrics = metrics.copy()
    if kind == 'dl':
        metrics['network'] = metrics.model.apply(lambda x: x.split('_')[1])
        metrics['seq_len'] = metrics.model.apply(lambda x: int(x.split('_')[2].split('seq-len')[0]))
        metrics['steps'] = metrics.model.apply(lambda x: int(x.split('_')[3].split('steps')[0]))
        metrics.sort_values(['network', 'seq_len', 'steps'])
    elif kind == 'ml':
        metrics.model = metrics.model.apply(lambda x: x.replace('random_', 'random').replace('logistic_', 'logistic'))
        metrics['ml_model'] = metrics.model.apply(lambda x: x.split('_')[1])
        metrics['steps'] = metrics.model.apply(lambda x: int(x.split('_')[3].split('steps')[0]))
        metrics.sort_values(['ml_model', 'steps'])
    return metrics

def _plot_nn(data, model, values, pdf):
    if model is not None:
        data = data[data['network'].str.contains(model)]
        if len(data) > 0:
            for val in values:
#                 plot = data[['seq_len', 'steps', val]].pivot(index='seq_len', columns='steps', values=val)
#                 plt.title(f"{model.upper()} - {val.upper()}")
#                 sns.heatmap(plot)
                plot = data[['seq_len', 'steps', val]].pivot(index='seq_len', columns='steps', values=val)
                plt.title(f"{model.upper()} - {val.upper()}")
                for col in plot:
                    plt.plot(plot[col], label=f"{col} Step(s)")
                plt.xlim(plot.index.min(), plot.index.max()*1.5)
                rng = range(plot.index.min(), plot.index.max()+2, 2)
                plt.xticks(rng, rng)
                plt.xlabel('Sequence Length')
                plt.ylabel(f"{val.upper()}")
                plt.grid(True)
                plt.legend(loc=1)
                pdf.savefig()
                plt.close()
    else:
        if len(data) > 0:
            for val in values:
#                 plot = data[['seq_len', 'steps', val]].pivot(index='seq_len', columns='steps', values=val)
#                 plt.title(f"{model.upper()} - {val.upper()}")
#                 sns.heatmap(plot)
                data['myindex'] = data[['network', 'seq_len']].apply(lambda x: f"{x.network}_{x.seq_len}_", axis=1)
                plot = data[['myindex', 'steps', val]].pivot(index = 'myindex', columns='steps', values=val)
                for i in range(10):
                    plt.title(f"ALL - {val.upper()} - {i+1} Sequence Length")
                    for idx, row in plot[plot.index.str.contains(f"_{i+1}_")].iterrows():
                        plt.plot(row, label=idx)
                    plt.xlabel('Steps')
                    plt.ylabel(f"{val.upper()}")
                    plt.grid(True)
                    plt.legend(loc='upper left')
                    pdf.savefig()
                    plt.close()

def _plot_ml(data, model, values, pdf):
    if model is not None:
        data = data[data['ml_model'].str.contains(model)]
        if len(data) > 0:
            for val in values:
#                 plot = data[['seq_len', 'steps', val]].pivot(index='seq_len', columns='steps', values=val)
#                 plt.title(f"{model.upper()} - {val.upper()}")
#                 sns.heatmap(plot)
                plot = data[['steps', val]].sort_values('steps')
                plot = plot[plot['steps'] < 1_000]
                plt.title(f"{model.upper()} - {val.upper().replace('_', '-')}")
                plt.plot(plot['steps'].values, plot[val].values)
#                 plt.xlim(plot.index.min(), plot.index.max()*1.5)
#                 rng = range(plot.index.min(), plot.index.max()+2, 2)
#                 plt.xticks(rng, rng)
                plt.xlabel('Steps Ahead')
                plt.ylabel(f"{val.upper().replace('_', '-')}")
                plt.grid(True)
#                 plt.legend(loc=1)
                pdf.savefig()
                plt.close()
    else:
        if len(data) > 0:
            for val in values:
#                 plot = data[['seq_len', 'steps', val]].pivot(index='seq_len', columns='steps', values=val)
#                 plt.title(f"{model.upper()} - {val.upper()}")
#                 sns.heatmap(plot)
                data['myindex'] = data[['ml_model', 'steps']].apply(lambda x: f"{x.ml_model}_{x.seq_len}_", axis=1)
                plot = data[['myindex', val]].pivot(index = 'myindex', columns='steps', values=val)
                for i in range(10):
                    plt.title(f"ALL - {val.upper().replace('_', '-')} - {i+1} Sequence Length")
                    for idx, row in plot[plot.index.str.contains(f"_{i+1}_")].iterrows():
                        plt.plot(row, label=idx)
                    plt.xlabel('Steps')
                    plt.ylabel(f"{val.upper().replace('_', '-')}")
                    plt.grid(True)
                    plt.legend(loc='upper left')
                    pdf.savefig()
                    plt.close()
        
def plot_nn_comps(data, pdf):
    metrics =  ['log-loss', 'au-roc', 'f1', 'precision', 'recall', 'accuracy']
    _plot_nn(data, 'gru', metrics, pdf)
    _plot_nn(data, 'lstm', metrics, pdf)
    _plot_nn(data, 'rnn', metrics, pdf)
    _plot_nn(data, 'cstm1', metrics, pdf)
    _plot_nn(data, None, metrics, pdf)

def plot_ml_comps(data, pdf):
    metrics =  ['log-loss', 'au-roc', 'f1', 'precision', 'recall', 'accuracy']
    _plot_ml(data, 'logisticregression', metrics, pdf)
    _plot_ml(data, 'randomforest', metrics, pdf)
    _plot_ml(data, 'xgboost', metrics, pdf)
#     _plot_ml(data, None, metrics, pdf)
    
@click.command()
@click.argument('glob')
@click.argument('output_directory')
@click.option('--kind', type=click.Choice(['dl', 'ml']), default='dl')
def compare_evaluations(glob, output_directory, kind):
    output_directory = create_directory(output_directory)
    
    LOGGER.info("Reading metric files ...")
    metrics = read_multiple_csvs(get_files(glob))
    
    LOGGER.info("Preparing metrics table ...")
    metrics = setup_metrics_table(metrics, kind)
    
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
