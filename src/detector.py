import click
from functools import partial

from data.pcap_processor import process_pcap_to_csv
from data.netflow_aggregator import write_netflow_agg_scenario
from data.pcap_aggregator import write_pcap_agg_scenario
from train import train_models
from train_nn import train_rnn_models
from compare_evaluations import compare_evaluations

click.option = partial(click.option, show_default=True)

@click.group()
def cli():
    pass

cli.add_command(process_pcap_to_csv, 'process-pcap')
cli.add_command(write_netflow_agg_scenario, 'aggregate-netflow')
cli.add_command(write_pcap_agg_scenario, 'aggregate-pcap')
cli.add_command(train_models, 'train-models')
cli.add_command(train_rnn_models, 'train-rnn-models')
cli.add_command(compare_evaluations, 'compare-evaluations')

if __name__ == "__main__":
    cli()