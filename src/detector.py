import click
from functools import partial
import warnings

from .data.pcap_processor import process_pcap_to_csv
from .data.netflow_aggregator import write_netflow_agg_scenario
from .data.pcap_aggregator import write_pcap_agg_scenario
from .commands.train import train_models
from .commands.train_nn import train_rnn_models
from .commands.compare_evaluations import compare_evaluations
from .commands.train_nn_multiple import train_multiple


click.option = partial(click.option, show_default=True)

@click.group()
def cli():
    pass

cli.add_command(process_pcap_to_csv, 'process-pcap')
cli.add_command(write_netflow_agg_scenario, 'aggregate-netflow')
cli.add_command(write_pcap_agg_scenario, 'aggregate-pcap')
cli.add_command(train_models, 'train-models')
cli.add_command(train_rnn_models, 'train-rnn-models')
cli.add_command(train_multiple, 'train-rnn-multiple')
cli.add_command(compare_evaluations, 'compare-evaluations')

if __name__ == "__main__":
    cli()