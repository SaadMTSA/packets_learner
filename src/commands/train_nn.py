import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import click
from functools import partial
import logging as LOGGER
from data import (
    create_directory,
    split_data,
    preprocess_netflow_data,
    preprocess_pcap_data,
    prepare_netflow_sequantial_data,
    prepare_pcap_sequantial_data,
)

from models import RNNModel, LSTMModel, GRUModel

LOGGER.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=LOGGER.INFO)
click.option = partial(click.option, show_default=True)


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("label_column")
@click.argument("output_directory")
@click.option(
    "--packet_type", default="netflow", type=click.Choice(["netflow", "pcap"])
)
@click.option("--test_set_size", default=0.3, type=click.FLOAT)
@click.option("--random_seed", default=2018, type=click.INT)
@click.option(
    "-m",
    "model_list",
    multiple=True,
    default=["rnn", "gru", "lstm"],
    help="List of models to train",
)
@click.option("--rnn_seq", default=10, type=click.INT)
@click.option("--epochs", default=10, type=click.INT)
@click.option("--batch_size", default=32, type=click.INT)
@click.option("--fast", "fast", flag_value=True, default=False)
def train_rnn_models(
    data_file,
    label_column,
    output_directory,
    packet_type,
    test_set_size,
    random_seed,
    model_list,
    rnn_seq,
    epochs,
    batch_size,
    fast,
):
    """ Trains multiple machine learning models for a specific datatable.
    """
    output_directory = create_directory(output_directory)
    np.random.seed(random_seed)
    LOGGER.info(f"Reading {data_file} ...")

    data = pd.read_csv(data_file)
    if packet_type == "netflow":
        data = preprocess_netflow_data(data, label_column)
    else:
        data = preprocess_pcap_data(data, label_column)

    LOGGER.info(f"Read {len(data)} records")
    LOGGER.info(f"Preparing training and testing data ...")
    
    if packet_type == 'netflow':
        x, y = prepare_netflow_sequantial_data(data, rnn_seq)
    else:
        x, y = prepare_pcap_sequantial_data(data, rnn_seq)
    x_tr, x_te, y_tr, y_te = split_data(x, y, test_set_size, random_seed)

    mapper = {
        "rnn": RNNModel,
        "lstm": LSTMModel,
        "gru": GRUModel,
    }

    for model_name in model_list:
        LOGGER.info(f"Instantiating {model_name} ...")

        if model_name not in mapper:
            LOGGER.warning(f"Model {model_name} is unknown. Skipping it!")
            continue

        model = mapper[model_name](x.shape[1:])
        LOGGER.info(model.model.summary())

        LOGGER.info(f"Fitting {model_name} to the train set ...")
        model.fit(x_tr, y_tr, x_te, y_te, epochs, batch_size)

        LOGGER.info(f"Evaluating {model_name} on the test set ...")
        model.evaluate(x_te, np.argmax(y_te, axis=-1), output_directory)

        LOGGER.info(f"Saving {model_name} to pickle file ...")
        model.pickle_model(output_directory)

    LOGGER.info("Done training all the models.")