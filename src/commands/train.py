import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import click
from functools import partial
import logging as LOGGER
from src.data.data import (
    create_directory,
    split_data,
    preprocess_netflow_data,
    preprocess_pcap_data,
)

from src.models import RandomForestModel, XGBoostModel, LogisticRegressionModel, RNNModel

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
    default=["random_forest", "xgboost", "logistic_regression"],
    help="List of models to train",
)
@click.option("--fast", "fast", flag_value=True, default=False)
@click.option("--predict", "predict", type=click.INT, default=0)
@click.option("--transition", "transition", type=click.INT, default=0)
@click.option("--save", "save", flag_value=True, default=False)
def train_models(
    data_file,
    label_column,
    output_directory,
    packet_type,
    test_set_size,
    random_seed,
    model_list,
    fast,
    predict,
    transition,
    save,
):
    """ Trains multiple machine learning models for a specific datatable.
    """
    output_directory = create_directory(output_directory)
    np.random.seed(random_seed)
    LOGGER.info(f"Reading {data_file} ...")
    if data_file.endswith('parquet'):
        data = pd.read_parquet(data_file).dropna(how='all')
    elif data_file.endswith('csv'):
        data = pd.read_csv(data_file).dropna(how='all')
    else:
        raise NotImplementedError(f'Extension of {datafile} is not supported')
        
    if packet_type == "netflow":
        data = preprocess_netflow_data(data, label_column, transition)
    else:
        data = preprocess_pcap_data(data, label_column)
    
    LOGGER.info(f"Read {len(data)} records")
    LOGGER.info(f"Preparing training and testing data ...")
    
    if predict != 0:
        x, y = data.iloc[:-predict, :-1], data.iloc[predict:, -1]
    else:
        x, y = data.iloc[:, :-1], data.iloc[:, -1]
    
    x_tr, x_te, y_tr, y_te = split_data(x.values, y.values, test_set_size, random_seed, True if transition == 2 else False)
    mapper = {
        "random_forest": RandomForestModel,
        "xgboost": XGBoostModel,
        "logistic_regression": LogisticRegressionModel,
    }

    for model_name in model_list:
        LOGGER.info(f"Instantiating {model_name} ...")

        if model_name not in mapper:
            LOGGER.warning(f"Model {model_name} is unknown. Skipping it!")
            continue

        model = mapper[model_name]

        model = model(fast=fast, random_state=random_seed)

        LOGGER.info(f"Fitting {model_name} to the train set ...")
        model.fit(x_tr, y_tr)

        cur_scenario = f"{packet_type}_{model_name}_{'fast' if fast else 'no-fast'}_{predict}steps_{'transition' if transition else 'binary'}"
        cur_output_dir = create_directory(output_directory / cur_scenario)
        
        LOGGER.info(f"Evaluating {model_name} on the test set ...")
        model.evaluate(x_te, y_te, cur_output_dir, data.columns[:-1], transition=transition)

        if save:
            LOGGER.info(f"Saving {model_name} to pickle file ...")
            model.pickle_model(cur_output_dir)

    LOGGER.info("Done training all the models.")