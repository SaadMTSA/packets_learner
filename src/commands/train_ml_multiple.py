import click
from .train import train_models
import logging as LOGGER

LOGGER.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=LOGGER.INFO)

@click.command()
@click.pass_context


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
@click.option("-p", "--predict", default=(1, 10), type=click.Tuple([int, int, int]))
@click.option("--transition", default=0)
@click.option("--save", "save", flag_value=True, default=False)
def train_multiple(
    ctx,
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
        for f in range(predict[0], predict[1]+1, predict[2]):
            LOGGER.info(f"Invoking train_models with forward_predict={f}")
            ctx.invoke(
                train_models,
                data_file=data_file,
                label_column=label_column,
                output_directory=output_directory,
                packet_type=packet_type,
                test_set_size=test_set_size,
                random_seed=random_seed,
                model_list=model_list,
                fast=fast,
                predict=f,
                transition=transition,
                save=save,
            )