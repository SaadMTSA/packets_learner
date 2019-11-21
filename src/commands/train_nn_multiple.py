import click
from .train_nn import train_rnn_models
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
@click.option("--test_set_size", default=0.3, type=click.FLOAT)
@click.option(
    "-m",
    "model_list",
    multiple=True,
    default=["rnn", "gru", "lstm"],
    help="List of models to train",
)
@click.option("-r", "--rnn_seq", default=(1, 10, 1), type=click.Tuple([int, int, int]))
@click.option("-e", "--epochs", default=10, type=click.INT)
@click.option("-b", "--batch_size", default=32, type=click.INT)
@click.option("-f", "--forward_predict", default=(1, 10, 1), type=click.Tuple([int, int, int]))
@click.option("-s", "--standardize", flag_value=True, default=False)
@click.option("-p", "--poly", flag_value=True, default=False)
@click.option("--transition", "transition", type=click.INT, default=0)
def train_multiple(
    ctx,
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
    forward_predict,
    standardize,
    poly,
    transition,
):
    for r in range(rnn_seq[0], rnn_seq[1]+1, rnn_seq[2]):
        for f in range(forward_predict[0], forward_predict[1]+1, forward_predict[2]):
            LOGGER.info(f"Invoking train_rnn_models with sequence_length={r} & forward_predict={f}")
            ctx.invoke(
                train_rnn_models,
                data_file=data_file,
                label_column=label_column,
                output_directory=output_directory,
                packet_type=packet_type,
                test_set_size=test_set_size,
                random_seed=random_seed,
                model_list=model_list,
                rnn_seq=r,
                epochs=epochs,
                batch_size=batch_size,
                forward_predict=f,
                standardize=standardize,
                poly=poly,
                transition=transition,
            )