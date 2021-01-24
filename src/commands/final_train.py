import matplotlib.pyplot as plt
import scikitplot as skplt
from keras.models import Sequential, Model, Input
from keras.layers import (
    Dense,
    Activation,
    Input,
    SimpleRNN,
    GRU,
    LSTM,
    concatenate,
    Bidirectional,
    TimeDistributed,
    Embedding,
)
from itertools import cycle
import numpy as np
import keras
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from skopt import BayesSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    average_precision_score,
    log_loss,
    roc_auc_score,
)
import click
from src.data.data import (
    create_directory,
)
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({"font.size": 18})
plt.rcParams['figure.dpi'] = 200

def prepare_pcap_sequantial_data(
    data, seq_len, forward_predict, standardize, poly, transition
):
    if forward_predict < 1:
        raise ValueError(f"forward_predict should be > 0, {forward_predict} was given!")

    remove = len(data) % seq_len
    preprocess_features = Pipeline(
        steps=[
            (
                "ExtractFeatures",
                FunctionTransformer(
                    lambda x: x[:, :-3] if remove == 0 else x[:-remove, :-3],
                    validate=True,
                ),
            ),
            (
                "Reshaping",
                FunctionTransformer(
                    lambda x: x.reshape((-1, seq_len, x.shape[-1])), validate=True
                ),
            ),
            (
                "Align",
                FunctionTransformer(lambda x: x[:-forward_predict], validate=False),
            ),
        ]
    )

    if standardize:
        preprocess_features.steps.insert(1, ("Scaling", StandardScaler()))

    if poly:
        preprocess_features.steps.insert(
            1, ("PolyFeatures", PolynomialFeatures(2, interaction_only=True))
        )

    preprocess_labels = Pipeline(
        steps=[
            (
                "ExtractLabels",
                FunctionTransformer(
                    lambda x: (x[:, -3:] if remove == 0 else x[:-remove, -3:]),
                    validate=True,
                ),
            ),
            (
                "Reshaping",
                FunctionTransformer(
                    lambda x: x.reshape((-1, seq_len, 3)), validate=True
                ),
            ),
            #             ("Align", FunctionTransformer(lambda x: x[forward_predict:, -1, :].reshape(-1, 4 if transition == 2 else 2), validate=False)),
            (
                "Align",
                FunctionTransformer(
                    lambda x: x[forward_predict:, :, :].reshape(-1, seq_len, 3),
                    validate=False,
                ),
            ),
        ]
    )
    print(len(data) / 10.0)
    x = preprocess_features.fit_transform(data)
    y = preprocess_labels.fit_transform(data)

    return x, y


def split_data(x, y, test_size, random_state):
    #     if x.shape[0] * x.shape[1] > 35_000:
    #         rr = np.random.choice(x.shape[0], int(35_000/x.shape[1]))
    #         x = x[rr]
    #         y = y[rr]
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
all_metrics = {}


# In[42]:
@click.command()
@click.argument("output_directory")
@click.option("--test_set_size", default=0.3, type=click.FLOAT)
@click.option("--random_seed", default=27, type=click.INT)
@click.option("--rnn_seq", default=10, type=click.INT)
@click.option("--epochs", default=10, type=click.INT)
@click.option("--batch_size", default=32, type=click.INT)
@click.option("--forward_predict", default=1, type=click.INT)
@click.option("--standardize", "standardize", flag_value=True, default=True)
@click.option("--poly", "poly", flag_value=True, default=False)
@click.option("--transition", "transition", type=click.INT, default=0)
def train_final_model(
    output_directory,
    test_set_size,
    random_seed,
    rnn_seq,
    epochs,
    batch_size,
    forward_predict,
    standardize,
    poly,
    transition,
):
    
    data = pd.read_parquet('../data/no_background/full_pcap_1s_irc.parquet').drop(columns=['label', 'n_abnormal', 'n_background', 'n_normal'])
    data.head()


#     # In[ ]:


#     rnn_seq = 2
    n_classes = 3
#     forward_predict = 1
#     standardize = True
#     poly = False
#     transition = 0

    output_directory = create_directory(output_directory)
    x, y = prepare_pcap_sequantial_data(
                data, rnn_seq, forward_predict, standardize, poly, transition 
            )

    x_tr, x_te, y_tr, y_te = split_data(x, y, 0.3, 27)

    x_tr = [np.array(x_tr[:,:,i]).reshape(-1,rnn_seq,1) for i in range(10)]
    x_te = [np.array(x_te[:,:,i]).reshape(-1,rnn_seq,1) for i in range(10)]

    inputs = [Input(shape=(rnn_seq,1)) for i in range(10)]
    rnn_units = [Bidirectional(GRU(10, recurrent_dropout=0.3, return_sequences=True))(inputs[i]) for i in range(10)]
    all_units = concatenate(rnn_units)
    out = TimeDistributed(Dense(units=n_classes, activation='sigmoid', name='Output'))(all_units)
    model = Model(inputs, out)
    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(optimizer=optimizer,
              loss=keras.losses.binary_crossentropy,
              metrics=["accuracy"])

    hist = model.fit(x_tr, y_tr, epochs=10, batch_size=32, validation_data=(x_te, y_te))     
    y_pred = model.predict(x_te)
    metrics = {}
    f1s = []

    for i in range(3):
        f1s.append([])
        for k in range(100):
            truth = y_te[:, :, i].reshape(-1)
            prediction = (y_pred >= k/100.0)[:,:,i]
            prediction = prediction.reshape(-1,)
            f1s[i].append(f1_score(truth, prediction))
    thresholds = [f1s[0].index(max(f1s[0])) / 100.0, f1s[1].index(max(f1s[1])) / 100.0, f1s[2].index(max(f1s[2])) / 100.0]
    for i in range(3):
        truth = y_te[:, :, i].reshape(-1)
        prediction = (y_pred >= thresholds[i])[:, :, i]
        prediction_prob = y_pred[:, :, i].reshape(-1)

        prediction = prediction.reshape(-1)
        plt.show()
        metrics[f"f1_{i}"] = f1_score(truth, prediction)
        metrics[f"precision_{i}"] = precision_score(truth, prediction)
        metrics[f"recall_{i}"] = recall_score(truth, prediction)
        metrics[f"accuracy_{i}"] = accuracy_score(truth, prediction)
        metrics[f"au-roc_{i}"] = roc_auc_score(truth, prediction_prob)



    ax = plt.subplot(111)
    ax.grid(True)
    name = ["General", "Spam", "DDoS"]
    colors = cycle(["viridis", "Greens", "inferno"])
    for i in range(3):
        y_prob = np.array(
            [1 - y_pred.reshape(-1, 3)[:, i], y_pred.reshape(-1, 3)[:, i]]
        ).transpose()
        y_truth = [name[i] if k == 1 else "0" for k in y_te.reshape(-1, 3)[:, i]]
        skplt.metrics.plot_roc(
            y_truth,
            y_prob,
            plot_macro=False,
            plot_micro=False,
            classes_to_plot=[name[i]],
            ax=ax,
            cmap=next(colors),
        )
    final_dir = create_directory(output_directory / f"full_{rnn_seq}_{forward_predict}")
    pd.DataFrame([metrics], columns=metrics.keys()).to_csv(final_dir / "metrics.csv")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fancybox=True, ncol=1)
    plt.savefig(final_dir / "roc.png", dpi=200)
    print("DONE WITH "+f"{rnn_seq}_{forward_predict}")

