from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline


def preprocess_netflow_data(data, label_col):
    data = data.fillna(0, axis=None)
    data["n_background_rate"] = data["n_background"] / data["n_conn"]
    data["n_normal_rate"] = data["n_normal"] / data["n_conn"]
    data[label_col] = data["n_conn"] - data["n_background"] - data["n_normal"]
    data[label_col] = data[label_col].apply(lambda x: 1 if x > 0 else 0)
    data["n_background"] = data["n_conn"] - data["n_normal"]
    return data


def preprocess_pcap_data(data, label_col):
    data[label_col] = data[label_col] > 0
    data[label_col] = data[label_col].map({True: 1, False: 0})
    return data

def prepare_netflow_sequantial_data(data, seq_len, forward_predict, standardize, poly):
    if forward_predict < 1:
        raise ValueError(f"forward_predict should be > 0, {forward_predict} was given!")

    remove = len(data) % seq_len
    preprocess_features = Pipeline(
        steps=[
            ("ExtractFeatures", FunctionTransformer(lambda x: x[:, :-1] if remove == 0 else x[:-remove, :-1], validate=True)),
            ("Reshaping", FunctionTransformer(lambda x: x.reshape((-1, seq_len, x.shape[-1])), validate=True)),
            ("Align", FunctionTransformer(lambda x: x[:-forward_predict], validate=False)),
        ]
    )

    if standardize:
        preprocess_features.steps.insert(1, ("Scaling", StandardScaler()))
        
    if poly:
        preprocess_features.steps.insert(1, ("PolyFeatures", PolynomialFeatures(2, interaction_only=True)))
        

    preprocess_labels = Pipeline(
        steps=[
            ("ExtractLabels", FunctionTransformer(lambda x: (x[:, -1] if remove == 0 else x[:-remove, -1]).reshape(-1,1), validate=True)),
            ("Encoding", OneHotEncoder(sparse=False)),
            ("Reshaping", FunctionTransformer(lambda x: x.reshape((-1, seq_len, 2)), validate=True)),            
            ("Align", FunctionTransformer(lambda x: x[forward_predict:, -1, :].reshape(-1, 2), validate=False)),
        ]
    )
    
    x = preprocess_features.fit_transform(data)
    y = preprocess_labels.fit_transform(data)

    return x, y



def prepare_pcap_sequantial_data(data, seq_len, forward_predict, standardize, poly):
    raise NotSupportedException("Implementation isn't done, yet!")
        


    preprocess_labels = Pipeline(
        steps=[
            ("ExtractLabels", FunctionTransformer(lambda x: x[:, -1].reshape(-1,1), validate=True)),
            ("Encoding", OneHotEncoder(sparse=False)),
            ("Reshaping", FunctionTransformer(lambda x: x.reshape((-1, seq_len, 2)), validate=True)),            
            ("Align", FunctionTransformer(lambda x: x[forward_predict:, 0, :].reshape(-1, 2), validate=False)),
        ]
    )
    
    x = preprocess_features.fit_transform(data)
    y = preprocess_labels.fit_transform(data)

    return x, y


def split_data(x, y, test_size, random_state):
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def create_directory(directory):
    directory = Path(directory)
    os.makedirs(directory, exist_ok=True)
    return directory