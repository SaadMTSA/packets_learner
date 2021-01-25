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

def _prepare_orderedpair_data(data, downsample=0.01):
    first = data.iloc[:-1,-1]
    second = data.iloc[1:,-1]
    label = first.values * 2 + second.values
    data.iloc[1:,-1] = label
    data = data.iloc[1:,:] 
    to_keep = []
    for i in label:
        if i == 1 or i == 2:
            to_keep.append(True)
        else:
            to_keep.append(np.random.rand(1)[0] <= downsample)
    data = data[to_keep]
    data = data.append(data)
    return data

def _prepare_transition_data(data):
    first = data.iloc[:-1,-1]
    second = data.iloc[1:,-1]
    label = first.values != second.values
    data.iloc[1:,-1] = label
    data.iloc[:,-1] = data.iloc[:,-1].apply(lambda x: 1 if x else 0)
    return data.iloc[1:,:]


def preprocess_netflow_data(data, label_col, transition, nn=False):
    data = data.fillna(0, axis=None)
#     data["n_background_rate"] = data["n_background"] / data["n_conn"]
#     data["n_normal_rate"] = data["n_normal"] / data["n_conn"]
    data[label_col] = data["n_conn"] - data["n_normal"]
#     data = data.drop(columns='n_normal')
    data[label_col] = data[label_col].apply(lambda x: 1 if x > 0 else 0)
    print(data[label_col].value_counts())
    data["n_background"] = data["n_conn"] - data["n_normal"]
    if nn:
        if transition == 0:
            best_columns = ['n_dst_ipclass_b', 'ent_src_ip', 'n_normal', 'n_packets', 'ent_src_port>=1024', 'ent_dst_ip_a', 'ent_dst_port<1024', 'n_udp', 'n_dst_ipclass_c', 'n_dst_port>=1024']
#         else:
#             best_columns = ['s_dst_ip_b', 'sd_duration', 's_dstip', 's_src_ip_b', 'n_dst_port<1024', 'n_src_ip_c', 'n_src_ip_a', 's_src_ip_a', 'n_background', 'n_background']

        data=data[best_columns]
    if transition == 1:
        data = _prepare_transition_data(data)
    elif transition == 2:
        data = _prepare_orderedpair_data(data)
    return data


def preprocess_pcap_data(data, label_col):
    data[label_col] = data["n_packets"] - data["n_normal"]
    data[label_col] = data[label_col] > 0
    data[label_col] = data[label_col].map({True: 1, False: 0})
    data=data.drop(columns=['n_abnormal', 'n_background'])
    return data

def prepare_netflow_sequantial_data(data, seq_len, forward_predict, standardize, poly, transition):
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
            ("Reshaping", FunctionTransformer(lambda x: x.reshape((-1, seq_len, 4 if transition == 2 else 2)), validate=True)),            
#             ("Align", FunctionTransformer(lambda x: x[forward_predict:, -1, :].reshape(-1, 4 if transition == 2 else 2), validate=False)),
            ("Align", FunctionTransformer(lambda x: x[forward_predict:, :, :].reshape(-1, seq_len, 4 if transition == 2 else 2), validate=False)),
        ]
    )
    print(len(data)/10.0)
    x = preprocess_features.fit_transform(data)
    y = preprocess_labels.fit_transform(data)

    return x, y



def prepare_pcap_sequantial_data(data, seq_len, forward_predict, standardize, poly, transition):
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
            ("Reshaping", FunctionTransformer(lambda x: x.reshape((-1, seq_len, 4 if transition == 2 else 2)), validate=True)),            
#             ("Align", FunctionTransformer(lambda x: x[forward_predict:, -1, :].reshape(-1, 4 if transition == 2 else 2), validate=False)),
            ("Align", FunctionTransformer(lambda x: x[forward_predict:, :, :].reshape(-1, seq_len, 4 if transition == 2 else 2), validate=False)),
        ]
    )
    print(len(data)/10.0)
    x = preprocess_features.fit_transform(data)
    y = preprocess_labels.fit_transform(data)

    return x, y

def split_data(x, y, test_size, random_state, stratified=False):
    if stratified:
#         temp = pd.Series(np.argmax(y, axis=-1))
#         print(temp.value_counts())
#         print(temp.value_counts() / len(temp))
        sss = StratifiedShuffleSplit(1, test_size=0.3)
        for train_index, test_index in sss.split(x, y):
            return x[train_index], x[test_index], y[train_index], y[test_index]
    else:
        if x.shape[0] * x.shape[1] > 35_000:
            rr = np.random.choice(x.shape[0], int(35_000/x.shape[1]))
            x = x[rr]
            y = y[rr]
        return train_test_split(x, y, test_size=test_size, random_state=random_state)


def create_directory(directory):
    directory = Path(directory)
    os.makedirs(directory, exist_ok=True)
    return directory