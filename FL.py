# pip install scikit-learn pandas numpy pcapy

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from pcapy import open_live, dump_open

# Function to load .pcap files and extract features
def load_pcap(file_path):
    packets = []
    
    capfile = pcapy.open_offline(file_path)
    
    while True:
        (header, packet) = capfile.next()
        if header is None:
            break
        
        packet_length = header.getlen()
        timestamp = header.getts()[0]
        
        packets.append({'length': packet_length, 'timestamp': timestamp})
    
    return pd.DataFrame(packets)

# Federated learning client class
class FLClient:
    def __init__(self, data):
        self.data = data
        self.model = IsolationForest()

    def train(self):
        X = self.data.drop(columns=['label'], errors='ignore')
        self.model.fit(X)

    def get_model(self):
        return self.model

def aggregate_models(models):
    n_models = len(models)
    
    global_model = IsolationForest()
    global_model.set_params(**models[0].get_params())
    
    for attr in ['estimators_', 'estimator_weights_']:
        global_model.__setattr__(attr, np.mean([model.__getattribute__(attr) for model in models], axis=0))
    
    return global_model

# Simulated federated learning process
def federated_learning(file_paths):
    clients = []
    
    for file_path in file_paths:
        data = load_pcap(file_path)
        client = FLClient(data)
        client.train()
        clients.append(client.get_model())

    global_model = aggregate_models(clients)
    
    return global_model

if __name__ == "__main__":
    pcap_files = ['data1.pcap', 'data2.pcap', 'data3.pcap']
    global_model = federated_learning(pcap_files)
    print("Federated model trained for anomaly detection.")