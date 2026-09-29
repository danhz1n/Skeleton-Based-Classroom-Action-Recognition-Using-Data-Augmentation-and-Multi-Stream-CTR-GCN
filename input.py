import pickle
import numpy as np

# Load data
with open('./data/dav/train_data.pkl', 'rb') as f:
    data = pickle.load(f)

# Load label
with open('./data/dav/train_label.pkl', 'rb') as f:
    labels = pickle.load(f)

print("Data type   :", type(data))
print("Data shape  :", np.array(data).shape if isinstance(data, list) else data.shape)
print("Label type  :", type(labels))
print("Num samples :", len(labels))
print("Num classes :", len(set(labels)))
print("Labels      :", sorted(set(labels)))