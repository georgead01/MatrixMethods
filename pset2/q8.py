import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data_path =  'pset2/data/gisette/GISETTE/gisette_train.data'
train_labels_path = 'pset2/data/gisette/GISETTE/gisette_train.labels'

train_data = pd.read_csv(train_data_path, delimiter=' ', header=None).to_numpy()[:, :-1]
train_labels = pd.read_csv(train_labels_path, delimiter=' ', header=None).to_numpy()

A = train_data.T
m, n = A.shape
A_mean = A.mean(axis = 1, keepdims=True)
A_0 = A-A_mean

U, S_diag, VT = np.linalg.svd(A_0)
total_var = (S_diag**2).sum()/(n-1)

k = 0
tot_k_var = 0
while k < 0 or tot_k_var/(n-1) < 0.99 * total_var:
    sigma = S_diag[k]**2
    tot_k_var += sigma
    k += 1

#print(k)

B = U.T[:k, :]@A_0

plt.plot(B[0, np.where(train_labels.T==-1)[1]], B[1, np.where(train_labels.T==-1)[1]], 'C0o')
plt.plot(B[0, np.where(train_labels.T==1)[1]], B[1, np.where(train_labels.T==1)[1]], 'C1o')
plt.show()