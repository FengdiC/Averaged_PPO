import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('./progresserr-weighted-ppo-tune-0.csv', header=0, index_col='name')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:, :8]
ppo = data.to_numpy()
plt.figure()
plt.plot(range(0,ppo.shape[1]*800,800),np.mean(ppo,axis=0),label='ppo')

data = pd.read_csv('./progressbatch-ac-err-in-buffer0.csv', header=0, index_col='name')
data.columns = data.columns.astype(int)
data = data.sort_index(axis=1, ascending=True)
data = data.iloc[:, :6]
batch = data.to_numpy()

plt.plot(range(0,batch.shape[1]*1000,1000),np.mean(batch,axis=0),label='batch-ac')
plt.legend()
plt.show()