import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def dummy_file():
    for filename in os.listdir('./'):
        file = os.path.join( filename)
        if not file.endswith('.csv'):
            continue
        # checking if it is a file
        dummy = os.path.join('dummy', filename)
        print(file)
        with open(file, 'r') as read_obj, open(dummy, 'w') as write_obj:
            # Iterate over the given list of strings and write them to dummy file as lines
            Lines = read_obj.readlines()
            Lines[0] = Lines[0].replace('\n',',hyperparam2\n')
            for line in Lines:
                write_obj.write(line)

dummy_file()
plt.figure()

for seed in range(1):
    biased_data = pd.read_csv('./dummy/progressbiased-ppo-tune-16cartpole.csv',
                                header=0,
                       parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                       index_col='timestamp')
    biased_data.columns = biased_data.columns.astype(int)
    biased_data = biased_data.sort_index(axis=1, ascending=True)
    # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
    # weighted.append(rets)
biased = biased_data.to_numpy()
# for i in reversed(range(10, weighted.shape[1], 1)):
#     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
mean = np.mean(biased, axis=0)
std = np.std(biased, axis=0) / np.sqrt(10)
print("biased: ",biased.shape[0])
plt.plot(biased_data.columns,mean,color='tab:green',label='biased')
plt.fill_between(biased_data.columns,mean +std, mean -std, color='tab:green', alpha=0.2, linewidth=0.9)

for seed in range(1):
    weighted_data = pd.read_csv('./dummy/progressweighted-ppo-tune-790cartpole.csv',
                                header=0,
                       parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                       index_col='timestamp')
    weighted_data.columns = weighted_data.columns.astype(int)
    weighted_data = weighted_data.sort_index(axis=1, ascending=True)
    # rets = weighted_data.loc[env + '-'+str(seed)].to_numpy()
    # weighted.append(rets)
weighted = weighted_data.to_numpy()
print("weighted: ",weighted.shape[0])
# for i in reversed(range(10, weighted.shape[1], 1)):
#     weighted[:, i] = np.mean(weighted[:, i - 10:i + 1], axis=1)
mean = np.mean(weighted, axis=0)
std = np.std(weighted, axis=0) / np.sqrt(10)
plt.plot(weighted_data.columns,mean,color='tab:orange',label='our correction')
plt.fill_between(biased_data.columns, mean +std, mean -std, color='tab:orange', alpha=0.2, linewidth=0.9)

for seed in range(1):
    naive_data = pd.read_csv('./dummy/progressbiased-ppo-tune-750cartpole.csv',
                                header=0,
                       parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                       index_col='timestamp')
    naive_data.columns = naive_data.columns.astype(int)
    naive_data = naive_data.sort_index(axis=1, ascending=True)
    # rets = naive_data.loc[env + '-'+str(seed)].to_numpy()
    # naive.append(rets)
naive = naive_data.to_numpy()
print("naive: ",naive.shape[0])
# for i in reversed(range(10, naive.shape[1], 1)):
#     naive[:, i] = np.mean(naive[:, i - 10:i + 1], axis=1)
mean = np.mean(naive, axis=0)
std = np.std(naive, axis=0) / np.sqrt(10)
plt.plot(naive_data.columns,mean,color='tab:blue',label='existing correction')
plt.fill_between(biased_data.columns, mean +std, mean -std, color='tab:blue', alpha=0.2, linewidth=0.9)

for seed in range(1):
    naive_data = pd.read_csv('./dummy/progressnaive-ppo-tune-790cartpole.csv',
                                header=0,
                       parse_dates={'timestamp': ['hyperparam','hyperparam2']},
                       index_col='timestamp')
    naive_data.columns = naive_data.columns.astype(int)
    naive_data = naive_data.sort_index(axis=1, ascending=True)
    # rets = naive_data.loc[env + '-'+str(seed)].to_numpy()
    # naive.append(rets)
naive = naive_data.to_numpy()
print("naive: ",naive.shape[0])
# for i in reversed(range(10, naive.shape[1], 1)):
#     naive[:, i] = np.mean(naive[:, i - 10:i + 1], axis=1)
mean = np.mean(naive, axis=0)
std = np.std(naive, axis=0) / np.sqrt(10)
plt.plot(naive_data.columns,mean,color='tab:red',label='existing correction')
plt.fill_between(biased_data.columns, mean +std, mean -std, color='tab:blue', alpha=0.2, linewidth=0.9)

plt.legend()
plt.show()