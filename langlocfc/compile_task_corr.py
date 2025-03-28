import sys
import os
import pandas as pd

try:
    with open('data_path.txt', 'r') as f:
        base_path = f.read().strip()
except FileNotFoundError:
    sys.stderr.write('Data path not set. Run `python -m langlocfc.set_data_path` before running any other scripts.\n')
    sys.stderr.flush()
    exit()

tc_path = os.path.join(base_path, 'derivatives', 'task_regression', 'task_corr')
out_path = os.path.join(base_path, 'derivatives', 'task_regression', 'task_corr.csv')

df = []
for path in os.listdir(tc_path):
    if not path.endswith('.csv'):
        continue
    df.append(pd.read_csv(os.path.join(tc_path, path)))

df = pd.concat(df)
cols = ['task_zR', 'fc_zR', 'task2res_zR', 'fcres_zR']
mean = df[cols].mean()
sem = df[cols].sem()
sem = sem.rename(lambda x: x + '_sem')

df = pd.concat([mean, sem]).to_csv(out_path)

