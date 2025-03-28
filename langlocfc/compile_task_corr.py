import os
import pandas as pd

tc_path = '/nese/mit/group/evlab/u/cshain/results/fMRI_parcellate/derivatives/task_regression/task_corr/'
out_path = '/nese/mit/group/evlab/u/cshain/results/fMRI_parcellate/derivatives/task_regression/task_corr.csv'

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

