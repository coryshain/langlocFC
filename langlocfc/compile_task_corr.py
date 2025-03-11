import os
import pandas as pd

tc_path = '/nese/mit/group/evlab/u/cshain/results/fMRI_parcellate/task_regression/task_corr/'
out_path = '/nese/mit/group/evlab/u/cshain/results/fMRI_parcellate/task_regression/task_corr.csv'

df = []
for path in os.listdir(tc_path):
    if not path.endswith('.csv'):
        continue
    df.append(pd.read_csv(os.path.join(tc_path, path)))

df = pd.concat(df)
mean = df[['task_zR', 'fc_zR']].mean()
sem = df[['task_zR', 'fc_zR']].sem()
sem = sem.rename(lambda x: x + '_sem')

df = pd.concat([mean, sem]).to_csv(out_path)

