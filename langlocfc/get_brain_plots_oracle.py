import sys
import os
import re
import numpy as np
import pandas as pd
import argparse

try:
    with open('data_path.txt', 'r') as f:
        base_path = f.read().strip()
except FileNotFoundError:
    sys.stderr.write('Data path not set. Run `python -m langlocfc.set_data_path` before running any other scripts.\n')
    sys.stderr.flush()
    exit()

session_match = re.compile(r'SUBJECTS/([^\/]+)/')
results_dir = os.path.join(base_path, 'derivatives', 'nolangloc')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get brain plots for the best (highest FC-to-task sp corr) sessions in the dataset.')
    parser.add_argument('-k', '--k', type=int, default=10, help='Number of best sessions to keep.')
    args = parser.parse_args()

    df = []
    sessions = os.listdir(results_dir)
    for i, session in enumerate(os.listdir(results_dir)):
        sys.stderr.write('\rProcessing %d/%d' % (i + 1, len(sessions)))
        sys.stderr.flush()
        df_path = os.path.join(results_dir, session, 'evaluation', 'main', 'evaluation.csv')
        if not os.path.exists(df_path):
            continue
        _df = pd.read_csv(df_path)
        _df = _df[_df.parcel == 'LANA_sub1']
        if not 'eval_Lang_S-N_score' in _df:
            continue
        row = dict(
            session=session,
            r_raw=np.squeeze(_df['eval_Lang_S-N_score'].values),
        )
        df.append(row)
    sys.stderr.write('\n')
    sys.stderr.flush()
    df = pd.DataFrame(df)
    df = df.sort_values(by='r_raw', ascending=False)
    df = df.head(args.k)
    print(df)
    sessions = df.session.tolist()
    if not os.path.exists('brain_plots_oracle'):
        os.makedirs('brain_plots_oracle')
    for session in sessions:
        in_path = os.path.join(results_dir, session, 'parcellation', 'main', 'plots', 'LANA_sub1_vs_Lang_S-N.png')
        out_path = os.path.join('brain_plots_oracle', '%s.png' % session)
        os.system('cp %s %s' % (in_path, out_path))
