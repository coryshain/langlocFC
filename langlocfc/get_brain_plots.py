import os
import re
import pandas as pd
import argparse

session_match = re.compile(r'SUBJECTS/([^\/]+)/')
results_dir = '/nese/mit/group/evlab/u/cshain/results/fMRI_parcellate/derivatives/nolangloc'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get brain plots for the best (highest even-odd sp corr) sessions in the dataset.')
    parser.add_argument('-k', '--k', type=int, default=10, help='Number of best sessions to keep.')
    args = parser.parse_args()

    df = pd.read_csv(os.path.join('stability', 'SvN_even_vs_odd.csv'))
    df = df.sort_values(by='r_raw', ascending=False)
    df = df.head(args.k)
    df['session'] = df.even_path.apply(lambda x: session_match.search(x).group(1))
    df = df[['session', 'r_raw']]
    sessions = df.session.tolist()
    if not os.path.exists('brain_plots'):
        os.makedirs('brain_plots')
    for session in sessions:
        in_path = os.path.join(results_dir, session, 'parcellation', 'main', 'plots', 'LANA_sub1_vs_Lang_S-N.png')
        out_path = os.path.join('brain_plots', '%s.png' % session)
        os.system('cp %s %s' % (in_path, out_path))
