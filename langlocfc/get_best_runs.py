import os
import re
import pandas as pd
import argparse

session_match = re.compile(r'SUBJECTS/([^\\]+)/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Get the best (highest even-odd sp corr) sessions in the dataset.')
    parser.add_argument('-k', '--k', type=int, default=100, help='Number of best sessions to keep.')
    args = parser.parse_args()

    df = pd.read_csv(os.path.join('stability', 'SvN_even_vs_odd.csv'))
    df = df.sort_values(by='r_raw', ascending=False)
    df = df.head(args.k)['']
    df['session'] = df.even_path.apply(lambda x: session_match.search(x).group(1))
    df = df['session', 'r_raw']
    if not os.path.exists('best_sessions'):
        os.makedirs('best_sessions')
    df.to_csv(os.path.join('best_sessions', 'top_%d_sessions.csv' % args.k), index=False)