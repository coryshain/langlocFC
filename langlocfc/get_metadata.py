import os
import string
import yaml
import pandas as pd
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import argparse

ALPHA = string.ascii_lowercase

def index_to_char(ix):
    assert ix < 26**2 - 1, 'Index is too big to convert to char'
    head = ix // 26
    tail = ix % 26
    if head > 0:
        head = ALPHA[head]
    else:
        head = ''
    tail = ALPHA[tail]

    return head + tail

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Get k (parcellation granularity) by session''')
    argparser.add_argument('results_dir', help='Path to results directory')
    argparser.add_argument('-p', '--parcellation_id', default='main', help='ID of parcellation to use')
    argparser.add_argument('-F', '--filter', default=None, help='String that results directories must contain')
    args = argparser.parse_args()

    paths_by_subject = {}

    for x in os.listdir(args.results_dir):
        if args.filter and args.filter not in x:
            continue
        try:
            int(x.split('_')[0])
            x.split('_')[1]
            subject = int(x.split('_')[0])
            if not subject in paths_by_subject:
                paths_by_subject[subject] = []
            paths_by_subject[subject].append(x)
        except ValueError:
            continue

    if not os.path.exists('stitched'):
        os.makedirs('stitched')

    out = []
    for subject in paths_by_subject:
        n_sessions = len(paths_by_subject[subject])
        for subject_path in paths_by_subject[subject]:
            cfg_path = os.path.join(
                args.results_dir,
                subject_path,
                'parcellation',
                args.parcellation_id,
                'parcellate_kwargs_optimized.yml'
            )
            if not os.path.exists(cfg_path):
                cfg_path = os.path.join(
                    args.results_dir,
                    subject_path,
                    'parcellation',
                    args.parcellation_id,
                    'parcellate_kwargs.yml'
                )
            if os.path.exists(cfg_path):
                with open(cfg_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                action_sequence = cfg['action_sequence']

                # Get runs and TRs
                sample_id = None
                k = None
                for action in action_sequence:
                    if action['type'] == 'sample':
                        sample_id = action['id']
                        k = action['kwargs']['n_networks']
                        break
                assert sample_id, 'sample_id not found'
                metadata_path = os.path.join(
                    args.results_dir,
                    subject_path,
                    'sample',
                    sample_id,
                    'metadata.csv'
                )

                metadata = pd.read_csv(metadata_path)
                n_trs = metadata.n_trs.values.squeeze()
                n_runs = metadata.n_runs.values.squeeze()
                row = dict(
                    subject=subject,
                    n_sessions=n_sessions,
                    k=k,
                    n_trs=n_trs,
                    n_runs=n_runs
                )
                out.append(row)

    out = pd.DataFrame(out)
    if args.filter:
        filename_base = args.filter
    else:
        filename_base = ''
    out.to_csv(os.path.join(args.results_dir, filename_base + 'metadata.csv'), index=False)
    _out = out[out.n_sessions > 1].groupby('subject')[['k']].sem().mean()
    _out.to_csv(os.path.join(args.results_dir, filename_base + 'metadata_k_sem.csv'), index=False)
    _out = pd.concat([out[['n_runs']].mean(), out[['n_runs']].sem()])
    _out.to_csv(os.path.join(args.results_dir, filename_base + 'metadata_n_runs.csv'), index=False)
    _out = pd.concat([out[['n_trs']].mean(), out[['n_trs']].sem()])
    _out.to_csv(os.path.join(args.results_dir, filename_base + 'metadata_n_trs.csv'), index=False)
