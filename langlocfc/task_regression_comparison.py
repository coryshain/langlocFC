import sys
import os
import re
import yaml
import numpy as np
import pandas as pd
from nilearn import image, masking
import argparse

sys.path.append('parcellate')

try:
    with open('data_path.txt', 'r') as f:
        base_path = f.read().strip()
except FileNotFoundError:
    sys.stderr.write('Data path not set. Run `python -m langlocfc.set_data_path` before running any other scripts.\n')
    sys.stderr.flush()
    exit()

TASK_REGRESSION_DIR = os.path.join(base_path, 'derivatives', 'task_regression')
PARCELLATE_CFG_DIR = os.path.join(TASK_REGRESSION_DIR, 'parcellate_cfg')
PARCELLATE_DIR = os.path.join(base_path, 'derivatives')
REFERENCE_ATLASES = ['LANG', 'LANA']

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Compare parcellation results with and without task regression''')
    argparser.add_argument('-f', '--fisher', action='store_true', help='Whether to use Fisher (vs arithmetic) average of correlations')
    argparser.add_argument('-o', '--output_dir', default=None, help='Prefix to use for parcellation output directory')
    args = argparser.parse_args()

    eps = 1e-3

    sessions = []
    for path in os.listdir(os.path.join(PARCELLATE_CFG_DIR, 'residualized')):
        match = re.match('(.*)_residualized.yml', path)
        if not match:
            continue
        sess_name = match.group(1)
        sessions.append(sess_name)

    mask = None
    df = []
    for i, sess_name in enumerate(sessions):
        sys.stderr.write('\rProcessing %d/%d' % (i + 1, len(sessions)))
        sys.stderr.flush()
        unresidualized_cfg = os.path.join(
            PARCELLATE_DIR, 'unresidualized', sess_name, 'config.yml'
        )
        residualized_cfg = os.path.join(
            PARCELLATE_DIR, 'residualized', sess_name, 'config.yml'
        )
        residualizedRand_cfg = os.path.join(
            PARCELLATE_DIR, 'residualizedRand', sess_name, 'config.yml'
        )
        with open(unresidualized_cfg, 'r') as f:
            unresidualized_cfg = yaml.safe_load(f)
        with open(residualized_cfg, 'r') as f:
            residualized_cfg = yaml.safe_load(f)
        with open(residualizedRand_cfg, 'r') as f:
            residualizedRand_cfg = yaml.safe_load(f)

        row = dict(session=sess_name)
        if args.output_dir:
            unresidualized_output_dir = os.path.join(
                args.output_dir,
                os.path.basename(os.path.dirname(unresidualized_cfg['output_dir'])),
                os.path.basename(unresidualized_cfg['output_dir'])
            )
            residualized_output_dir = os.path.join(
                args.output_dir,
                os.path.basename(os.path.dirname(residualized_cfg['output_dir'])),
                os.path.basename(residualized_cfg['output_dir'])
            )
            residualizedRand_output_dir = os.path.join(
                args.output_dir,
                os.path.basename(os.path.dirname(residualizedRand_cfg['output_dir'])),
                os.path.basename(residualizedRand_cfg['output_dir'])
            )
        else:
            unresidualized_output_dir = unresidualized_cfg['output_dir']
            residualized_output_dir = residualized_cfg['output_dir']
            residualizedRand_output_dir = residualizedRand_cfg['output_dir']
        for ref in REFERENCE_ATLASES:
            unresidualized = image.load_img(
                os.path.join(
                    unresidualized_output_dir, 'parcellation', 'main', '%s_sub1.nii.gz' % ref
                )
            )
            residualized = image.load_img(
                os.path.join(
                    residualized_output_dir, 'parcellation', 'main', '%s_sub1.nii.gz' % ref
                )
            )
            residualizedRand = image.load_img(
                os.path.join(
                    residualizedRand_output_dir, 'parcellation', 'main', '%s_sub1.nii.gz' % ref
                )
            )
            if mask is None:
                mask = image.get_data(
                    masking.compute_brain_mask(unresidualized, connected=False, opening=False, mask_type='gm')) > 0.5
            unresidualized = image.get_data(unresidualized)[mask]
            residualized = image.get_data(residualized)[mask]
            residualizedRand = image.get_data(residualizedRand)[mask]
            r = np.corrcoef(unresidualized, residualized)[0, 1]
            if args.fisher:
                r = np.arctanh(r * (1 - eps))
            row['%s_r_unresidualized2residualized' % ref] = r
            r = np.corrcoef(unresidualized, residualizedRand)[0, 1]
            if args.fisher:
                r = np.arctanh(r * (1 - eps))
            row['%s_r_unresidualized2residualizedRand' % ref] = r
            r = np.corrcoef(residualized, residualizedRand)[0, 1]
            if args.fisher:
                r = np.arctanh(r * (1 - eps))
            row['%s_r_residualized2residualizedRand' % ref] = r
        df.append(row)

    print()

    df = pd.DataFrame(df)
    df.to_csv(os.path.join(TASK_REGRESSION_DIR, 'task_regression_comparison.csv'), index=False)

    del df['session']
    df_mean = df.mean()
    df_sem = df.rename(lambda x: '%s_sem' % x).sem()
    df = pd.concat([df_mean, df_sem], axis=1)
    df.to_csv(os.path.join(TASK_REGRESSION_DIR, 'task_regression_comparison_summary.csv'))
