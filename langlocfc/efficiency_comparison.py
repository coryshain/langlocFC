import sys
import os
import re
import yaml
import numpy as np
import pandas as pd
from nilearn import image, masking
import argparse

try:
    with open('data_path.txt', 'r') as f:
        base_path = f.read().strip()
except FileNotFoundError:
    sys.stderr.write('Data path not set. Run `python -m langlocfc.set_data_path` before running any other scripts.\n')
    sys.stderr.flush()
    exit()
MULTISESSION_DIR = os.path.join(base_path, 'derivatives', 'nolangloc_multisession')
RESULTS_DIR = os.path.join(base_path, 'derivatives', 'stability_runs')
REFERENCE_ATLASES = ['LANG', 'LANA']

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Compare parcellation results with and without task regression''')
    argparser.add_argument('-f', '--fisher', action='store_true', help='Whether to use Fisher (vs arithmetic) average of correlations')
    argparser.add_argument('-o', '--output_dir', default=None, help='Prefix to use for parcellation output directory')
    args = argparser.parse_args()

    eps = 1e-3

    sessions = []
    atlases_by_subject = {}
    runK = set()
    mask = None
    paths = os.listdir(MULTISESSION_DIR)
    for i, path in enumerate(paths):
        sys.stderr.write('\rProcessing %d/%d' % (i + 1, len(paths)))
        sys.stderr.flush()
        match = re.match('([^_]+)_run([0-9]+)', path)
        if not match:
            continue
        subject, n_runs = match.groups()
        runK.add(n_runs)
        if subject not in atlases_by_subject:
            atlases_by_subject[subject] = {}
        if not n_runs in atlases_by_subject[subject]:
            atlases_by_subject[subject][n_runs] = {}
        for ref in REFERENCE_ATLASES:
            path = f'{ref}_sub1.nii.gz'
            img = image.load_img(os.path.join(
                MULTISESSION_DIR, '%s_run%s' % (subject, n_runs), 'parcellation', 'main', path
            ))
            if mask is None:
                mask = image.get_data(
                    masking.compute_brain_mask(img, connected=False, opening=False, mask_type='gm')) > 0.5
            img = image.get_data(img)[mask]
            atlases_by_subject[subject][n_runs][ref] = img

    print()

    runK = sorted(list(runK))
    R = []
    for subject in atlases_by_subject:
        for ref in REFERENCE_ATLASES:
            X = np.stack([atlases_by_subject[subject][_runK][ref] for _runK in runK], axis=0)
            _R = np.corrcoef(X)
            if args.fisher:
                _R = np.arctanh(_R * (1 - eps))
            _R = pd.DataFrame(_R, index=runK, columns=runK)
            _R['subject'] = subject
            _R['reference_atlas'] = ref
            R.append(_R)

    R = pd.concat(R, axis=0)
    R = R.reset_index(names='runK')

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    R.to_csv(os.path.join(RESULTS_DIR, 'multisession_stability.csv'), index=False)

    for ref in REFERENCE_ATLASES:
        _R = R[R.reference_atlas == ref][['runK'] + runK]
        _R = _R.groupby('runK').mean().reset_index()
        _R.to_csv(os.path.join(RESULTS_DIR, f'multisession_stability_summary_{ref}.csv'), index=False)
