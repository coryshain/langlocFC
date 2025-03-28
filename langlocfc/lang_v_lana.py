import sys
import os
import yaml
from nilearn import image, masking
import numpy as np
import pandas as pd
import argparse


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Evaluate stability of LangFC from LANG vs LanA''')
    argparser.add_argument('config_paths', nargs='+', help='Paths to by-subject config files')
    argparser.add_argument('-f', '--fisher', action='store_true', help='Fisher-transform correlations')
    argparser.add_argument('-o', '--output_dir', default=None, help='Prefix to use for parcellation output directory')
    args = argparser.parse_args()

    eps = 1e-3

    r = []
    same = []
    for i, config_path in enumerate(args.config_paths):
        sys.stderr.write('\r  %d/%d' % (i + 1, len(args.config_paths)))
        sys.stderr.flush()
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        if args.output_dir:
            output_dir = os.path.join(args.output_dir, os.path.basename(cfg['output_dir']))
        else:
            output_dir = cfg['output_dir']
        parcellation_path = os.path.join(output_dir, 'parcellation', 'main')
        lang_path = os.path.join(parcellation_path, 'LANG_sub1.nii.gz')
        lana_path = os.path.join(parcellation_path, 'LANA_sub1.nii.gz')
        lang = image.load_img(lang_path)
        mask = image.get_data(masking.compute_brain_mask(lang, connected=False, opening=False, mask_type='gm')) > 0.5
        lang = image.get_data(lang)[mask]
        lana = image.get_data(image.load_img(lana_path))[mask]
        _same = np.allclose(lang, lana)
        same.append(_same)
        _r = np.corrcoef(lang, lana)[0, 1]
        if args.fisher:
            _r = np.arctanh(_r * (1 - eps))
        r.append(_r)
    r = pd.Series(r)
    same = pd.Series(same)
    diff_r = r[~same]
    print()
    print('Proportion same: %0.03f' % same.mean(), '| z(r) for different. mean: %0.03f' % diff_r.mean(),
          ', sem: %0.03f' % diff_r.sem())


