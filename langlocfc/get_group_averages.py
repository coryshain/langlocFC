import sys
import os
import re
import yaml
import numpy as np
from nilearn import image
import argparse

atlas_match = re.compile('([A-Za-z_]+)_sub([0-9]+).nii.gz')

REFERENCE_ATLASES = ['LANG', 'LANA']

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Get group averages''')
    argparser.add_argument('config_paths', nargs='+', help='Paths to config files')
    argparser.add_argument('-m', '--max_subnetworks', type=int, default=4, help='Maximum number of subnetworks to include')
    argparser.add_argument('-o', '--output_dir', default=None, help='Prefix to use for parcellation output directory')
    args = argparser.parse_args()

    atlases = {}
    counts = {}
    img_ref = None
    for i, config_path in enumerate(args.config_paths):
        sys.stderr.write('\rProcessing %d/%d' % (i + 1, len(args.config_paths)))
        sys.stderr.flush()
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        if args.output_dir:
            output_dir = os.path.join(args.output_dir, os.path.basename(cfg['output_dir']))
        else:
            output_dir = cfg['output_dir']
        parcellation_dir = os.path.join(output_dir, 'parcellation', 'main')
        for network in REFERENCE_ATLASES:
            _atlases = {}
            for atlas_name in ('sub1', 'all'):
                if atlas_name == 'sub1':
                    path = f'{network}_{atlas_name}.nii.gz'
                else:
                    path = f'{network}.nii.gz'
                img = image.load_img(os.path.join(parcellation_dir, path))
                if img_ref is None:
                    img_ref = img
                img = (image.get_data(img) > 0.5).astype(int)
                _atlases[f'{network}_{atlas_name}'] = img
            _atlases[f'{network}_diff'] = np.clip(_atlases[f'{network}_all'] - _atlases[f'{network}_sub1'], 0, 1)
            for atlas_name in _atlases:
                if atlas_name not in atlases:
                    atlases[atlas_name] = None
                if atlases[atlas_name] is None:
                    atlases[atlas_name] = _atlases[atlas_name]
                else:
                    atlases[atlas_name] += _atlases[atlas_name]
                if atlas_name not in counts:
                    counts[atlas_name] = 0
                counts[atlas_name] += 1

    print()

    if not os.path.exists('group_atlas_data'):
        os.makedirs('group_atlas_data')
    for atlas_name in atlases:
        print(counts[atlas_name])
        img = atlases[atlas_name]
        img = img / counts[atlas_name]
        img = image.new_img_like(img_ref, img)
        img.to_filename(os.path.join('group_atlas_data', f'{atlas_name}.nii.gz'))