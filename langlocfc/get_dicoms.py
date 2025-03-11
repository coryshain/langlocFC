import argparse
import os
import yaml
import pandas as pd

from langlocfc.initialize import SUBJECTS_DIR, get_functional_dicoms, parse_cfg

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Get table of paths to dicom files')
    argparser.add_argument('config_paths', nargs='+', help='Paths to parcellate config files.')
    args = argparser.parse_args()

    out = []
    for config_path in args.config_paths:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        session = os.path.basename(cfg['output_dir'])

        session_dir = os.path.join(SUBJECTS_DIR, session)
        datacfg_path = os.path.join(session_dir, 'data.cfg')
        dicoms = get_functional_dicoms(datacfg_path)
        functionals = list(range(1, len(dicoms) + 1))
        for functional, dicom in zip(functionals, dicoms):
            row = dict(
                session=session,
                session_dir=session_dir,
                functional=functional,
                dicom=dicom
            )
            out.append(row)
    df = pd.DataFrame(out)
    df.to_csv('langlocFC_dicoms.csv', index=False)

