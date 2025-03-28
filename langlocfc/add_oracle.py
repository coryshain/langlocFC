import os
import copy
import yaml
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Add oracle analysis (labeling against localizer task) to a config file.')
    argparser.add_argument('config_paths', nargs='+', help='Paths to config files.')
    args = argparser.parse_args()

    for config_path in args.config_paths:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        if 'evaluation_atlases' in cfg['evaluate']['main'] and \
                'Lang_S-N' in cfg['evaluate']['main']['evaluation_atlases']:
            if not 'label' in cfg:
                cfg['label'] = {}
            cfg['label']['oracle'] ={
                'reference_atlases': {'Lang_S-N': cfg['evaluate']['main']['evaluation_atlases']['Lang_S-N']}
            }

            cfg['evaluate']['oracle'] = copy.deepcopy(cfg['evaluate']['main'])
            cfg['evaluate']['oracle']['labeling_id'] = 'oracle'
            cfg['parcellate']['oracle'] = copy.deepcopy(cfg['parcellate']['main'])
            cfg['parcellate']['oracle']['evaluation_id'] = 'oracle'

            with open(config_path, 'w') as f:
                yaml.safe_dump(cfg, f, sort_keys=False)