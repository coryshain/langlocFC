import os
import yaml
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Evaluate stability of S-N vs langlocFC''')
    argparser.add_argument('config_paths', nargs='+', help='Paths to by-subject config files')
    argparser.add_argument('-e', '--evaluation_id', default='main', help='ID of parcellation to use')
    argparser.add_argument('-p', '--parcellation_id', default='main', help='ID of parcellation to use')
    args = argparser.parse_args()

    SvN_by_subject = {}
    LANG_sub1_by_subject = {}
    LANA_sub1_by_subject = {}
    for path in args.config_paths:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        subject = int(os.path.basename(os.path.dirname(path)).split('_')[0])
        print(subject)

        SvN_path = cfg['evaluate'][args.evaluation_id]['evaluation_atlases'].get('Lang_S-N', None)
        if SvN_path:
            if subject not in SvN_by_subject:
                SvN_by_subject[subject] = set()
            SvN_by_subject[subject].add(SvN_path)

        LANG_sub1_path = os.path.join(cfg['output_dir'], 'parcellation', args.parcellation_id, 'LANG_sub1.nii.gz')
        if os.path.exists(LANG_sub1_path):
            if subject not in LANG_sub1_by_subject:
                LANG_sub1_by_subject[subject] = set()
            LANG_sub1_by_subject[subject].add(LANG_sub1_path)

        LANA_sub1_path = os.path.join(cfg['output_dir'], 'parcellation', args.parcellation_id, 'LANA_sub1.nii.gz')
        if os.path.exists(LANA_sub1_path):
            if subject not in LANA_sub1_by_subject:
                LANA_sub1_by_subject[subject] = set()
            LANA_sub1_by_subject[subject].add(LANA_sub1_path)

    print(SvN_by_subject)
    print(LANG_sub1_by_subject)
    print(LANA_sub1_by_subject)
