import os
import re
import copy
import yaml
import numpy as np
import pandas as pd
import argparse

min_runs = {
    'all': 100,
    'nolangloc': 100,
    'nonlinguistic': 15,
    'onlylangloc': 0
}

K = 50
RUNCOUNTS = (1, 2, 5, 10, 25, 50)
N_NETWORKS = 100

SESSION_MATCH = re.compile(r'SUBJECTS/([^\/]+)/')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Generate multisession parcellate runs from existing configs')
    argparser.add_argument('-c', '--config_dir', default='cfg', help='Directory in which to dump config files')
    args = argparser.parse_args()

    config_dir = args.config_dir
    runs_by_subject = {}

    evo = pd.read_csv('/nese/mit/group/evlab/u/cshain/results/fMRI_parcellate/derivatives/stability_nolangloc/SvN_even_vs_odd.csv')
    evo.subject = evo.subject.astype(str)
    
    for experiment in ('nolangloc',):
        cfg_by_subject = {}
        sessions_by_subject = {}

        for session in os.listdir(os.path.join(config_dir, experiment)):
            subject = session.split('_')[0]
            with open(os.path.join(config_dir, experiment, session), 'r') as f:
                cfg = yaml.safe_load(f)
            for evaluation in cfg['evaluate']['main'].get('evaluation_atlases', {}):
                cfg['evaluate']['main']['evaluation_atlases'][evaluation] = set([cfg['evaluate']['main']['evaluation_atlases'][evaluation]])
            if subject not in runs_by_subject:
                runs_by_subject[subject] = 0
            runs_by_subject[subject] += len(cfg['sample']['main']['functional_paths'])
            if subject in cfg_by_subject:
                cfg_by_subject[subject]['sample']['main']['functional_paths'].extend(cfg['sample']['main']['functional_paths'])
                for evaluation in cfg['evaluate']['main'].get('evaluation_atlases', {}):
                    if evaluation not in cfg_by_subject[subject]['evaluate']['main']['evaluation_atlases']:
                        cfg_by_subject[subject]['evaluate']['main']['evaluation_atlases'][evaluation] = set()
                    for atlas in cfg['evaluate']['main']['evaluation_atlases'][evaluation]:
                        cfg_by_subject[subject]['evaluate']['main']['evaluation_atlases'][evaluation].add(atlas)
            else:
                cfg_by_subject[subject] = cfg
            if subject in sessions_by_subject:
                sessions_by_subject[subject] = sessions_by_subject[subject] + 1
            else:
                sessions_by_subject[subject] = 1
        config_dir_out = os.path.join(config_dir, experiment + '_multisession')
        
        for subject in cfg_by_subject:
            cfg = cfg_by_subject[subject]
            output_dir = cfg_by_subject[subject]['output_dir']
            output_dir_dir = os.path.dirname(output_dir) + '_multisession'
            output_dir = os.path.join(output_dir_dir, subject)
            cfg['output_dir'] = output_dir
            _subject = str(int(subject))
            scores = evo[evo.subject == subject]
            if len(scores) > 0:
                scores = scores.sort_values('r_raw', ascending=False)
                score = scores.iloc[0]
                score, even_path = float(score.r_raw), score.even_path
            else:
                score = 0
                even_path = None
            cfg['score'] = score
            for evaluation in cfg['evaluate']['main'].get('evaluation_atlases', {}):
                is_lang = False
                if 'Lang_S-N' in evaluation or 'Lang_ODD_S-N' in evaluation or 'Lang_EVEN_S-N' in evaluation:
                    is_lang = True
                atlases = sorted(list(cfg['evaluate']['main']['evaluation_atlases'][evaluation]))
                if even_path and is_lang:
                    session = SESSION_MATCH.search(even_path)
                    atlas = None
                    for _atlas in atlases:
                        if session.group(1) in _atlas:
                            atlas = _atlas
                            break
                    assert atlas, 'No atlas found for %s in %s' % (evaluation, atlases)
                else:
                    atlas = sorted(list(cfg['evaluate']['main']['evaluation_atlases'][evaluation]))[0]
                cfg['evaluate']['main']['evaluation_atlases'][evaluation] = atlas

        subjects_sorted = sorted(
            [x for x in cfg_by_subject.keys() if runs_by_subject[x] >= K],
            key=lambda x: cfg_by_subject[x]['score'], reverse=True
        )

        for i, subject in enumerate(subjects_sorted):
            cfg = cfg_by_subject[subject]
            del cfg['score']
            del cfg['grid']
            if 'aggregate' in cfg:
                del cfg['aggregate']
            for runcount in RUNCOUNTS:
                _cfg = copy.deepcopy(cfg)
                assert len(_cfg['sample']['main']['functional_paths']) >= runcount, 'Not enough runs for %s' % subject
                _cfg['sample']['main']['functional_paths'] = _cfg['sample']['main']['functional_paths'][:runcount]
                _cfg['sample']['main']['n_networks'] = N_NETWORKS
                _cfg['output_dir'] = _cfg['output_dir'] + '_run%02d' % runcount
                if not os.path.exists(config_dir_out):
                    os.makedirs(config_dir_out)
                with open(os.path.join(config_dir_out, subject + '_' + experiment + '_run%02d' % runcount + '_multisession.yml'), 'w') as f:
                    yaml.safe_dump(_cfg, f, sort_keys=False)

