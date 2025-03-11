import os
import math
import yaml


CFG_PATH = '../../parcellate/cfg/'
NDEV = 100

def process_cfg(path, run_type, dev_type):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    cfg['output_dir'] = cfg['output_dir'].replace(run_type, '%s%s' % (run_type, dev_type))
    del cfg['sample']['main']['n_networks']
    if dev_type == 'Main':
        cfg['grid'] = {'n_networks': list(range(5, 101, 5))}
    elif dev_type == 'DevNobpTimecourse':
        cfg['sample']['low_pass'] = None
        cfg['sample']['high_pass'] = None
        cfg['sample']['use_connectivity_profile'] = False
    elif dev_type == 'DevBpTimecourse':
        cfg['sample']['low_pass'] = 0.1
        cfg['sample']['high_pass'] = 0.01
        cfg['sample']['use_connectivity_profile'] = False
    elif dev_type == 'DevConnRegions':
        cfg['sample']['low_pass'] = 0.1
        cfg['sample']['high_pass'] = 0.01
        cfg['sample']['use_connectivity_profile'] = True
        cfg['sample']['use_connectivity_to_regions'] = True
        cfg['sample']['binarize_connectivity'] = False
    elif dev_type == 'DevConnRegionsBin':
        cfg['sample']['low_pass'] = 0.1
        cfg['sample']['high_pass'] = 0.01
        cfg['sample']['use_connectivity_profile'] = True
        cfg['sample']['use_connectivity_to_regions'] = True
        cfg['sample']['binarize_connectivity'] = True
    elif dev_type == 'DevConnDownsample':
        cfg['sample']['low_pass'] = 0.1
        cfg['sample']['high_pass'] = 0.01
        cfg['sample']['target_affine'] = (4, 4, 4)
        cfg['sample']['use_connectivity_profile'] = True
        cfg['sample']['use_connectivity_to_regions'] = False
        cfg['sample']['binarize_connectivity'] = False
    elif dev_type == 'DevConnDownsampleBin':
        cfg['sample']['low_pass'] = 0.1
        cfg['sample']['high_pass'] = 0.01
        cfg['sample']['target_affine'] = (4, 4, 4)
        cfg['sample']['use_connectivity_profile'] = True
        cfg['sample']['use_connectivity_to_regions'] = False
        cfg['sample']['binarize_connectivity'] = True
    else:
        raise ValueError('Unknown dev_type: %s' % dev_type)

    return cfg
    

if __name__ == '__main__':
    sessions_by_subject_nonling = {}
    in_dir = os.path.join(CFG_PATH, 'nonlinguistic')
    for path in os.listdir(in_dir):
        if path.endswith('.yml'):
            subject = path.split('_')[0]
            if subject not in sessions_by_subject_nonling:
                sessions_by_subject_nonling[subject] = []
            sessions_by_subject_nonling[subject].append(os.path.join(in_dir, path))
    sessions_by_subject_nolang = {}
    in_dir = os.path.join(CFG_PATH, 'nolangloc')
    for path in os.listdir(in_dir):
        if path.endswith('.yml'):
            subject = path.split('_')[0]
            if subject not in sessions_by_subject_nolang:
                sessions_by_subject_nolang[subject] = []
            sessions_by_subject_nolang[subject].append(os.path.join(in_dir, path))

    subjects = sorted([x for x in sessions_by_subject_nonling if len(sessions_by_subject_nonling[x]) == 1 and len(sessions_by_subject_nolang[x]) == 1])

    step = len(subjects) / (NDEV - 1)

    subjects_dev = []
    for i, subject in enumerate(subjects):
        if int(math.floor(i % step)) == 0 or subject == '497':
            subjects_dev.append(subject)

    for sessions_by_subject, run_type in zip((sessions_by_subject_nolang, sessions_by_subject_nonling), ('nolangloc', 'nonlinguistic')):
        in_dir = os.path.join(CFG_PATH, run_type)
        for session in os.listdir(in_dir):
            print(session)
            subject = session.split('_')[0]
            _session = os.path.join(in_dir, session)
            if subject in subjects_dev:
                dev_types = ('DevNobpTimecourse', 'DevBpTimecourse', 'DevConnRegions', 'DevConnRegionsBin', 'DevConnDownsample', 'DevConnDownsampleBin')
                assert len(sessions_by_subject[subject]) == 1, 'Got too many sessions: %s' % sessions_by_subject[subject]
            else:
                dev_types = ('Main',)
            for dev_type in dev_types:
                out_dir = os.path.join(CFG_PATH, '%s%s' % (run_type, dev_type))
                out_path = os.path.join(out_dir, session)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                cfg = process_cfg(_session, run_type, dev_type)
                with open(out_path.replace('%s.yml' % run_type, '%s%s.yml' % (run_type, dev_type)), 'w') as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

        
