import re
import os
import yaml
import pandas as pd
from langlocfc.initialize import parse_cfg

SUBJECTS = os.path.join(os.sep, 'orcd', 'archive', 'evelina9', '001', 'u', 'Shared', 'SUBJECTS')

rs_metadata = pd.read_csv('evlab_resting_state_scans.csv')
rs_metadata = rs_metadata[rs_metadata['ExperimentValues::IPS'] == 150]

sessions = rs_metadata['Subjects::UniqueID'].astype(str).str.zfill(3) + '_' + rs_metadata['ScanSessions::SessionID'].str.split().str[0] + '_PL2017'
sessions = set(sessions.to_list())

try:
    with open('data_path.txt', 'r') as f:
        base_path = f.read().strip()
except FileNotFoundError:
    sys.stderr.write('Data path not set. Run `python -m langlocfc.set_data_path` before running any other scripts.\n')
    sys.stderr.flush()
    exit()

study_sessions = set()
expt_path = os.path.join(base_path, 'derivatives', 'nolangloc')
for session_path in os.listdir(expt_path):
    if os.path.exists(os.path.join(expt_path, session_path, 'config.yml')):
        study_sessions.add(session_path)
sessions &= study_sessions

selected = []
for session in sessions:
    session_path = os.path.join(SUBJECTS, session)
    cfg_path = os.path.join(session_path, 'data.cfg')
    if not os.path.exists(cfg_path):
        continue
    functionals = parse_cfg(cfg_path)['functionals']
    if isinstance(functionals, str):
        functionals = functionals.split()
    assert isinstance(functionals, list), 'Expected functionals to be type `list`, got %s' % type(functionals).__name__
    functionals_ = []
    for functional in functionals:
        try:
            functional = int(functional)
        except ValueError:
            functional = re.search('-(\d+)\.nii', functional).group(1)
        functionals_.append(functional)
    functionals = functionals_
    dicom_summary_path = os.path.join(session_path, 'dicom_summary.csv')
    if not os.path.exists(dicom_summary_path):
        continue
    dicoms = pd.read_csv(dicom_summary_path, header=0, names=['RUN_NUM', 'IPS', 'NAME', 'OTHER'])
    ips_by_dicom = dicoms.set_index('RUN_NUM')['IPS'].to_dict()
    dicom_by_ips = {}
    for dicom in ips_by_dicom:
        ips = ips_by_dicom[dicom]
        if not ips in dicom_by_ips:
            dicom_by_ips[ips] = []
        dicom_by_ips[ips].append(dicom)
    rs_candidates = set(dicom_by_ips[150]) & set(functionals)
    if len(rs_candidates) != 1:
        continue
    rs_dicom_ix = rs_candidates.pop()
    rs_functional_ix = functionals.index(rs_dicom_ix) + 1

    if not 179 in dicom_by_ips:
        continue
    svn_candidates = dicom_by_ips[179]
    if not len(svn_candidates):
        continue
    svn_candidates = set([functionals.index(x) + 1 for x in svn_candidates if x in functionals])
    cat_paths = sorted([os.path.join(session_path, x) for x in os.listdir(session_path) if x.endswith('.cat') and 'langlocSN' in x], key=len)
    if not len(cat_paths):
        continue
    cat_path = cat_paths[0]
    runs = parse_cfg(cat_path)['runs']
    if not len(runs) > 1:
        continue
    svn_functional_ix = int(runs[0])
    if svn_functional_ix not in svn_candidates:
        continue
    svn_functional_oos_ix = int(runs[1])
    if svn_functional_oos_ix not in svn_candidates:
        continue

    row = dict(
        subject=session.split('_')[0],
        session=session,
        rest=os.path.join(SUBJECTS, session, 'Parcellate', 'func', 'sdwrfunc_run-%02d_bold.nii' % rs_functional_ix),
        task=os.path.join(SUBJECTS, session, 'Parcellate', 'func', 'sdwrfunc_run-%02d_bold.nii' % svn_functional_ix),
        task_oos=os.path.join(SUBJECTS, session, 'Parcellate', 'func', 'sdwrfunc_run-%02d_bold.nii' % svn_functional_oos_ix)
    )
    selected.append(row)

selected = sorted(selected, key=lambda x: int(x['subject']))

for session in selected:
    for train_type in ['rest', 'task', 'task_oos']:
        name = session['session'] + '_' + train_type
        cfg_dir = os.path.join('cfg', 'rest_v_task')
        cfg_path = os.path.join(cfg_dir, name + '.yml')
        with open(os.path.join(expt_path, session['session'], 'config.yml'), 'r') as f:
            cfg = yaml.safe_load(f)
        del cfg['grid']
        out_dir = cfg['output_dir'].replace('nolangloc', 'rest_v_task') + '_' + train_type
        cfg['output_dir'] = out_dir
        cfg['sample']['main']['functional_paths'] = [session[train_type]]
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False)



