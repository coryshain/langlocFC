import sys
import os
import traceback
import copy
import re
import shutil
import numpy as np
import pandas as pd
import yaml
from pymatreader import read_mat
import h5py
from scipy import io
import argparse

SUBJECT_RE = re.compile('(\d+)_.+_PL2017$')
MODELFILES_RE = re.compile('.*modelfiles_(.+).cfg')
SUBJECTS_DIR = os.path.join(os.sep, 'nese', 'mit', 'group', 'evlab', 'u', 'Shared', 'SUBJECTS')
SUBJECTS = [x for x in os.listdir(SUBJECTS_DIR) if SUBJECT_RE.match(x)]
SUBJECTS = sorted(SUBJECTS, key=lambda x: int(SUBJECT_RE.match(x).group(1)))
df = pd.read_csv('evlab_experiments_2024-06-18.csv')[['Experiment', 'Alternative Names', 'ExperimentType', 'Nonlinguistic']].to_dict('records')
EXPERIMENTS = {x['Experiment']: x['ExperimentType'] for x in df}
NONLINGUISTIC = {x['Experiment']: bool(x['Nonlinguistic']) for x in df}
for x in ['langloc', 'langlocSN', 'SWJNaud', 'SWN', 'SWNlocIPS168_2runs' 'SWNlocIPS168_3runs'] + ['langloc_DiffTasks_%d' % i for i in range(1, 6)]:
    EXPERIMENTS[x] = 'Lang Loc'
    NONLINGUISTIC[x] = False
for x in ('ToMshort',):
    EXPERIMENTS[x] = 'Other Loc'
    NONLINGUISTIC[x] = False
for x in (
        'MDloc',
        'spatialFIN',
        'DyLoc',
        'Cloudy',
        'Cloudy_PL2017_ap3',
        'Cloudy_PL2017_ap3_full',
        'Cloudy_PL2017_ap4',
        'Cloudy_allvers',
        'MathTask',
        'Math_2015onward',
        'Math_ 2015onward',
):
    EXPERIMENTS[x] = 'Other Loc'
    NONLINGUISTIC[x] = True
for x in df:
    if x['Alternative Names'] and not x['Alternative Names'] is np.nan:
        for alt in x['Alternative Names'].split(', '):
            EXPERIMENTS[alt] = x['ExperimentType']
            NONLINGUISTIC[alt] = bool(x['Nonlinguistic'])
CONTRASTS = {}
CONTRAST_TYPES = {}
for x in EXPERIMENTS:
    if 'swn' in x.lower() or EXPERIMENTS[x] == 'Lang Loc':
        CONTRASTS[x] = ['S-N', 'ODD_S-N', 'EVEN_S-N']
        CONTRAST_TYPES[x] = 'Lang'
    elif 'mdloc' in x.lower() or x.lower() in ('spatialfin',):
        CONTRASTS[x] = ['H-E']
        CONTRAST_TYPES[x] = 'SpatWM'
    elif x.lower().startswith('tom'):
        CONTRASTS[x] = ['bel-pho']
        CONTRAST_TYPES[x] = 'ToM'
    elif x.lower() in ('cloudy', 'cloudy_pl2017_ap3', 'cloudy_pl2017_ap3_full', 'cloudy_pl2017_ap4', 'cloudy_pl2017_ap4_full', 'cloudy_allvers'):
        CONTRASTS[x] = ['ment-phys']
        CONTRAST_TYPES[x] = 'ToM_NV'
    elif x.lower() in ('mathfin_2009-2011', 'mathtask', 'math_2015onward', 'math_ 2015onward'):
        CONTRASTS[x] = ['H-E']
        CONTRAST_TYPES[x] = 'Math'
    elif x.lower() in ('dyloc',):
        CONTRASTS[x] = ['Faces-Objects', 'Bodies-Objects', 'Scenes-Objects']
        CONTRAST_TYPES[x] = 'Vis'
    elif x.lower() in ('music_2009', 'music_task', 'musictask_2010'):
        CONTRASTS[x] = ['I-B']
        CONTRAST_TYPES[x] = 'Music'
LANA = pd.read_csv('LanA_sessions.csv')[['Session', 'Experiment']].to_dict('records')
LANA = {x['Session']: x['Experiment'] for x in LANA}
FUNC_SUBDIR = os.path.join('Parcellate', 'func')
DELIM_RE = re.compile('[ \t,]+')
EXPT_NAMES = list(pd.read_csv('evlab_expts.csv').Experiment.unique())
CONFIG_OUT_DIR = 'cfg'
RESULTS_DIR = os.path.join('..', 'results', 'fMRI_parcellation')
N_NETWORKS = 2
MAX_NETWORKS = 100
N_ENSEMBLE = 10
N_SAMPLES = 256
N_SAMPLES_FINAL = 256
N_ALIGNMENTS = 512
HIGH_PASS = 0.01
LOW_PASS = 0.1
EVAL_TYPE = 'con'
CONFIG_NAMES = [
    'all',
    'onlylangloc',
    'nolangloc',
    'nonlinguistic'
]


def get_nii_path(run, subject):
    if run == 'mask':
        filename = 'wc1art_mean_rfunc_run-01_bold.nii'
    else:
        filename = 'sdwrfunc_run-%02d_bold.nii' % run
    return os.path.join(SUBJECTS_DIR, subject, FUNC_SUBDIR, filename)


def get_functional_dicoms(path):
    in_func = False
    with open(path, 'r') as f:
        for line in f:
            if 'functionals' in line:
                in_func = True
            elif in_func:
                return [int(x) for x in DELIM_RE.split(line.strip())]
    return []


def get_expt_names(subject):
    names = []
    cats = [x for x in os.listdir(os.path.join(SUBJECTS_DIR, subject)) if x.endswith('.cat')]
    for name in EXPT_NAMES:
        for cat in cats:
            if name in cat:
                names.append(name)
                break
    return names


def get_langloc_functionals(subject, langloc):
    subject_dir = os.path.join(SUBJECTS_DIR, subject)
    cats = [x for x in os.listdir(subject_dir) if (x.endswith('.cat') and langloc in x)]
    assert len(cats) == 1, 'There should be exactly 1 langloc *.cat file for subject %s. Got %d' % (subject, len(cats))
    functionals = []
    with open(os.path.join(subject_dir, cats[0]), 'r') as f:
        for line in f:
            if 'runs' in line:
                for functional in line.strip().split():
                    try:
                        functional = int(functional)
                        functionals.append(functional)
                    except ValueError:
                        pass
                break
    return functionals


def expand_config(config):
    out = {}

    name = 'masknoalign'
    _config = copy.deepcopy(config)
    _config['align_to_reference'] = False
    _config['output_dir'] = _config['output_dir'] + '_' + name
    out[name] = _config

    name = 'maskalign'
    _config = copy.deepcopy(config)
    _config['align_to_reference'] = True
    _config['output_dir'] = _config['output_dir'] + '_' + name
    out[name] = _config

    name = 'nomasknoalign'
    _config = copy.deepcopy(config)
    del _config['mask']
    _config['align_to_reference'] = False
    _config['output_dir'] = _config['output_dir'] + '_' + name
    out[name] = _config

    name = 'nomaskalign'
    _config = copy.deepcopy(config)
    del _config['mask']
    _config['align_to_reference'] = True
    _config['output_dir'] = _config['output_dir'] + '_' + name
    out[name] = _config

    return out


def parse_cfg(path):
    out = {}
    header = None
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('%'):
                if line.startswith('#'):
                    parts = DELIM_RE.split(line[1:])
                    header = parts[0]
                    if len(parts) > 2:
                        out[header] = parts[1:]
                    elif len(parts) > 1:
                        out[header] = [parts[1]]
                elif header is not None:
                    if header not in out:
                        out[header] = []
                    out[header].append(line)

    for header in out:
        vals = out[header]
        valset = set()
        _vals = []
        for val in vals:
            if val not in valset:
                _vals.append(val)
                valset.add(val)
        if len(_vals) == 1:
            _vals = vals[0]
        out[header] = _vals

    return out
           

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Initialize the fMRI parcellation experiment (data and configs).')
    argparser.add_argument('-m', '--use_mask', action='store_true', help='Use subject-specific mask (else, use nilearn grey matter template).')
    argparser.add_argument('-n', '--dry_run', action='store_true', help='Simulate execution without actually creating or changing any files')
    argparser.add_argument('-c', '--config_dir', default='../../parcellate/cfg', help='Directory in which to dump config files')
    args = argparser.parse_args()

    s = 0
    #s = 1410
    max_s = len(SUBJECTS)
    #max_s = 200
    errors = open('errors.txt', 'w')
    contrast_names = open('contrast_names.txt', 'w')
    subjects_by_contrast = {}
    while s < max_s:
        subject = SUBJECTS[s]
        if not os.path.exists(os.path.join(SUBJECTS_DIR, subject, 'Parcellate', 'func')):
            s += 1
            continue
        found = False
        for path in os.listdir(os.path.join(SUBJECTS_DIR, subject, 'Parcellate', 'func')):
            if path.startswith('sdwrfunc_run-'):
                found = True
                break
        if not found:
            continue
        try:
            if subject in LANA:
                langloc = LANA[subject]
            else:
                participant_id = subject.split('_')[0]
            subject_dir = os.path.join(SUBJECTS_DIR, subject)
            modelfiles_paths_all = [] 
            modelfiles_paths = []
            for path in [os.path.join(subject_dir, x) for x in os.listdir(subject_dir) if MODELFILES_RE.match(x) and not x.startswith('.')]:
                model = parse_cfg(path)
                if 'model_name' not in model or 'design' not in model or not isinstance(model['design'], str) or not os.path.exists(model['design']) or os.path.isdir(model['design']):
                    # Modelfile is ill-formed in some way. Skip
                    continue
                modelfiles_paths_all.append(path)
                if langloc == model['model_name']:
                    modelfiles_paths.append(path)
            if len(modelfiles_paths_all):
                modelfiles_path_all = sorted(modelfiles_paths_all, key=len)
            if len(modelfiles_paths):
                modelfiles_path = sorted(modelfiles_paths, key=len)
                modelfiles_path = modelfiles_paths[0]
            else:
                modelfiles_path = None
            if modelfiles_path:
                model = parse_cfg(modelfiles_path)
                catfile_path = model['design']
            else:
                catfiles = [os.path.join(subject_dir, x) for x in os.listdir(subject_dir) if x.endswith(f'{langloc}.cat')]
                assert len(catfiles) == 1, f'Incorrect number of matching catfiles for {subject_dir}. Expected 1, got {len(catfiles)}.'
                catfile_path = catfiles[0]
                for path in modelfiles_paths_all:
                    model = parse_cfg(path)
                    if os.path.exists(model['design']) and os.path.samefile(catfile_path, model['design']):
                        modelfiles_path = path
                        langloc = model['model_name']
                        break
    
            cat = parse_cfg(catfile_path)
            if 'runs' in cat and isinstance(cat['runs'], str):
                cat['runs'] = [cat['runs']]
            cat['runs'] = [int(x) for x in cat['runs']]
    
            print('SUBJECT #%d: %s' % (s + 1, subject))
            print('  Localizer: %s' % langloc)
            print('  modelfiles path: %s' % modelfiles_path)
            print('  *.cat path: %s' % catfile_path)
 
            # Find IDs of functional runs
            datacfg_path = os.path.join(subject_dir, 'data.cfg')
            dicoms = get_functional_dicoms(datacfg_path)
            functionals = list(range(1, len(dicoms) + 1))
            langloc_functionals = cat['runs']
            nonlinguistic_functionals = set()
    
            if len(functionals) > 0:  # Need at least one functional rune to perform parcellation
                config = {
                    'sample': {
                        'main': {
                            'functional_paths': [],
                            'n_samples': N_SAMPLES,
                            'high_pass': HIGH_PASS,
                            'low_pass': LOW_PASS,
                        }
                    },
                    'align': {
                        'main': {'n_alignments': N_ALIGNMENTS}
                    },
                    'label': {
                        'main': {'reference_atlases': 'all'}
                    },
                    'evaluate': {
                        'main': {}
                    },
                    'aggregate': {
                        'main': {'subnetwork_id': 1}
                    },
                    'parcellate': {
                        'main': {'sample': {'n_samples': N_SAMPLES_FINAL}}
                    },
                    'grid': {
                        'n_networks': [[2, 100]]
                    }
                }
        
                # Grey matter mask
                if args.use_mask:
                    mask_path = get_nii_path('mask', subject)
                    config['sample']['main']['mask'] = mask_path

                for _modelfile_path in modelfiles_paths_all:
                    model_cfg = parse_cfg(_modelfile_path)
                    model_name = model_cfg['model_name']
                    if model_name in NONLINGUISTIC and NONLINGUISTIC[model_name]:
                        cat_path = model_cfg.get('design', None)
                        if cat_path:
                            cat_cfg = parse_cfg(cat_path)
                            if 'runs' in cat_cfg and isinstance(cat_cfg['runs'], str):
                                cat_cfg['runs'] = [cat_cfg['runs']]
                            for run in cat_cfg.get('runs', []):
                                nonlinguistic_functionals.add(run)
                    spm_path = os.path.join(subject_dir, f'firstlevel_{model_name}', 'SPM.mat')
                    if not os.path.exists(spm_path):
                        # Not modeled, skip
                        continue
                    name2ix = {}
                    try:
                        with h5py.File(spm_path, 'r') as f:
                            if 'SPM' in f and 'xCon' in f['SPM'] and 'name' in f['SPM/xCon']:
                                for i in range(len(f['SPM/xCon/name'])):
                                    name = ''.join([chr(x[0]) for x in f[f['SPM/xCon/name'][i, 0]]])
                                    name2ix[name] = i + 1
                            else:
                                # No contrast atlases available for evaluation, skip
                                continue    
                    except OSError:  # Old-style matlab file
                        try:
                            f = io.loadmat(spm_path)
                        except MemoryError:
                            continue
                        if 'SPM' in f and f['SPM'][0][0].dtype.names is not None and 'xCon' in f['SPM'][0][0].dtype.names \
                                and  f['SPM'][0][0]['xCon'].dtype.names is not None and 'name' in f['SPM'][0][0]['xCon'].dtype.names:
                            for i in range(len(f['SPM'][0][0]['xCon']['name'][0])):
                                name = f['SPM'][0][0]['xCon']['name'][0][i][0]
                                name2ix[name] = i + 1
                        else:
                            # No contrast atlases available for evaluation, skip
                            continue    
                    if model_name in CONTRASTS:
                        for contrast in CONTRASTS[model_name]:
                            if contrast not in name2ix:
                                continue
                            contrast_path = '%s_%04d.nii' % (EVAL_TYPE, name2ix[contrast])
                            contrast_path = os.path.join(subject_dir, f'firstlevel_{model_name}', contrast_path)
                            if 'evaluation_atlases' not in config['evaluate']['main']:
                                config['evaluate']['main']['evaluation_atlases'] = {}
                            contrast_type = CONTRAST_TYPES[model_name]
                            _contrast = contrast_type + '_' + contrast
                            if model_name == langloc or _contrast not in config['evaluate']['main']['evaluation_atlases']:
                                if _contrast not in config['evaluate']['main']['evaluation_atlases'] or model_name == langloc:
                                    # The contrast has not been identified for this subject,
                                    # or the current model is the selected langloc expt
                                    config['evaluate']['main']['evaluation_atlases'][_contrast] = contrast_path
                                if _contrast not in subjects_by_contrast:
                                    subjects_by_contrast[_contrast] = []
                                subjects_by_contrast[_contrast].append(subject)
                    else:
                        contrast_names.write(subject + ': ' + model_name + '\n')
                        for x in name2ix:
                            contrast_names.write(f'  {x}\n')
                        contrast_names.write('\n')
                        contrast_names.flush()
 
                for config_name in CONFIG_NAMES:
                    _config = copy.deepcopy(config)
                    if config_name == 'all':
                        _functionals = functionals[:]
                    elif config_name == 'nolangloc':
                        _functionals = [x for x in functionals if not x in langloc_functionals]
                    elif config_name == 'onlylangloc':
                        _functionals = langloc_functionals[:]
                    elif config_name == 'nonlinguistic':
                        _functionals = sorted([int(x) for x in nonlinguistic_functionals])
                    else:
                        print('No non-langloc functional runs. Skipping...')
                        raise ValueError('Unrecognized config type %s' % config_name)
                    if not len(_functionals):
                        continue
                    for i, functional in enumerate(_functionals):
                        nii_path = get_nii_path(functional, subject)
                        _functionals[i] = nii_path
                    _config['sample']['main']['functional_paths'] = _functionals
                    _config['output_dir'] = os.path.join(RESULTS_DIR, config_name, subject)
                    config_dir = os.path.join(args.config_dir, config_name)
                    config_path = os.path.join(config_dir, '%s_%s.yml' % (subject, config_name))
                    print('  Saving config to %s.' % config_path)
                    if not args.dry_run:
                        if not os.path.exists(config_dir):
                            os.makedirs(config_dir)
                        with open(config_path, 'w') as f:
                            yaml.safe_dump(_config, f, sort_keys=False)
        except Exception as e:
            e_str = type(e).__name__ + ': ' + str(e) + '\n' + ''.join(traceback.format_tb(e.__traceback__))
            print(e_str)
            errors.write(subject + '\n')
            errors.write(e_str + '\n\n\n')
            errors.flush()
        except KeyboardInterrupt:
            errors.close()
            contrast_names.close()
            sys.exit()
            pass
            
        s += 1
    errors.close()
    contrast_names.close()
    with open('subjects_by_contrast.txt', 'w') as f:
        for contrast in subjects_by_contrast:
            f.write('%s | N subjects: %d\n' % (contrast, len(subjects_by_contrast[contrast])))
            for subject in subjects_by_contrast[contrast]:
                f.write('  %s\n' % subject)
            f.write('\n')
            f.flush()

