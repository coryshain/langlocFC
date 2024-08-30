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
    max_s = len(SUBJECTS)
    errors = open('errors.txt', 'w')
    contrast_names = open('contrast_names.txt', 'w')
    subjects_by_contrast = {}
    contrasts_by_participant = {}
    configs = {}  # Keyed by subject, then by config name

    print('Computing configs...')
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
            s += 1
            continue
        # TODO: Delete this check after tests are passed
        if not subject in LANA:
            s += 1
            continue
        try:
            participant_id = subject.split('_')[0]
            if subject in LANA:
                langloc = LANA[subject]
            else:
                langloc = None

            subject_dir = os.path.join(SUBJECTS_DIR, subject)

            firstlevels = {}
            firstlevel_functionals = {}
            firstlevel_catnames = {}
            langloc_firstlevels = None
            langloc_functionals = []
            nonlinguistic_functionals = set()
            modelfile_paths = []

            # First collect all modelfile paths and grab any that match the langloc experiment
            for path in [os.path.join(subject_dir, x) for x in os.listdir(subject_dir) if MODELFILES_RE.match(x) and not x.startswith('.')]:
                model = parse_cfg(path)
                if 'model_name' not in model or 'design' not in model or not isinstance(model['design'], str) or not os.path.exists(model['design']) or os.path.isdir(model['design']):
                    # Modelfile is ill-formed in some way. Skip
                    continue
                model_name = model['model_name']
                cat_path = model['design']
                cat_name = os.path.basename(cat_path).replace('.cat', '')
                cat = parse_cfg(cat_path)
                runs = cat.get('runs', None)
                if runs is None:
                    # Modelfile points to bad catfile, skip
                    continue
                if isinstance(runs, str):
                    runs = runs.split()
                runs = [int(x) for x in runs]
                if model_name in NONLINGUISTIC and NONLINGUISTIC[model_name]:
                    for run in runs:
                        nonlinguistic_functionals.add(run)
                spm_path = os.path.join(subject_dir, f'firstlevel_{model_name}', 'SPM.mat')
                if not os.path.exists(spm_path):
                    # Not modeled, skip
                    continue
                if (not model_name in firstlevels) or (len(spm_path) < len(firstlevels[model_name])):
                    firstlevels[model_name] = spm_path
                    firstlevel_functionals[model_name] = runs
                    firstlevel_catnames[model_name] = cat_name
                if (model_name == langloc) or (langloc is None and EXPERIMENTS[model_name] == 'Lang Loc'):
                    if (langloc_firstlevels is None) or (len(spm_path) < len(langloc_firstlevels)):
                        langloc_firstlevels = spm_path
                        langloc_functionals = runs
                        langloc = model_name  # Reset in case the name was just inferred above, no-op otherwise

            if not len(firstlevels):
                # No modelfiles found, so possibly a newer subject with a different directory structure. Search
                # DefaultMNIPlusStructural instead
                results_dir = os.path.join(subject_dir, 'DefaultMNI_PlusStructural', 'results', 'firstlevel')
                if os.path.exists(results_dir):
                    for model_name in os.listdir(results_dir):
                        spm_file = os.path.join(results_dir, model_name, 'SPM.mat')
                        if not os.path.exists(spm_file):
                            # Not modeled, skip
                            continue
                        cat_files = sorted([x for x in os.listdir(subject_dir) if x.endswith(f'{model_name}.cat')], key=len)
                        if len(cat_files):
                            cat_file = cat_files[0]
                            cat_path = os.path.join(subject_dir, cat_file)
                            cat_name = cat_file.replace('.cat', '')
                            cat = parse_cfg(cat_path)
                            runs = cat['runs']
                            if isinstance(runs, str):
                                runs = runs.split()
                            runs = [int(x) for x in runs]
                            if model_name in NONLINGUISTIC and NONLINGUISTIC[model_name]:
                                for run in runs:
                                    nonlinguistic_functionals.add(run)
                            if (not model_name in firstlevels) or (len(spm_file) < len(firstlevels[model_name])):
                                firstlevels[model_name] = spm_file
                                firstlevel_functionals[model_name] = runs
                                firstlevel_catnames[model_name] = cat_name
                            if (model_name == langloc) or (langloc is None and EXPERIMENTS[model_name] == 'Lang Loc'):
                                if (langloc_firstlevels is None) or (len(spm_file) < len(langloc_firstlevels)):
                                    langloc_firstlevels = spm_file
                                    langloc_functionals = runs

            if langloc_firstlevels is None:
                # Nothing exactly matched the langloc experiment, so now loop through the firstlevels for any that
                # use a catfile named to match the langloc experiment. If one is found, use it.
                for model_name in firstlevels:
                    if firstlevel_catnames[model_name].endswith(langloc):
                        langloc = model_name
                        langloc_firstlevels = firstlevels[model_name]
                        langloc_functionals = firstlevel_functionals[model_name]
                        break

            print('SUBJECT #%d: %s' % (s + 1, subject))
            print('  Localizer: %s' % langloc)
            print('  SPM path: %s' % langloc_firstlevels)
            print('  Langloc functionals: %s' % ', '.join([str(x) for x in langloc_functionals]))
 
            # Find IDs of functional runs
            datacfg_path = os.path.join(subject_dir, 'data.cfg')
            dicoms = get_functional_dicoms(datacfg_path)
            functionals = list(range(1, len(dicoms) + 1))

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

                for model_name in firstlevels:
                    spm_path = firstlevels[model_name]
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
                                # The contrast has not been identified for this subject,
                                # or the current model is the selected langloc expt
                                config['evaluate']['main']['evaluation_atlases'][_contrast] = contrast_path
                                if participant_id not in contrasts_by_participant:
                                    contrasts_by_participant[participant_id] = {}
                                if _contrast not in contrasts_by_participant[participant_id]:
                                    contrasts_by_participant[participant_id][_contrast] = contrast_path
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
                    if subject not in configs:
                        configs[subject] = {}
                    configs[subject][config_name] = _config
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

    print('Saving configs...')
    for subject in configs:
        print(subject)
        for config_name in configs[subject]:
            participant_id = subject.split('_')[0]
            _config = configs[subject][config_name]
            # Fill in any missing contrasts that are available from other sessions
            for contrast in contrasts_by_participant[participant_id]:
                if contrast not in _config['evaluate']['main']['evaluation_atlases']:
                    _config['evaluate']['main']['evaluation_atlases'][contrast] = contrasts_by_participant[participant_id][contrast]
            config_dir = os.path.join(args.config_dir, config_name)
            config_path = os.path.join(config_dir, '%s_%s.yml' % (subject, config_name))
            print('  Saving config to %s.' % config_path)
            if not args.dry_run:
                if not os.path.exists(config_dir):
                    os.makedirs(config_dir)
                with open(config_path, 'w') as f:
                    yaml.safe_dump(_config, f, sort_keys=False)

    errors.close()
    contrast_names.close()
    with open('subjects_by_contrast.txt', 'w') as f:
        for contrast in subjects_by_contrast:
            f.write('%s | N subjects: %d\n' % (contrast, len(subjects_by_contrast[contrast])))
            for subject in subjects_by_contrast[contrast]:
                f.write('  %s\n' % subject)
            f.write('\n')
            f.flush()

