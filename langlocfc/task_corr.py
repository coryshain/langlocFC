import sys
import os
import copy
from pymatreader import read_mat
import yaml
import numpy as np
import pandas as pd
from nilearn import image, masking, glm
import argparse

from langlocfc.initialize import parse_cfg

sys.path.append('parcellate')
from parcellate.data import InputData, standardize_array
from parcellate.util import get_action_attr

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Compare network correlations to task responses''')
    argparser.add_argument('config_paths', nargs='+', help='Paths to by-subject config files')
    argparser.add_argument('-e', '--evaluation_id', default='main', help='ID of evaluation to use')
    argparser.add_argument('-p', '--parcellation_id', default='main', help='ID of parcellation to use')
    argparser.add_argument('-s', '--sample_id', default='main', help='ID of sample to use')
    argparser.add_argument('-f', '--fisher', action='store_true', help='Whether to use Fisher (vs arithmetic) average of correlations')
    argparser.add_argument('-o', '--output_dir', default=None, help='Prefix to use for parcellation output directory')    
    args = argparser.parse_args()

    task = 'Lang_S-N'
    network = 'LANA_sub1'
    eps = 1e-3
    results_dir = '/nese/mit/group/evlab/u/cshain/results/fMRI_parcellate/task_corr'

    # Collect unique langloc SPM.mat files
    spm_paths = set()
    spm_to_sess = {}
    spm_to_fc = {}
    spm_to_cfg = {}
    for config_path in args.config_paths:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        eval_path = cfg['evaluate'][args.evaluation_id].get('evaluation_atlases', {}).get(task, None)
        if eval_path:
            spm_path = os.path.join(os.path.dirname(eval_path), 'SPM.mat')
            if os.path.exists(spm_path):
                spm_paths.add(spm_path)
                spm_sess = spm_path
                spm_sess_ = None
                while not spm_sess.endswith('SUBJECTS'):
                    spm_sess_ = spm_sess
                    spm_sess = os.path.dirname(spm_sess)
                spm_sess = os.path.basename(spm_sess_)
                if spm_path not in spm_to_sess:
                    spm_to_sess[spm_path] = spm_sess
                if args.output_dir:
                    network_path = os.path.join(args.output_dir, os.path.basename(cfg['output_dir']), 'parcellation', args.parcellation_id, '%s.nii.gz' % network)
                else:
                    network_path = os.path.join(cfg['output_dir'], 'parcellation', args.parcellation_id, '%s.nii.gz' % network)
                if not spm_path in spm_to_fc:
                    spm_to_fc[spm_path] = []
                spm_to_fc[spm_path].append(network_path)
                if not spm_path in spm_to_cfg:
                    spm_to_cfg[spm_path] = []
                spm_to_cfg[spm_path].append(cfg)

    for spm_path in spm_paths:
        SPM = read_mat(spm_path)['SPM']
        spm_sess = spm_path
        spm_sess_ = None
        while not spm_sess.endswith('SUBJECTS'):
            spm_sess_ = spm_sess
            spm_sess = os.path.dirname(spm_sess)
        spm_sess = spm_sess_
        cfg_path = os.path.join(spm_sess, 'data.cfg')
        cfg = parse_cfg(cfg_path)
        run_map = [int(os.path.basename(x).split('-')[1]) for x in cfg['dicoms']]
        run_map = {x: i for i, x in enumerate(run_map)}
        runs = set()
        for p in SPM['xY']['P']:
            if 'DefaultMNI_PlusStructural' in spm_path:
                run = int(p.split('_run-')[-1].split('_')[0])
            else:
                run = run_map[int(p.split('-')[-1].split('.')[0])]
            runs.add(run)
        run_seq = []
        for run in sorted(list(runs)):
            run_path = os.path.join(spm_sess, 'Parcellate', 'func', 'sdwrfunc_run-%02d_bold.nii' % run)
            run_seq.append(run_path)
        dfs = []
        for session in SPM['Sess']['U']:
            df = []
            for i in range(len(session['name'])):
                name = session['name'][i]
                ons = session['ons'][i]
                dur = session['dur'][i]
                _df = pd.DataFrame(dict(onset=ons, duration=dur, trial_type=name))
                df.append(_df)
            df = pd.concat(df)
            df = df.sort_values('onset')
            dfs.append(df)
        for fc, cfg in zip(spm_to_fc[spm_path], spm_to_cfg[spm_path]):
            sess_name = os.path.basename(spm_sess)
            _results_dir = os.path.join(results_dir, sess_name)
            if not os.path.exists(_results_dir):
                os.makedirs(_results_dir)
            if args.output_dir:
                _output_dir = os.path.join(args.output_dir, os.path.basename(cfg['output_dir']))
            else:
                _output_dir = cfg['output_dir']
            _cfg_opt_path = os.path.join(_output_dir, 'parcellation', args.parcellation_id, 'parcellate_kwargs_optimized.yml')
            with open(_cfg_opt_path, 'r') as f:
                _cfg_opt = yaml.safe_load(f)
            _cfg = copy.deepcopy(cfg)
            del _cfg['grid']
            del _cfg['aggregate']
            _cfg['sample'][args.sample_id]['n_networks'] = get_action_attr('sample', _cfg_opt['action_sequence'], 'kwargs')['n_networks']
            
            print('Fitting firstlevels for %s' % fc)
            data_cfg = cfg['sample'][args.sample_id].copy()
            data_cfg['functional_paths'] = run_seq
            kwargs = {}
            for kwarg in (
                'functional_paths',
                'fwhm',
                'resampling_target_nii',
                'mask_path',
                'detrend',
                'standardize',
                'envelope',
                'tr',
                'low_pass',
                'high_pass'
            ):
                if kwarg in data_cfg:
                    kwargs[kwarg] = data_cfg[kwarg]
            kwargs['high_pass'] = 0.01
            data = InputData(**kwargs)
            mask_img = image.new_img_like(data.nii_ref, data.mask)
            nii = [data.unflatten(x) for x in data.functionals]
            network = image.math_img('img > 0.5', img=image.load_img(fc))
            network_sel = image.get_data(network).astype(bool)
            firstlevel = glm.first_level.FirstLevelModel(
                t_r=data_cfg.get('tr', 2), hrf_model='SPM', high_pass=None, drift_model=None, mask_img=mask_img, minimize_memory=False
            )
            firstlevel.fit(nii, events=dfs)
            predicted = firstlevel.predicted
            nii_in = []
            nii_pred = []
            for _nii_in, _nii_pred in zip(nii, predicted):
                _nii_in = image.get_data(_nii_in)[network_sel]
                _nii_in = standardize_array(_nii_in)
                nii_in.append(_nii_in)
                _nii_pred = image.get_data(_nii_pred)[network_sel]
                _nii_pred = standardize_array(_nii_pred)
                nii_pred.append(_nii_pred)
            nii_in = np.concatenate(nii_in, axis=-1)
            nii_pred = np.concatenate(nii_pred, axis=-1)
            T = nii_in.shape[-1]
            R_task = np.sum(nii_in * nii_pred, axis=-1) / T
            if args.fisher:
                R_task = np.arctanh(R_task * (1 - eps))
            R_task = pd.Series(R_task)
            R_fc = np.dot(nii_in, nii_in.T) / T
            R_fc = R_fc[np.tril_indices(R_fc.shape[0], k=-1)]
            if args.fisher:
                R_fc = np.arctanh(R_fc * (1 - eps))
            R_fc = pd.Series(R_fc)
            n_voxels = network_sel.sum()
            task_mean = R_task.mean()
            task_sem = R_task.sem()
            fc_mean = R_fc.mean()
            fc_sem = R_fc.sem()
            out = pd.DataFrame([dict(
                network_nii_path=fc,
                spm_path=spm_path,
                trs=T,
                n_voxels=n_voxels,
                task_zR=task_mean,
                task_zR_sem=task_sem,
                fc_zR=fc_mean,
                fc_zR_sem=fc_sem
            )])
            filenames = []
            for ix, _nii in enumerate(nii):
                nii_dir = os.path.join(_results_dir, 'unresidualized')
                if not os.path.exists(nii_dir):
                    os.makedirs(nii_dir)
                filename = os.path.join(nii_dir, 'func_unresidualized_%d.nii.gz' % (ix + 1))
                filenames.append(filename)
                _nii.to_filename(filename)
            _cfg['output_dir'] = os.path.join(_results_dir, 'unresidualized')
            _cfg['sample'][args.sample_id]['functional_paths'] = filenames
            _cfg_path = os.path.join(_results_dir, '%s_unresidualized.yml' % sess_name)
            with open(_cfg_path, 'w') as f:
                yaml.safe_dump(_cfg, f)
            for ix, _nii in enumerate(firstlevel.residuals):
                nii_dir = os.path.join(_results_dir, 'residualized')
                if not os.path.exists(nii_dir):
                    os.makedirs(nii_dir)
                filename = os.path.join(nii_dir, 'func_residualized_%d.nii.gz' % (ix + 1))
                _nii.to_filename(filename)
            _cfg['output_dir'] = os.path.join(_results_dir, 'residualized')
            _cfg['sample'][args.sample_id]['functional_paths'] = filenames
            _cfg_path = os.path.join(_results_dir, '%s_residualized.yml' % sess_name)
            with open(_cfg_path, 'w') as f:
                yaml.safe_dump(_cfg, f)

            outfile = os.path.join(_results_dir, 'task_corr.csv')
            out.to_csv(outfile, index=False)

