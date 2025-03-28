import sys
import os
import copy
import yaml
import numpy as np
import pandas as pd
from nilearn import image, masking, glm
import argparse

sys.path.append('parcellate')
from parcellate.data import InputData, detrend_array, standardize_array
from parcellate.util import get_action_attr

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Compare network correlations to task responses''')
    argparser.add_argument('config_paths', nargs='+', help='Paths to task regression config files')
    argparser.add_argument('-f', '--fisher', action='store_true', help='Whether to use Fisher (vs arithmetic) average of correlations')
    args = argparser.parse_args()

    eps = 1e-3

    for config_path in args.config_paths:
        with open(config_path, 'r') as f:
            cfg_tr = yaml.safe_load(f)
        spm_path = cfg_tr['spm_path']
        sess_name = cfg_tr['sess_name']
        results_dir = cfg_tr['results_dir']
        parcellate_output_dir = cfg_tr['parcellate_output_dir']
        output_dir = cfg_tr['output_dir']
        sample_id = cfg_tr['sample_id']
        parcellation_id = cfg_tr['parcellation_id']
        fc_all = cfg_tr['fc_all']
        cfg_all = cfg_tr['cfg_all']
        run_seq = cfg_tr['run_seq']
        dfs = cfg_tr['dfs']
        dfs_rand = []
        for i in range(len(dfs)):
            dfs[i] = pd.DataFrame(dfs[i])
            _df = dfs[i].copy()
            _df.trial_type = np.random.permutation(_df.trial_type)
            dfs_rand.append(_df)

        nii = None
        functionals = None
        predicted = None
        unresidualized_filenames = None
        residualized_filenames = None
        fitted = False
        for fc, cfg in zip(fc_all, cfg_all):
            with open(cfg, 'r') as f:
                cfg = yaml.safe_load(f)
            fc_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(fc))))
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            if output_dir:
                _output_dir = os.path.join(output_dir, os.path.basename(cfg['output_dir']))
            else:
                _output_dir = cfg['output_dir']
            if parcellate_output_dir:
                _parcellate_output_dir = os.path.join(parcellate_output_dir,
                                                      os.path.basename(cfg['output_dir']))
            else:
                _parcellate_output_dir = None
            _cfg_opt_path = os.path.join(_output_dir, 'parcellation', parcellation_id,
                                         'parcellate_kwargs_optimized.yml')
            with open(_cfg_opt_path, 'r') as f:
                _cfg_opt = yaml.safe_load(f)
            _cfg = copy.deepcopy(cfg)
            del _cfg['grid']
            del _cfg['aggregate']
            #_cfg['sample'][sample_id]['n_networks'] = get_action_attr('sample', _cfg_opt['action_sequence'], 'kwargs')[
            #    'n_networks']
            _cfg['sample'][sample_id]['n_networks'] = 100

            if not fitted:
                print('Fitting firstlevels for %s' % sess_name)
                data_cfg = cfg['sample'][sample_id].copy()
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
                kwargs['low_pass'] = None
                kwargs['high_pass'] = None
                kwargs['detrend'] = False
                kwargs['standardize'] = False
                data = InputData(**kwargs)
                mask_img = image.new_img_like(data.nii_ref, data.mask)
                functionals = [data.unflatten(x) for x in data.functionals]
                firstlevel_kwargs = dict(
                    t_r=data_cfg.get('tr', 2),
                    mask_img=mask_img,
                    hrf_model="spm",
                    signal_scaling=False,
                    drift_model=None,
                    high_pass=None,
                    minimize_memory=False
                )
                firstlevel = glm.first_level.FirstLevelModel(**firstlevel_kwargs)
                firstlevel_rand = glm.first_level.FirstLevelModel(**firstlevel_kwargs)
                for i in range(len(dfs)):
                    dfs[i].onset = dfs[i].onset * data_cfg.get('tr', 2)
                    dfs[i].duration = dfs[i].duration * data_cfg.get('tr', 2)
                    dfs_rand[i].onset = dfs_rand[i].onset * data_cfg.get('tr', 2)
                    dfs_rand[i].duration = dfs_rand[i].duration * data_cfg.get('tr', 2)
                firstlevel.fit(functionals, events=dfs)
                firstlevel_rand.fit(functionals, events=dfs_rand)

                predicted = firstlevel.predicted
                residuals = firstlevel.residuals
                residuals_rand = firstlevel_rand.residuals

                unresidualized_filenames = []
                for ix, _nii in enumerate(functionals):
                    nii_dir = os.path.join(results_dir, 'data', sess_name)
                    if not os.path.exists(nii_dir):
                        os.makedirs(nii_dir)
                    filename = os.path.join(nii_dir, 'func_unresidualized_%d.nii.gz' % (ix + 1))
                    unresidualized_filenames.append(filename)
                    _nii.to_filename(filename)
                residualized_filenames = []
                for ix, _nii in enumerate(residuals):
                    nii_dir = os.path.join(results_dir, 'data', sess_name)
                    if not os.path.exists(nii_dir):
                        os.makedirs(nii_dir)
                    filename = os.path.join(nii_dir, 'func_residualized_%d.nii.gz' % (ix + 1))
                    residualized_filenames.append(filename)
                    _nii.to_filename(filename)
                residualized_rand_filenames = []
                for ix, _nii in enumerate(residuals_rand):
                    nii_dir = os.path.join(results_dir, 'data', sess_name)
                    if not os.path.exists(nii_dir):
                        os.makedirs(nii_dir)
                    filename = os.path.join(nii_dir, 'func_residualizedRand_%d.nii.gz' % (ix + 1))
                    residualized_rand_filenames.append(filename)
                    _nii.to_filename(filename)

                _cfg['output_dir'] = os.path.join(parcellate_output_dir, 'unresidualized', sess_name)
                _cfg['sample'][sample_id]['functional_paths'] = unresidualized_filenames
                parcellate_cfg_dir = os.path.join(results_dir, 'parcellate_cfg', 'unresidualized')
                if not os.path.exists(parcellate_cfg_dir):
                    os.makedirs(parcellate_cfg_dir)
                _cfg_path = os.path.join(parcellate_cfg_dir, '%s_unresidualized.yml' % sess_name)
                with open(_cfg_path, 'w') as f:
                    yaml.safe_dump(_cfg, f)
                _cfg['output_dir'] = os.path.join(parcellate_output_dir, 'residualized', sess_name)
                _cfg['sample'][sample_id]['functional_paths'] = residualized_filenames
                parcellate_cfg_dir = os.path.join(results_dir, 'parcellate_cfg', 'residualized')
                if not os.path.exists(parcellate_cfg_dir):
                    os.makedirs(parcellate_cfg_dir)
                _cfg_path = os.path.join(parcellate_cfg_dir, '%s_residualized.yml' % sess_name)
                with open(_cfg_path, 'w') as f:
                    yaml.safe_dump(_cfg, f)
                _cfg['output_dir'] = os.path.join(parcellate_output_dir, 'residualizedRand', sess_name)
                _cfg['sample'][sample_id]['functional_paths'] = residualized_rand_filenames
                parcellate_cfg_dir = os.path.join(results_dir, 'parcellate_cfg', 'residualizedRand')
                if not os.path.exists(parcellate_cfg_dir):
                    os.makedirs(parcellate_cfg_dir)
                _cfg_path = os.path.join(parcellate_cfg_dir, '%s_residualizedRand.yml' % sess_name)
                with open(_cfg_path, 'w') as f:
                    yaml.safe_dump(_cfg, f)

#                functionals = [
#                    #data.unflatten(standardize_array(detrend_array(data.bandpass(image.get_data(x)[data.mask], tr=data.tr, lower=0.01, upper=0.1)))) for x in functionals
#                    image.get_data(x)[data.mask] for x in functionals
#                ]
#                predicted = [
#                    #data.unflatten(standardize_array(detrend_array(data.bandpass(image.get_data(x)[data.mask], tr=data.tr, lower=0.01, upper=0.1)))) for x in predicted
#                    image.get_data(x)[data.mask] for x in predicted
#                ]
#                residuals = [
#                    #data.unflatten(standardize_array(detrend_array(data.bandpass(image.get_data(x)[data.mask], tr=data.tr, lower=0.01, upper=0.1)))) for x in residuals
#                    image.get_data(x)[data.mask] for x in residuals
#                ]
#                residuals_rand = [
#                    #data.unflatten(standardize_array(detrend_array(data.bandpass(image.get_data(x)[data.mask], tr=data.tr, lower=0.01, upper=0.1)))) for x in residuals_rand
#                    image.get_data(x)[data.mask] for x in residuals_rand
#                ]
                
                fitted = True

            network = image.math_img('img > 0.5', img=image.load_img(fc))
            network_sel = image.get_data(network).astype(bool)
            nii_in = []
            nii_pred = []
            nii_res = []
            for _nii_in, _nii_pred, _nii_res in zip(functionals, predicted, residuals):
                _nii_in = image.get_data(_nii_in)[network_sel]
                _nii_in = standardize_array(_nii_in)
                nii_in.append(_nii_in)
                _nii_pred = image.get_data(_nii_pred)[network_sel]
                _nii_pred = standardize_array(_nii_pred)
                nii_pred.append(_nii_pred)
                _nii_res = image.get_data(_nii_res)[network_sel]
                _nii_res = standardize_array(_nii_res)
                nii_res.append(_nii_res)
            nii_in = np.concatenate(nii_in, axis=-1)
            nii_pred = np.concatenate(nii_pred, axis=-1)
            nii_res = np.concatenate(nii_res, axis=-1)
            T = nii_in.shape[-1]
            R_task = np.sum(nii_in * nii_pred, axis=-1) / T
            if args.fisher:
                R_task = np.arctanh(R_task * (1 - eps))
            R_task = pd.Series(R_task)
            R_task2res = np.sum(nii_in * nii_res, axis=-1) / T
            if args.fisher:
                R_task2res = np.arctanh(R_task2res * (1 - eps))
            R_task2res = pd.Series(R_task2res)
            R_fc = np.dot(nii_in, nii_in.T) / T
            R_fc = R_fc[np.tril_indices(R_fc.shape[0], k=-1)]
            if args.fisher:
                R_fc = np.arctanh(R_fc * (1 - eps))
            R_fc = pd.Series(R_fc)
            R_fcres = np.dot(nii_res, nii_res.T) / T
            R_fcres = R_fcres[np.tril_indices(R_fcres.shape[0], k=-1)]
            if args.fisher:
                R_fcres = np.arctanh(R_fcres * (1 - eps))
            R_fcres = pd.Series(R_fcres)
            n_voxels = network_sel.sum()
            task_mean = R_task.mean()
            task_sem = R_task.sem()
            fc_mean = R_fc.mean()
            fc_sem = R_fc.sem()
            task2res_mean = R_task2res.mean()
            task2res_sem = R_task2res.sem()
            fcres_mean = R_fcres.mean()
            fcres_sem = R_fcres.sem()
            out = pd.DataFrame([dict(
                network_nii_path=fc,
                spm_path=spm_path,
                trs=T,
                n_voxels=n_voxels,
                task_zR=task_mean,
                task_zR_sem=task_sem,
                fc_zR=fc_mean,
                fc_zR_sem=fc_sem,
                task2res_zR=task2res_mean,
                task2res_zR_sem=task2res_sem,
                fcres_zR=fcres_mean,
                fcres_zR_sem=fcres_sem,
            )])
            task_corr_dir = os.path.join(results_dir, 'task_corr')
            if not os.path.exists(task_corr_dir):
                os.makedirs(task_corr_dir)
            outfile = os.path.join(task_corr_dir, '%s_task_corr.csv' % fc_name)
            out.to_csv(outfile, index=False)

        done_dir = os.path.join(results_dir, 'done')
        if not os.path.exists(done_dir):
            os.makedirs(done_dir)
        done_file = os.path.join(done_dir, '%s.txt' % sess_name)
        with open(done_file, 'w') as f:
            f.write('done')
