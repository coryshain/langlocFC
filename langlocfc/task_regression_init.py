import os
from pymatreader import read_mat
import yaml
import pandas as pd
import argparse

from langlocfc.initialize import parse_cfg

bash = '''\
#!/bin/bash
#SBATCH --job-name={sess_name}_task_regression
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --ntasks=4
#SBATCH --output="{sess_name}_task_regression_%N_%j.out"
{partition}

python -m langlocfc.task_regression {config_path} -f
'''

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Compare network correlations to task responses''')
    argparser.add_argument('config_paths', nargs='+', help='Paths to by-subject config files')
    argparser.add_argument('-e', '--evaluation_id', default='main', help='ID of evaluation to use')
    argparser.add_argument('-p', '--parcellation_id', default='main', help='ID of parcellation to use')
    argparser.add_argument('-s', '--sample_id', default='main', help='ID of sample to use')
    argparser.add_argument('-o', '--output_dir', default=None, help='Prefix to use for parcellation output directory')
    argparser.add_argument('-P', '--parcellate_output_dir', default=None, help='Prefix to use for parcellation output directory')
    argparser.add_argument('-O', '--overwrite', action='store_true', help='Overwrite existing jobs')
    argparser.add_argument('--partition', help='SLURM partition to use')
    args = argparser.parse_args()

    task = 'Lang_S-N'
    network = 'LANA_sub1'
    try:
        with open('data_path.txt', 'r') as f:
            base_path = f.read().strip()
    except FileNotFoundError:
        sys.stderr.write(
            'Data path not set. Run `python -m langlocfc.set_data_path` before running any other scripts.\n')
        sys.stderr.flush()
        exit()

    results_dir = os.path.join(base_path, 'derivatives', 'task_regression')

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
                spm_to_cfg[spm_path].append(config_path)

    for spm_path in spm_paths:
        spm_sess_ = None
        spm_sess = spm_path
        while not spm_sess.endswith('SUBJECTS'):
            spm_sess_ = spm_sess
            spm_sess = os.path.dirname(spm_sess)
        spm_sess = spm_sess_
        sess_name = os.path.basename(spm_sess)
        if not args.overwrite and os.path.exists('%s_task_regression.pbs' % sess_name):
            continue
        print('Processing', sess_name)
        SPM = read_mat(spm_path)['SPM']
        cfg_path = os.path.join(spm_sess, 'data.cfg')
        cfg = parse_cfg(cfg_path)
        if 'functionals' in cfg:
            functionals = cfg['functionals']
            if not isinstance(functionals, list):
                if functionals.endswith('.dcm'):
                    functionals = [os.path.basename(functionals).split('-')[-1]]
                else:
                    functionals = functionals.split()
            else:
                functionals = [os.path.basename(x).split('-')[-2] for x in functionals]
            run_map = [int(x) for x in functionals]
        else:
            continue
        run_map = {x: i + 1 for i, x in enumerate(run_map)}
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
        sessions = SPM['Sess']['U']
        if not isinstance(sessions, list):
            sessions = [sessions]
        for session in sessions:
            df = []
            for i in range(len(session['name'])):
                name = session['name'][i]
                ons = session['ons'][i]
                dur = session['dur'][i]
                try:
                    len(ons)
                except TypeError:
                    ons = [ons]
                    dur = [dur]
                _df = pd.DataFrame(dict(onset=ons, duration=dur, trial_type=name))
                df.append(_df)
            df = pd.concat(df)
            df = df.sort_values('onset')
            df = df.to_dict(orient='records')
            dfs.append(df)

        out = dict(
            spm_path=spm_path,
            sess_name=sess_name,
            results_dir=results_dir,
            output_dir=args.output_dir,
            parcellate_output_dir=args.parcellate_output_dir,
            sample_id=args.sample_id,
            parcellation_id=args.parcellation_id,
            fc_all=spm_to_fc[spm_path],
            cfg_all=spm_to_cfg[spm_path],
            run_seq=run_seq,
            dfs=dfs
        )

        cfg_dir = os.path.join(results_dir, 'cfg')
        if not os.path.exists(cfg_dir):
            os.makedirs(cfg_dir)
        results_path = os.path.join(cfg_dir, '%s_task_regression.yml' % sess_name)
        with open(results_path, 'w') as f:
            yaml.safe_dump(out, f)

        filename = '%s_task_regression.pbs' % sess_name
        if args.partition:
            partition = '#SBATCH --partition={partition}'.format(partition=args.partition)
        else:
            partition = ''
        outstr = bash.format(
            sess_name=sess_name,
            config_path=results_path,
            partition=partition
        )
        with open(filename, 'w') as f:
            f.write(outstr)




