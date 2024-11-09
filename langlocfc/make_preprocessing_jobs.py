import os
import re

date_re = re.compile('_(\d{8})[a-z]?_')

bash = '''\
#!/bin/bash
#SBATCH --job-name=parcellate_preproc_{subject}
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --partition=evlab
#SBATCH --exclude=node[100-116],node033,node094,node105
#SBATCH --output="parcellate_preproc_{subject}_%N_%j.out"

matlab -nodisplay -r "addpath('/om/weka/evlab/shared/software/conn'); conn_module el init; el preprocessing {subject} {cfg};"
'''

subjects_dir = '/nese/mit/group/evlab/u/Shared/SUBJECTS/'
cfg = os.path.join(os.getcwd(), 'pipeline_preproc_Parcellate.cfg')

subjects = [x for x in os.listdir(subjects_dir) if x.endswith('_PL2017')]

for subject in subjects:
    assert os.path.exists(os.path.join(subjects_dir, subject)), 'Directory not found for subject %s' % subject
    date = date_re.search(subject)
    if date:
        date = int(date.group(1))
        if date > 20240800:  # August 2024 or later, skip
            continue
    done = False
    if os.path.exists(os.path.join(subjects_dir, subject, 'Parcellate', 'func')):
        sdwr_max = None
        func_max = None
        for path in os.listdir(os.path.join(subjects_dir, subject, 'Parcellate', 'func')):
            if path.startswith('func_run-'):
                run = int(path[9:11])
                if func_max is None or run > func_max:
                    func_max = run
            if path.startswith('sdwrfunc_run-'):
                run = int(path[13:15])
                if sdwr_max is None or run > sdwr_max:
                    sdwr_max = run
        if sdwr_max is not None and func_max is not None and sdwr_max == func_max:
            done = True
    if done:
        continue
    outfile = 'parcellate_preproc_%s.pbs' % subject
    outstr = bash.format(
        subject=subject,
        subjects_dir=subjects_dir,
        cfg=cfg
    )
    with open(outfile, 'w') as o:
        o.write(outstr)

