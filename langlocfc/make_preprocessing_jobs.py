import os

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
    done = False
    if os.path.exists(os.path.join(subjects_dir, subject, 'Parcellate', 'func')):
        for path in os.listdir(os.path.join(subjects_dir, subject, 'Parcellate', 'func')):
            if path.startswith('sdwrfunc_run-'):
                done = True
                break
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

