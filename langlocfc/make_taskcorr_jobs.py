import os
import argparse

bash = '''\
#!/bin/bash
#SBATCH --job-name=taskcorr_{subject}
#SBATCH --time=04:00:00
#SBATCH --mem=32GB
#SBATCH --output="taskcorr_{subject}_%N_%j.out"
{partition}

python -m langlocfc.task_corr {config_path} -f -o ../../results/fMRI_parcellate/nolangloc
'''

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Make SLURM jobs to analyze task correlations')
    argparser.add_argument('config_paths', nargs='+', help='Paths to config files')
    argparser.add_argument('--partition', help='SLURM partition to use')
    args = argparser.parse_args()

    for config_path in args.config_paths:
        subject = os.path.basename(config_path)[:-4]
        outfile = 'taskcorr_%s.pbs' % subject
        if args.partition:
           partition = '#SBATCH --partition={partition}'.format(partition=args.partition)
        outstr = bash.format(
            subject=subject,
            config_path=config_path,
            partition=partition
        )
        with open(outfile, 'w') as o:
            o.write(outstr)
    
