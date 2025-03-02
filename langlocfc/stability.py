import sys
import os
import yaml
import numpy as np
import pandas as pd
from nilearn import image, masking
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from langlocfc import resources
import argparse

ALL_REFERENCE = [
    'LANG',
    'FPN_A',
    'FPN_B',
    'DN_A',
    'DN_B',
    'CG_OP',
    'SAL_PMN',
    'dATN_A',
    'dATN_B',
    'AUD',
    'PM_PPr',
    'SMOT_A',
    'SMOT_B',
    'VIS_C',
    'VIS_P',
    'LANA',
]
TASKS = ['Lang_S-N', 'Lang_EVEN_S-N', 'Lang_ODD_S-N']
RTYPES = ['raw', 'masked12', 'masked6']

RUN_EVEN_VS_ODD = True
RUN_WITHIN_SUBECTS = True
RUN_BETWEEN_SUBJECTS = True
RUN_BETWEEN_NETWORKS = True

def get_rtype(nii, rtype='raw'):
    if rtype == 'raw':
        return nii
    elif rtype == 'masked12':
        return nii[fROI12]
    elif rtype == 'masked6':
        return nii[fROI6]
    else:
        return [nii[x] for x in fROIs]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Evaluate stability of S-N vs langlocFC''')
    argparser.add_argument('config_paths', nargs='+', help='Paths to by-subject config files')
    argparser.add_argument('-e', '--evaluation_id', default='main', help='ID of parcellation to use')
    argparser.add_argument('-p', '--parcellation_id', default='main', help='ID of parcellation to use')
    argparser.add_argument('-f', '--fisher', action='store_true', help='Fisher-transform correlations')
    argparser.add_argument('-o', '--output_dir', default=None, help='Prefix to use for parcellation output directory')
    args = argparser.parse_args()

    eps = 1e-3

    print('Loading data')
    with pkg_resources.as_file(pkg_resources.files(resources).joinpath('fROI12_SN220.nii')) as path:
        fROI12 = image.smooth_img(path, fwhm=None)
    mask = image.get_data(masking.compute_brain_mask(fROI12, connected=False, opening=False, mask_type='gm')) > 0.5
    fROI12 = image.get_data(fROI12).astype(bool)[mask]

    with pkg_resources.as_file(pkg_resources.files(resources).joinpath('fROI06_SN220.nii')) as path:
        fROI6 = image.smooth_img(path, fwhm=None)
    fROI6 = image.get_data(fROI6).astype(bool)[mask]

    with pkg_resources.as_file(pkg_resources.files(resources).joinpath('allParcels-language-SN220.nii')) as path:
        fROIs = image.smooth_img(path, fwhm=None)
    fROIs = image.get_data(fROIs)[mask]
    fROIs_ = []
    for i in range(6):
        fROIs_.append(fROIs == i + 1)
    fROIs = fROIs_

    nii_paths = {}
    path2nii = {}
    for i, path in enumerate(args.config_paths):
        sys.stderr.write('\r%d/%d' % (i + 1, len(args.config_paths)))
        sys.stderr.flush()
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        subject = os.path.basename(cfg['output_dir']).split('_')[0]
        if args.output_dir:
            output_dir = os.path.join(args.output_dir, os.path.basename(cfg['output_dir']))
        else:
            output_dir = cfg['output_dir']
        parcellation_path = os.path.join(output_dir, 'parcellation', args.parcellation_id)
        for setname in ALL_REFERENCE + TASKS:
            is_task = setname in TASKS
            if is_task:
                filename = 'eval_%s.nii.gz'
                nii_path = cfg['evaluate'][args.evaluation_id]['evaluation_atlases'][setname]
            else:
                nii_path = os.path.join(parcellation_path, '%s_sub1.nii.gz' % setname)

            if is_task:
                sess = nii_path
                sess_ = None
                while not sess.endswith('SUBJECTS'):
                    sess_ = sess
                    sess = os.path.dirname(sess)
                sess = os.path.basename(sess_)
            else:
                sess = os.path.basename(cfg['output_dir'])

            if nii_path not in path2nii and os.path.exists(nii_path):
                nii = image.load_img(nii_path)
                nii = image.get_data(nii)[mask]
                path2nii[nii_path] = nii
            if setname not in nii_paths:
                nii_paths[setname] = {}
            if subject not in nii_paths[setname]:
                nii_paths[setname][subject] = []
            nii_paths[setname][subject].append(dict(path=nii_path, sess=sess))

    print('')

    print('Even vs odd')
    if RUN_EVEN_VS_ODD:
        SvN_even_vs_odd = []
        for subject in nii_paths['Lang_EVEN_S-N']:
            if subject not in nii_paths['Lang_ODD_S-N']:
                continue
            # Filter out any repeats
            niis = {}
            for even, odd in zip(nii_paths['Lang_EVEN_S-N'][subject], nii_paths['Lang_ODD_S-N'][subject]):
                niis[even['path']] = dict(even=even, odd=odd)
            for even_path in niis:
                even = niis[even_path]['even']
                odd = niis[even_path]['odd']
                row = dict(
                    subject=subject,
                    even_path=even['path'],
                    odd_path=odd['path'],
                )
                for rtype in RTYPES:
                    nii_even = get_rtype(path2nii[even['path']], rtype=rtype)
                    nii_odd = get_rtype(path2nii[odd['path']], rtype=rtype)
                    r = np.corrcoef(nii_even, nii_odd)[0, 1]
                    if args.fisher:
                        r = np.arctanh(r * (1 - eps))
                    row['r_%s' % rtype] = r
                SvN_even_vs_odd.append(row)
        SvN_even_vs_odd = pd.DataFrame(SvN_even_vs_odd)
        SvN_even_vs_odd.to_csv('stability_SvN_even_vs_odd.csv', index=False)

    print('Within subjects for Lang_S-N, LANG, LANA')
    if RUN_WITHIN_SUBECTS:
        within_subjects = []
        for setname in ('Lang_S-N', 'LANG', 'LANA'):
            for subject in nii_paths[setname]:
                if len(nii_paths[setname][subject]) < 2:
                    continue
                # Filter out any repeats
                niis = {}
                for nii in nii_paths[setname][subject]:
                    niis[nii['path']] = nii
                niis = list(niis.values())
                row = dict(
                    subject=subject,
                    setname=setname,
                    n_sess=len(niis),
                )
                for rtype in RTYPES:
                    R = []
                    for i in range(len(niis)):
                        for j in range(i + 1, len(niis)):
                            a = get_rtype(path2nii[niis[i]['path']], rtype=rtype)
                            b = get_rtype(path2nii[niis[j]['path']], rtype=rtype)
                            r = np.corrcoef(a, b)[0, 1]
                            if args.fisher:
                                r = np.arctanh(r * (1 - eps))
                            R.append(r)
                    R = pd.Series(R)
                    row['r_%s' % rtype] = R.mean()
                    row['r_%s_sem' % rtype] = R.sem()
                within_subjects.append(row)
        within_subjects = pd.DataFrame(within_subjects)
        within_subjects.to_csv('stability_within_subjects.csv', index=False)

    print('Between subjects for Lang_S-N, LANG, LANA')
    if RUN_BETWEEN_SUBJECTS:
        between_subjects = []
        for setname in ('Lang_S-N', 'LANG', 'LANA'):
            subjects = list(nii_paths[setname].keys())
            row = dict(
                setname=setname,
                n_subj=len(subjects)
            )
            for rtype in RTYPES:
                averaged_statmaps = {}
                for subject in nii_paths[setname]:
                    # Filter out any repeats
                    niis = {}
                    for nii in nii_paths[setname][subject]:
                        niis[nii['path']] = nii
                    niis = list(niis.values())
                    averaged_statmaps[subject] = [
                        np.stack(
                            [get_rtype(path2nii[x['path']], rtype=rtype) for x in niis],
                            axis=-1
                        ).mean(axis=-1)
                    ]
                R = []
                for i in range(len(subjects)):
                    for j in range(i + 1, len(subjects)):
                        r = np.corrcoef(averaged_statmaps[subjects[i]], averaged_statmaps[subjects[j]])[0, 1]
                        if args.fisher:
                            r = np.arctanh(r * (1 - eps))
                        R.append(r)
                R = pd.Series(R)
                row['r_%s' % rtype] = R.mean()
                row['r_%s_sem' % rtype] = R.sem()
            between_subjects.append(row)
        between_subjects = pd.DataFrame(between_subjects)
        between_subjects.to_csv('stability_between_subjects.csv', index=False)

    print('Between networks')
    between_networks = []
    for rtype in RTYPES:
        _between_networks = pd.DataFrame(index=ALL_REFERENCE, columns=ALL_REFERENCE)
        for i in range(len(ALL_REFERENCE)):
            network1 = ALL_REFERENCE[i]
            for subject in nii_paths[network1]:
                for j in range(i + 1, len(ALL_REFERENCE)):
                    network2 = ALL_REFERENCE[j]
                    a_paths = nii_paths[network1][subject]
                    b_paths = nii_paths[network2][subject]
                    for a_path, b_path in zip(a_paths, b_paths):
                        a = get_rtype(path2nii[a_path['path']], rtype=rtype)
                        b = get_rtype(path2nii[b_path['path']], rtype=rtype)
                        r = np.corrcoef(a, b)[0, 1]
                        if args.fisher:
                            r = np.arctanh(r * (1 - eps))
                        if not isinstance(_between_networks.loc[network1, network2], list):
                            _between_networks.loc[network1, network2] = []
                        _between_networks.loc[network1, network2].append(r)
                        if not isinstance(_between_networks.loc[network2, network1], list):
                            _between_networks.loc[network2, network1] = []
                        _between_networks.loc[network2, network1].append(r)
        between_networks_mean  = pd.DataFrame(index=ALL_REFERENCE, columns=ALL_REFERENCE)
        between_networks_sem = pd.DataFrame(index=ALL_REFERENCE, columns=ALL_REFERENCE)
        for i in range(len(ALL_REFERENCE)):
            network1 = ALL_REFERENCE[i]
            for j in range(i + 1, len(ALL_REFERENCE)):
                network2 = ALL_REFERENCE[j]
                r = pd.Series(_between_networks.loc[network1, network2])
                r_mean = r.mean()
                r_sem = r.sem()
                between_networks_mean.loc[network1, network2] = r_mean
                between_networks_mean.loc[network2, network1] = r_mean
                between_networks_sem.loc[network1, network2] = r_sem
                between_networks_sem.loc[network2, network1] = r_sem
        between_networks_sem = between_networks_sem.rename(lambda x: x + '_sem', axis=1)
        _between_networks = pd.concat([between_networks_mean, between_networks_sem], axis=1)
        _between_networks['network'] = _between_networks.index
        _between_networks['rtype'] = rtype
        _between_networks = _between_networks.reset_index(drop=True)
        between_networks.append(_between_networks)
    between_networks = pd.concat(between_networks)
    between_networks.to_csv('stability_between_networks.csv', index=False)

