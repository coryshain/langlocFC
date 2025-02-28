import os
import yaml
import numpy as np
import pandas as pd
from nilearn import image, masking
from matplotlib import pyplot as plt
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from langlocfc import resources
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Evaluate stability of S-N vs langlocFC''')
    argparser.add_argument('config_paths', nargs='+', help='Paths to by-subject config files')
    argparser.add_argument('-e', '--evaluation_id', default='main', help='ID of parcellation to use')
    argparser.add_argument('-p', '--parcellation_id', default='main', help='ID of parcellation to use')
    argparser.add_argument('-f', '--fisher', action='store_true', help='Fisher-transform correlations')
    argparser.add_argument('-o', '--output_dir', default=None, help='Prefix to use for parcellation output directory')
    args = argparser.parse_args()

    eps = 1e-3 

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

    SvN_sess_by_subject = {}
    SvN_by_subject = {}
    SvNeven_by_subject = {}
    SvNodd_by_subject = {}
    LANG_sub1_by_subject = {}
    LANA_sub1_by_subject = {}
    for path in args.config_paths:
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        subject = int(os.path.basename(cfg['output_dir']).split('_')[0])
    
        SvN_path = cfg['evaluate'][args.evaluation_id].get('evaluation_atlases', {}).get('Lang_S-N', None)
        if SvN_path:
            SvN_sess = SvN_path
            SvN_sess_ = None
        while not SvN_sess.endswith('SUBJECTS'):
            SvN_sess_ = SvN_sess
            SvN_sess = os.path.dirname(SvN_sess)
        SvN_sess = os.path.basename(SvN_sess_)
        if subject not in SvN_sess_by_subject:
            SvN_sess_by_subject[subject] = set()
        SvN_sess_by_subject[subject].add(SvN_sess)
        if subject not in SvN_by_subject:
            SvN_by_subject[subject] = set()
        SvN_by_subject[subject].add(SvN_path)

        SvNeven_path = cfg['evaluate'][args.evaluation_id].get('evaluation_atlases', {}).get('Lang_EVEN_S-N', None)
        SvNodd_path = cfg['evaluate'][args.evaluation_id].get('evaluation_atlases', {}).get('Lang_ODD_S-N', None)
        if SvNeven_path and SvNodd_path:
            if subject not in SvNeven_by_subject:
                SvNeven_by_subject[subject] = []
            SvNeven_by_subject[subject].append(SvNeven_path)
            if subject not in SvNodd_by_subject:
                SvNodd_by_subject[subject] = []
            SvNodd_by_subject[subject].append(SvNodd_path)
     
        if args.output_dir:
            LANG_sub1_path = os.path.join(args.output_dir, os.path.basename(cfg['output_dir']), 'parcellation', args.parcellation_id, 'LANG_sub1.nii.gz')
        else:
            LANG_sub1_path = os.path.join(cfg['output_dir'], 'parcellation', args.parcellation_id, 'LANG_sub1.nii.gz')
        if os.path.exists(LANG_sub1_path):
            if subject not in LANG_sub1_by_subject:
                LANG_sub1_by_subject[subject] = set()
            LANG_sub1_by_subject[subject].add(LANG_sub1_path)
    
        if args.output_dir:
            LANA_sub1_path = os.path.join(args.output_dir, os.path.basename(cfg['output_dir']), 'parcellation', args.parcellation_id, 'LANA_sub1.nii.gz')
        else:
            LANA_sub1_path = os.path.join(cfg['output_dir'], 'parcellation', args.parcellation_id, 'LANA_sub1.nii.gz')
        if os.path.exists(LANA_sub1_path):
            if subject not in LANA_sub1_by_subject:
                LANA_sub1_by_subject[subject] = set()
            LANA_sub1_by_subject[subject].add(LANA_sub1_path)

    # Task stability and FC stability

    subjs_SvN = set()
    subjs_LANG = set()
    subjs_LANA = set()

    r_SvN = dict(raw=[], masked12=[], masked6=[])
    r_LANG = dict(raw=[], masked12=[], masked6=[])
    r_LANA = dict(raw=[], masked12=[], masked6=[])

    SvN_statmaps = dict(raw={}, masked12={}, masked6={})
    LANG_statmaps = dict(raw={}, masked12={}, masked6={})
    LANA_statmaps = dict(raw={}, masked12={}, masked6={})
    for subject in SvN_by_subject:
        #print(subject)
        for setname, pathset, r, subjs, statmaps in zip(
            ('S-N', 'LANG', 'LANA'),
            (SvN_by_subject[subject], LANG_sub1_by_subject[subject], LANA_sub1_by_subject[subject]),
            (r_SvN, r_LANG, r_LANA),
            (subjs_SvN, subjs_LANG, subjs_LANA),
            (SvN_statmaps, LANG_statmaps, LANA_statmaps)
        ):
            pathset = list(pathset)
            if len(pathset) < 1:
                continue
            #print('  ', setname)
            subjs.add(subject)
            for i in range(len(pathset)):
                a = pathset[i]
                #print('    ', a)
                a = image.load_img(a)
                a = image.get_data(a)[mask]
                for rtype in statmaps:
                    if subject not in statmaps[rtype]:
                        statmaps[rtype][subject] = []
                    if rtype == 'raw':
                        _a = a
                    elif rtype == 'masked12':
                        _a = a[fROI12]
                    elif rtype == 'masked6':
                        _a = a[fROI6]
                    else:
                        _a = [a[x] for x in fROIs]
                    statmaps[rtype][subject].append(_a)
                    for j in range(i+1, len(pathset)):
                        b = pathset[j]
                        #print('      ', b)
                        b = image.get_data(image.load_img(b))[mask]
                        if rtype == 'raw':
                            _b = b
                        elif rtype == 'masked12':
                            _b = b[fROI12]
                        elif rtype == 'masked6':
                            _b = b[fROI6]
                        else:
                            _b = [b[x] for x in fROIs]
                        r_raw = np.corrcoef(_a, _b)[0, 1]
                        if args.fisher:
                            r_raw = np.arctanh(r_raw * (1 - eps))
                        r[rtype].append(r_raw)
                        #print('      ', r[-1])

    within_subjects = []
    between_subjects = []
    if not os.path.exists('stability_hist'):
        os.makedirs('stability_hist')
    for setname, r, subjs, statmaps in zip(
            ('S-N', 'LANG', 'LANA'),
            (r_SvN, r_LANG, r_LANA),
            (subjs_SvN, subjs_LANG, subjs_LANA),
            (SvN_statmaps, LANG_statmaps, LANA_statmaps)
    ):
        print('Compiling', setname)
        print('  Within subjects')
        for rtype in ('raw', 'masked12', 'masked6'):
            _r = pd.Series(r[rtype])
            row = dict(statname=setname, nsubj=len(subjs), ncorr=len(_r), rtype=rtype, r_mean=_r.mean(), r_sem=_r.sem())
            within_subjects.append(row)
        
            hist = plt.hist(_r, bins=20, color='red', orientation='horizontal')
            plt.savefig('stability_hist/stability_hist_%s_%s.png' % (setname, rtype))
            plt.close('all')
        
            print('  Between subjects')
            averaged_statmaps = {}
            for rtype in ('raw', 'masked12', 'masked6'):
                rs = []
                for subject in statmaps[rtype]:
                    rs.append(np.stack(statmaps[rtype][subject], axis=-1).mean(axis=-1))
                averaged_statmaps[rtype] = rs
            for rtype in ('raw', 'masked12', 'masked6'):
                _r = []
                for i in range(len(statmaps[rtype])):
                    for j in range(i+1, len(statmaps[rtype])):
                        _r.append(np.corrcoef(averaged_statmaps[rtype][i], averaged_statmaps[rtype][j])[0, 1])
                _r = pd.Series(_r)
    
                row = dict(statname=setname, nsubj=len(_r), rtype=rtype, r_mean=_r.mean(), r_sem=_r.sem())
                between_subjects.append(row)

    within_subjects = pd.DataFrame(within_subjects)
    within_subjects.to_csv('stability_within_subjects.csv', index=False)

    between_subjects = pd.DataFrame(between_subjects)
    between_subjects.to_csv('stability_between_subjects.csv', index=False)

    # LANA vs LANG

    LANA_v_LANG = []
    for rtype in LANA_statmaps.keys():
        for subject in LANA_statmaps[rtype].keys():
            for i in range(len(LANA_statmaps[rtype][subject])):
                a = LANA_statmaps[rtype][subject][i]
                b = LANG_statmaps[rtype][subject][j]
            if a is not None and b is not None:
                r = np.corrcoef(a, b)[0, 1]
            LANA_v_LANG
        LANG = r_

    # EVEN vs ODD

    r_LANG_v_LANA = dict(raw=[], masked12=[], masked6=[])
    for subject in SvNeven_by_subject:
        even_pathset = SvNeven_by_subject[subject] 
        odd_pathset = SvNodd_by_subject[subject] 
        assert len(even_pathset) == len(odd_pathset), 'Even and odd pathsets should be the same length, got %d and %d' % (len(even_pathset), len(odd_pathset))
        for 
 
