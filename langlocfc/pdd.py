import os
import numpy as np
import pandas as pd
from pymatreader import read_mat
import h5py
from scipy import io
from nilearn import image

conditions = ['H_1c', 'G_2c', 'E_3c', 'C_4c', 'B_6c', 'A_12c', 'F_3nc', 'D_4nc', 'K_jab1c', 'J_jab4c', 'I_jab12c']

subjects = []
with open('nlength2_subjects.csv', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            subjects.append(line)

pdd_parcels = {}
for parcel in os.listdir('pdd_parcels'):
    parcel_name = parcel.replace('.nii', '')
    parcel = image.smooth_img(os.path.join('pdd_parcels', parcel), None)
    parcel = image.get_data(parcel)
    pdd_parcels[parcel_name] = parcel

out = []

for subject in subjects:
    subject_dir = os.path.join('/', 'nese', 'mit', 'group', 'evlab', 'u', 'Shared', 'SUBJECTS', subject)
    parcellation_dir = os.path.join('..', '..', 'results', 'fMRI_parcellation', 'nolangloc', subject, 'parcellation', 'main')
    firstlevel_dir = os.path.join(subject_dir, 'firstlevel_Nlength_con2')
    spm_path = os.path.join(firstlevel_dir, 'SPM.mat')
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
    except RuntimeError:
        # Bad SPM file, skip
        print('Bad SPM file for subject %s' % subject)
        continue

    for cond in conditions:
        contrast = image.get_data(image.smooth_img(os.path.join(firstlevel_dir, 'con_%04d.nii' % name2ix[cond]), None))
        network = image.get_data(image.smooth_img(os.path.join(parcellation_dir, 'LANG_sub1.nii.gz'), None))
        
        pdd_parcel_name = 'Overall'
        effect_size = (contrast * network).sum() / network.sum()
        out.append(dict(
            subject=subject,
            pdd_parcel=pdd_parcel_name,
            cond=cond,
            effect_size=effect_size
        ))

        for pdd_parcel_name in pdd_parcels:
            pdd_parcel = pdd_parcels[pdd_parcel_name]
            # mask = network * pdd_parcel
            mask = network[pdd_parcel]
            thresh = np.quantile(mask, 0.9)
            mask = mask > thresh
            effect_size = (contrast * mask).sum() / mask.sum()

            out.append(dict(
                subject=subject,
                pdd_parcel=pdd_parcel_name,
                cond=cond,
                effect_size=effect_size
            ))

out = pd.DataFrame(out)
out.to_csv('pdd_results.csv')
