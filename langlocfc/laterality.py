import os
import yaml
import numpy as np
import pandas as pd
import nibabel
from nilearn import image, masking

def LI(left, right):
    return (left - right) / (left + right)

try:
    with open('data_path.txt', 'r') as f:
        base_path = f.read().strip()
except FileNotFoundError:
    sys.stderr.write('Data path not set. Run `python -m langlocfc.set_data_path` before running any other scripts.\n')
    sys.stderr.flush()
    exit()
RESULTS_DIR = os.path.join(base_path, 'derivatives', 'laterality')

out = []
nolangloc_dir = os.path.join(base_path, 'derivatives', 'nolangloc')
mask_L = None
mask_R = None
for x in os.listdir(nolangloc_dir):
    cfg_path = os.path.join(nolangloc_dir, x, 'config.yml')
    if not os.path.exists(cfg_path):
        continue
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    session = os.path.basename(cfg['output_dir'])
    print(session)
    subject = session.split('_')[0]
    out_ = dict(session=session, subject=subject)
    parcellation_dir = os.path.join(nolangloc_dir, x, 'parcellation', 'main')
    SvN_path = os.path.join(parcellation_dir, 'eval_Lang_S-N.nii.gz')
    if not os.path.exists(SvN_path):
        SvN_path = None
    langFC_img = image.load_img(os.path.join(parcellation_dir, 'LANA_sub1.nii.gz'))
    if mask_L is None:
        mask = image.get_data(masking.compute_brain_mask(langFC_img, connected=False, opening=False, mask_type='gm')) > 0.5
        xmid = mask.shape[0] // 2
        i, j, k = np.meshgrid(np.arange(mask.shape[0]),
                          np.arange(mask.shape[1]),
                          np.arange(mask.shape[2]),
                          indexing='ij')
        i, j, k = image.coord_transform(i, j, k, langFC_img.affine)
        hemi_L = i < 0
        hemi_R = i > 0
        mask_L = mask & hemi_L
        mask_R = mask & hemi_R
    if SvN_path:
        SvN_img = image.load_img(SvN_path)
        SvN = image.get_data(SvN_img)
        SvN_L = (SvN[mask_L] >= 5).sum()
        SvN_R = (SvN[mask_R] >= 5).sum()
        out_['SvN_L'] = SvN_L
        out_['SvN_R'] = SvN_R
        out_['SvN_LI'] = LI(SvN_L, SvN_R)
    langFC = image.get_data(langFC_img)
    langFC_L = (langFC[mask_L] > 0.5).sum()
    langFC_R = (langFC[mask_R] > 0.5).sum()
    out_['langFC_L'] = langFC_L
    out_['langFC_R'] = langFC_R
    out_['langFC_LI'] = LI(langFC_L, langFC_R)
    out.append(out_)

out = pd.DataFrame(out)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
out.to_csv(os.path.join(RESULTS_DIR, 'laterality.csv'), index=False)

cols = [x for x in out.columns if (x.endswith('L') or x.endswith('R') or x.endswith('LI'))]
gb = out.groupby('subject')[cols].mean()
corr = pd.concat([gb.SvN_LI, gb.langFC_LI], axis=1).corr().iloc[0, 1]
print(corr)

out_summary = [dict(
    SvN_LI_mean=gb.SvN_LI.mean(),
    SvN_LI_sem=gb.SvN_LI.sem(),
    langFC_LI_mean=gb.langFC_LI.mean(),
    langFC_LI_sem=gb.langFC_LI.sem(),
    SvN_langFC_LI_corr=corr
)]
out_summary = pd.DataFrame(out_summary)
out_summary.to_csv(os.path.join(RESULTS_DIR, 'laterality_summary.csv'), index=False)
print(out_summary)

