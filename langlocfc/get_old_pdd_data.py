import os
import pandas as pd

data_dir = '../../data/fMRI_nlength/casto/main/'
data_paths = {
    'fROI': [
        'new_subjects_n15/PDD_parcels/mROI_NlengthEFFECT_langLOC/spm_ss_mROI_data.details.EffectSize.csv',
        'old_subjects_n25/PDD_parcels/mROI_NlengthEFFECT_langLOC/spm_ss_mROI_data.details.EffectSize.csv',
    ],
    'ROI': [
        'new_subjects_n15/PDD_parcels_anat/mROI_NlengthEFFECT_langLOC/spm_ss_mROI_data.details.EffectSize.csv',
        'old_subjects_n25/PDD_parcels_anat/mROI_NlengthEFFECT_langLOC/spm_ss_mROI_data.details.EffectSize.csv',
    ]
}

PDD_parcels = [
    'IFGorb',
    'IFGtri',
    'TP',
    'aSTS',
    'pSTS',
    'TPJ',
]

columns = ['subject', 'pdd_parcel', 'cond', 'effect_size']

for parcel_type in data_paths:
    df = []
    for path in data_paths[parcel_type]:
        df_ = pd.read_csv(os.path.join(data_dir, path))
        df_ = df_.rename({'EffectSize': 'effect_size', 'ROI': 'pdd_parcel', 'Effect': 'cond', 'Subject': 'subject'}, axis=1)
        df_ = df_[columns]
        df_ = df_[~df_.cond.isin({'A_12c-H_1c', 'I_jab12c-K_jab1c'})]
        df_.pdd_parcel = df_.pdd_parcel.apply(lambda x: PDD_parcels[x - 1])
        overall = df_.groupby(['subject', 'cond'], as_index=False).effect_size.mean()
        overall['pdd_parcel'] = 'Overall'
        df_ = pd.concat([overall, df_], axis=0)
        df.append(df_)
    df = pd.concat(df, axis=0)[columns]
    df = df.sort_values(['subject', 'cond']).reset_index(drop=True)
    df.to_csv(f'pdd_results_old_{parcel_type}.csv', index=True)

