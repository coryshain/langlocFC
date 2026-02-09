import numpy as np
import pandas as pd
from scipy import stats

LENGTH2X = {
    1: 0.,
    2: 1.,
    3: 2.,
    4: 3.,
    5: 3.5,
    6: 4.,
    8: 4.33,
    10: 4.66,
    12: 5.
}

def length2x(x):
    x = np.array(x)
    f = np.vectorize(lambda x: LENGTH2X[int(x)])
    return f(x)

fROI = pd.read_csv('pdd_results_old_fROI.csv')
FC = pd.read_csv('pdd_results.csv')
ROI = pd.read_csv('pdd_results_old_ROI.csv')

clens = ['C1', 'C2', 'C3', 'C4', 'C6', 'C12']
jlens = ['J1', 'J4', 'J12']
nclens= ['N3', 'N4']

c_plot_basis = length2x([1, 2, 3, 4, 6, 12])
j_plot_basis = length2x([1, 4, 12])

parcel_types = [fROI, FC, ROI]

for i in range(len(parcel_types)):
    df = parcel_types[i]
    df = df.rename({'effect_size': 'EffectSize', 'pdd_parcel': 'ROI', 'cond': 'Effect', 'subject': 'Subject'}, axis=1)
    df['StimType'] = np.zeros_like(df.Effect)
    df.StimType[df.Effect.str.contains('jab')] = 'J'
    df.StimType[df.Effect.str.contains('nc')] = 'N'
    df.StimType[(~df.Effect.str.contains('nc')) & (~df.Effect.str.contains('jab'))] = 'C'
    df['nlength'] = df.Effect.str.extract('(\d+)').astype(int)
    df['Effect'] = df['StimType'] + df['nlength'].astype(str)
    df = df.pivot(columns='Effect', index=['Subject', 'ROI'], values='EffectSize').reset_index()

    vals = df[clens].values
    b = np.linalg.lstsq(np.stack([np.ones(len(c_plot_basis)), c_plot_basis], axis=1), vals.T)[0]
    NLenC = b[1]
    df['NLenC'] = NLenC

    vals = df[jlens].values
    b = np.linalg.lstsq(np.stack([np.ones(len(j_plot_basis)), j_plot_basis], axis=1), vals.T)[0]
    NLenJ = b[1]
    df['NLenJ'] = NLenJ

    df['NLenDiff'] = df['NLenC'] - df['NLenJ']

    parcel_types[i] = df


fROI, FC, ROI = parcel_types

regions = [
    'IFGorb',
    'IFGtri',
    'TP',
    'aSTS',
    'pSTS',
    'TP'
]



_fROI, _FC, _ROI = fROI[fROI.ROI.isin(regions)], FC[FC.ROI.isin(regions)], ROI[ROI.ROI.isin(regions)]
print('=' * 50)
print('  fROI vs FC, NLenC:      beta: %.3f,' % (_fROI.NLenC.mean() - _FC.NLenC.mean()),
      stats.ttest_ind(_fROI.NLenC, _FC.NLenC))
print()
print('  FC vs ROI, NLenC:       beta: %.3f,' % (_FC.NLenC.mean() - _ROI.NLenC.mean()),
      stats.ttest_ind(_FC.NLenC, _ROI.NLenC))
