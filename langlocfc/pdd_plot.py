import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

df = pd.read_csv('pdd_results.csv')
df = df.rename({'effect_size': 'EffectSize', 'pdd_parcel': 'ROI', 'cond': 'Effect', 'subject': 'Subject'}, axis=1)
df['StimType'] = np.zeros_like(df.Effect)
df.StimType[df.Effect.str.contains('jab')] = 'J'
df.StimType[df.Effect.str.contains('nc')] = 'N'
df.StimType[(~df.Effect.str.contains('nc')) & (~df.Effect.str.contains('jab'))] = 'C'
df['nlength'] = df.Effect.str.extract('(\d+)').astype(int)

ROIs = [x for x in df.ROI.unique().tolist() if x != 'Overall']
lengths = [1, 2, 3, 4, 6, 12]
ylims = {'': (-0.68, 3.5)}
plot_basis = length2x(lengths)
plot_dir = 'pdd_plots'

xtick_pos = plot_basis
xtick_labels = [str(x) for x in lengths]

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

for ROI in ROIs:
    # Plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().tick_params(labelleft='on', labelbottom='on')
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().axhline(y=0, lw=1, c='gray', alpha=1)

    _df = df[df.StimType == 'C']
    clens = [1, 2, 3, 4, 6, 12]
    means = []
    errs = []
    D_C = []
    subjects = None
    for i, clen in enumerate(clens):
        d = _df[(_df.nlength == clen) & (_df.ROI == ROI)]
        if len(d.values):
            d = d.sort_values('Subject')
            if subjects is None:
                subjects = d.Subject.values
            d = d.EffectSize
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            m = d.mean()
            means.append(m)
            sem = d.sem()
            errs.append(sem)
            D_C.append(d.values)
    D_C = np.stack(D_C, axis=1)

    b = np.linalg.lstsq(np.stack([np.ones_like(means), plot_basis], axis=1), D_C.T)[0]
    NLenC = b[1]

    xline = np.linspace(plot_basis.min(), plot_basis.max(), 500)
    X = np.stack([np.ones_like(xline), xline], axis=1)
    yline = np.dot(X, b).mean(axis=-1)

    plt.errorbar(
        plot_basis,
        means,
        yerr=errs,
        fmt='ro',
        linestyle='none',
        ecolor='red',
        lw=2,
        capsize=0,
        label='normal'
    )
    plt.plot(
        xline,
        yline,
        linestyle='dashed',
        color='red',
    )

    _df = df[df.StimType == 'J']
    clens = [1, 4, 12]
    means = []
    errs = []
    D_J = []
    for i, clen in enumerate(clens):
        d = _df[(_df.nlength == clen) & (_df.ROI == ROI)]
        d = d.sort_values('Subject')
        d = d.EffectSize
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        m = d.mean()
        means.append(m)
        sem = d.sem()
        errs.append(sem)
        D_J.append(d.values)
    D_J = np.stack(D_J, axis=1)

    b = np.linalg.lstsq(np.stack([np.ones_like(means), length2x(clens)], axis=1), D_J.T)[0]
    NLenJ = b[1]
    xline = np.linspace(plot_basis.min(), plot_basis.max(), 500)
    X = np.stack([np.ones_like(xline), xline], axis=1)
    yline = np.dot(X, b).mean(axis=-1)

    plt.errorbar(
        [0, 3, 5],
        means,
        yerr=errs,
        fmt='bs',
        linestyle='none',
        ecolor='blue',
        lw=2,
        capsize=0,
        label='normal'
    )
    plt.plot(
        xline,
        yline,
        linestyle='dashed',
        color='blue'
    )

    _df = df[df.StimType == 'N']
    clens = [3, 4]
    means = []
    errs = []
    D_N = []
    for i, clen in enumerate(clens):
        d = _df[(_df.nlength == clen) & (_df.ROI == ROI)]
        d = d.sort_values('Subject')
        d = d.EffectSize
        d = d[np.isfinite(d)]
        if not len(d):
            continue
        m = d.mean()
        means.append(m)
        sem = d.sem()
        errs.append(sem)
        D_N.append(d.values)
    D_N = np.stack(D_N, axis=1)

    b = np.linalg.lstsq(np.stack([np.ones_like(means), length2x(clens)], axis=1), D_N.T)[0]
    NLenN = b[1]

    xline = np.linspace(plot_basis.min(), plot_basis.max(), 500)
    X = np.stack([np.ones_like(xline), xline], axis=1)
    yline = np.dot(X, b).mean(axis=-1)

    plt.errorbar(
        [2, 3],
        means,
        yerr=errs,
        fmt='mx',
        linestyle='none',
        ecolor='m',
        lw=2,
        capsize=0,
        label='normal'
    )
    plt.plot(
        xline,
        yline,
        linestyle='dashed',
        color='m'
    )

    plt.subplots_adjust(left=0.3)
    plt.xlim(plot_basis.min() - 0.2, plot_basis.max() + 0.2)

    plt.xticks(xtick_pos, labels=xtick_labels)
    plt.gcf().set_size_inches((2, 3))

    for ylim_key in ylims:
        ylim = ylims[ylim_key]
        plt.ylim(ylim)
        plt.savefig(os.path.join(plot_dir, 'PDD_nlength2_%s_plot%s.png' % (ROI, ylim_key)),
                    dpi=300)
    plt.close('all')



