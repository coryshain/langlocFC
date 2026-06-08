import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

FONTSIZE = 18
LEGEND_FONTSIZE = 10

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = FONTSIZE

try:
    with open('data_path.txt', 'r') as f:
        base_path = f.read().strip()
except FileNotFoundError:
    sys.stderr.write('Data path not set. Run `python -m langlocfc.set_data_path` before running any other scripts.\n')
    sys.stderr.flush()
    exit()

PARCELLATE_PATH = os.path.join(base_path, 'derivatives')
REF_PATH = (f'{PARCELLATE_PATH}/{{parcellation_type}}/plots/performance/'
             f'{{atlas}}_sub1_ref_sim.csv')
CORR_PATH = f'{PARCELLATE_PATH}/stability_{{parcellation_type}}/between_networks_all_{{atlas}}.csv'
EVAL_PATH = (f'{PARCELLATE_PATH}/{{parcellation_type}}/plots/performance/'
             f'{{atlas}}_sub1_eval_{{eval_type}}.csv')
GRID_PATH = (f'{PARCELLATE_PATH}/{{parcellation_type}}/plots/grid/'
             f'{{atlas}}_sub1_{{eval_type}}_grid.csv')
STABILITY_DIR = f'{PARCELLATE_PATH}/stability_{{parcellation_type}}/'
PARCELLATION_TYPES = ['nolangloc', 'nonlinguistic', 'unresidualized', 'residualized']
EVAL_TYPES = ['sim', 'contrast']
ATLAS_TYPES = {
    'Language': ['LANG', 'LANA'],
    'ToM': ['DN_B'],
    'MD': ['FPN_A'],
    'Auditory': ['AUD'],
    'Visual': ['VIS_C', 'VIS_P'],
}
EVALS = [
    'Lang_S-N', 'Lang_I-D', 'ToM_bel-pho', 'ToM_NV_ment-phys', 'SpatWM_H-E', 'Math_H-E', 'Music_I-B',
    'Vis_Faces-Objects', 'Vis_Bodies-Objects', 'Vis_Scenes-Objects'
]
EVAL_CLASSES = ['Language', 'Theory of Mind', 'Executive', 'Numerical', 'Music', 'Vision']
EVAL2NAME = {
    'Lang_S-N': 'Language (reading)',
    'Lang_I-D': 'Language (listening)',
    'ToM_bel-pho': 'Theory of Mind (verbal)',
    'ToM_NV_ment-phys': 'Theory of Mind (non-verbal)',
    'SpatWM_H-E': 'Executive (working memory)',
    'Math_H-E': 'Numerical (math)',
    'Music_I-B': 'Music (intact vs. scrambled)',
    'Vis_Faces-Objects': 'Vision (faces)',
    'Vis_Bodies-Objects': 'Vision (bodies)',
    'Vis_Scenes-Objects': 'Vision (scenes)'
}
EVAL2CLASS = {
    'Lang_S-N': 'Language',
    'Lang_I-D': 'Language',
    'ToM_bel-pho': 'Theory of Mind',
    'ToM_NV_ment-phys': 'Theory of Mind',
    'SpatWM_H-E': 'Executive',
    'Math_H-E': 'Numerical',
    'Music_I-B': 'Music',
    'Vis_Faces-Objects': 'Vision',
    'Vis_Bodies-Objects': 'Vision',
    'Vis_Scenes-Objects': 'Vision'
}
CLASS2COLOR = {
    'Language': (135, 83, 141),
    'Theory of Mind': (132, 60, 47),
    'Executive': (69, 153, 67),
    'Numerical': (0, 170, 172),
    'Music': (135, 209, 209),
    'Vision': (248, 152, 78)
}
ALL_REFERENCE = [
    'LANG',
    'LANA',
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
]
REFERENCE2NAME = {
    'LANG': 'LANG',
    'LANA': 'LanA',
    'FPN_A': 'FPN-A',
    'FPN_B': 'FPN-B',
    'DN_A': 'DN-A',
    'DN_B': 'DN-B',
    'CG_OP': 'CG-OP',
    'SAL_PMN': 'SAL/PMN',
    'dATN_A': 'dATN-A',
    'dATN_B': 'dATN-B',
    'AUD': 'AUD',
    'PM_PPr': 'PM-PPr',
    'SMOT_A': 'SMOT-A',
    'SMOT_B': 'SMOT-B',
    'VIS_C': 'VIS-C',
    'VIS_P': 'VIS-P',
}
SUFFIX = '.pdf'
FISHER = True
EPS = 1e-3
DPI = 200
RUN_SIMPLIFIED = False

for parcellation_type in PARCELLATION_TYPES:
    # Stability
    if parcellation_type == 'nolangloc':
        ylim = (-0.3, 1.6)
        x_delta = 0.4
        plot_x_base = np.array((0, 2))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)

        within = pd.read_csv(
            os.path.join(STABILITY_DIR.format(parcellation_type=parcellation_type), 'within_subjects.csv')
        )
        between = pd.read_csv(
            os.path.join(STABILITY_DIR.format(parcellation_type=parcellation_type), 'between_subjects.csv')
        )

        source = []
        for a, mask_type in enumerate(('raw', 'masked6')):
            for k, setname in enumerate(('Lang_S-N', 'LANA')):
                _x = plot_x_base + (a * 2 + k) * x_delta
                col = 'r_%s' % mask_type
                semcol = '%s_sem' % col
                _within = within[within.setname == setname]
                _y_w_all = _within[col].values
                _y_w_all = _y_w_all[~np.isnan(_y_w_all)]
                _y_w = _within[col].mean()
                _y_w_err = _within[col].sem() * 1.96
                _between = between[between.setname == setname]
                subjects_df = _between.subjects.str.split('-', expand=True)
                subjects_df.columns = ['subject1', 'subject2']
                _between = pd.concat([_between, subjects_df], axis=1)
                subjects = set(_between['subject1'].unique().tolist()) | set(_between['subject2'].unique().tolist())
                _y_b_all = []
                for subject in subjects:
                    sel = (_between['subject1'] == subject) | (_between['subject2'] == subject)
                    vals = _between[sel][col].values
                    vals = vals[~np.isnan(vals)]
                    if len(vals):
                        _y_b_all.append(vals.mean())
                _y_b_all = pd.Series(_y_b_all)
                _y_b = _y_b_all.mean()
                _y_b_err = _y_b_all.sem() * 1.96
                _y_b_all = _y_b_all.values
                _y_all = [_y_w_all, _y_b_all]
                _y = np.stack([
                    _y_w, _y_b
                ])
                _y_err = np.stack([
                    _y_w_err, _y_b_err
                ])
                _df = pd.concat([
                    pd.Series(_y_w_all, name='within (%s, %s)' % (mask_type, setname)),
                    pd.Series(_y_b_all, name='between (%s, %s)' % (mask_type, setname))
                ], axis=1)
                n_sub = len(_y_w_all)
                source.append(_df.reset_index(drop=True))

                _c = 'black'
                if setname == 'Lang_S-N':
                    label = 'S-N'
                    color = 'none'
                    edgecolor = _c
                    if mask_type == 'raw':
                        label += ' (whole)'
                        linestyle = 'dotted'
                    else:
                        label += ' (masked)'
                        linestyle = '-'
                else:
                    label = 'LangFC'
                    color = _c
                    edgecolor = _c
                    if mask_type == 'raw':
                        label += ' (whole)'
                        linestyle = 'dotted'
                    else:
                        label += ' (masked)'
                        linestyle = '-'

                linewidth = 2

                plt.bar(
                    _x,
                    _y,
                    yerr=_y_err,
                    label=label,
                    width=x_delta,
                    capsize=0,
                    color=color,
                    edgecolor=edgecolor,
                    ecolor=edgecolor,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    error_kw=dict(linewidth=linewidth),
                    zorder=0
                )

                for i in range(len(_y)):
                    # if len(_y_all[i]) > n_sub:
                    #     _y_scatter = np.random.choice(_y_all[i], n_sub)
                    # else:
                    _y_scatter = _y_all[i]
                    plt.scatter(
                        np.random.random(len(_y_scatter)) * x_delta * 0.5 + _x[i] - x_delta * 0.5 / 2,
                        _y_scatter,
                        s=0.1,
                        alpha=0.1,
                        color=edgecolor,
                        zorder=1
                    )

        source = pd.concat(source, axis=1)

        tick_shift = x_delta * (2 - 0.5)
        plt.xticks(plot_x_base + tick_shift, ['Within', 'Between'], rotation=45, ha='right', rotation_mode='anchor')
        ylabel = 'z(r)'
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(ylim)
        plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=2, frameon=False, fontsize=18)
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.gcf().set_size_inches(6, 4)
        plt.tight_layout()
        print(f'plots/stability_within_between{SUFFIX}')
        plt.savefig(f'plots/stability_within_between{SUFFIX}', dpi=DPI)
        plt.close('all')
        source.to_csv(f'plots/stability_within_between.csv', index=False)

        if RUN_SIMPLIFIED:
            # Simplified plot
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)
            for a, mask_type in enumerate(('raw',)):
                for k, setname in enumerate(('LANA',)):
                    _x = plot_x_base + (a * 2 + k) * x_delta
                    col = 'r_%s' % mask_type
                    semcol = '%s_sem' % col
                    _within = within[within.setname == setname]
                    _between = between[between.setname == setname]
                    _y = np.concatenate([
                        _within[col].values, _between[col].values
                    ])
                    _y_err = np.concatenate([
                        _within[semcol].values, _between[semcol].values
                    ])

                    _c = 'black'
                    color = _c
                    edgecolor = _c
                    linestyle = '-'

                    linewidth = 2

                    plt.bar(
                        _x,
                        _y,
                        yerr=_y_err,
                        width=1,
                        capsize=0,
                        color=color,
                        edgecolor=edgecolor,
                        ecolor=edgecolor,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        error_kw=dict(linewidth=linewidth),
                        zorder=0
                    )
            tick_shift = 0
            plt.xticks(plot_x_base + tick_shift, ['Within', 'Between'], rotation=45, ha='right', rotation_mode='anchor')
            ylabel = 'Spatial corr'
            plt.ylabel(ylabel)
            if ylim is not None:
                plt.ylim(ylim)
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.gcf().set_size_inches(3, 4)
            plt.tight_layout()
            print(f'plots/stability_within_between_simplified{SUFFIX}')
            plt.savefig(f'plots/stability_within_between_simplified{SUFFIX}', dpi=DPI)
            plt.close('all')


    # Grid n voxels
    # Main plot
    if parcellation_type in ('nolangloc', 'nonlinguistic'):
        for atlas in ('LANG', 'LANA'):
            for eval_type in ('nvoxels_by_n_networks', 'eval_Lang_S-N_by_n_networks_sim'):
                plot_x_base = np.arange(10, 201, 10)
                cols = [str(x) for x in plot_x_base]
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)

                df = pd.read_csv(GRID_PATH.format(parcellation_type=parcellation_type, atlas=atlas, eval_type=eval_type))
                if 'label' in df:
                    sel = df.label == 'FC'
                    df = df[sel][cols]
                if FISHER and eval_type.startswith('eval'):
                    df = np.arctanh(df * (1 - EPS))
                source = df
                _y = df.mean(axis=0).values
                _y_err = df.sem(axis=0).values * 1.96
                _x = plot_x_base
                linewidth = 2
                label = '%s (FC)' % REFERENCE2NAME[atlas]
                color = (1, 0, 1)
                linestyle = '-'

                plt.plot(
                    _x,
                    _y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    zorder=0
                )
                plt.fill_between(
                    _x,
                    _y - _y_err,
                    _y + _y_err,
                    color=color,
                    alpha=0.3,
                    zorder=-1
                )

                cuts = (0.1, 0.2, 0.3, 0.4)
                _y_all = df.values
                for i in range(len(cuts)):
                    lq = np.nanquantile(_y_all, cuts[i], axis=0)
                    uq = np.nanquantile(_y_all, 1-cuts[i], axis=0)
                    plt.plot(
                        _x,
                        lq,
                        color=(0.5, 0.5, 0.5),
                        alpha=cuts[i],
                        linewidth=1
                    )
                    plt.plot(
                        _x,
                        uq,
                        color=(0.5, 0.5, 0.5),
                        alpha=cuts[i],
                        linewidth=1
                    )

                plt.xlabel('N Networks')
                if eval_type.startswith('nvoxels'):
                    ylabel = 'N Voxels'
                else:
                    ylabel = 'FC to S-N z(r)'
                plt.ylabel(ylabel)
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.gcf().set_size_inches(5, 3.75)
                plt.tight_layout()
                print(f'plots/performance_{parcellation_type}_{atlas}_{eval_type}_grid{SUFFIX}')
                plt.savefig(f'plots/performance_{parcellation_type}_{atlas}_{eval_type}_grid{SUFFIX}', dpi=DPI)
                plt.close('all')
                source.to_csv(f'plots/performance_{parcellation_type}_{atlas}_{eval_type}_grid.csv', index=False)

    # Simplified plot
    if RUN_SIMPLIFIED and parcellation_type in ('nolangloc',):
        for atlas in ('LANA',):
            for eval_type in ('eval_Lang_S-N_by_n_networks_sim',):
                plot_x_base = np.arange(10, 201, 10)
                cols = [str(x) for x in plot_x_base]
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)

                df = pd.read_csv(GRID_PATH.format(parcellation_type=parcellation_type, atlas=atlas, eval_type=eval_type))
                if 'label' in df:
                    sel = df.label == 'FC'
                    df = df[sel][cols]
                if FISHER and eval_type.startswith('eval'):
                    df = np.arctanh(df * (1 - EPS))
                _y = df.mean(axis=0).values
                _y_err = df.sem(axis=0).values
                _x = plot_x_base
                linewidth = 2
                label = '%s (FC)' % REFERENCE2NAME[atlas]
                color = (1, 0, 1)
                linestyle = '-'

                plt.plot(
                    _x,
                    _y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    zorder=0
                )

                plt.fill_between(
                    _x,
                    _y - _y_err,
                    _y + _y_err,
                    color=color,
                    alpha=0.3,
                    zorder=-1
                )

                plt.xlabel('N Networks')
                ylabel = 'LangFC vs task spatial corr'
                plt.ylabel(ylabel)
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.gcf().set_size_inches(5, 3.75)
                plt.tight_layout()
                print(f'plots/performance_{parcellation_type}_{atlas}_{eval_type}_grid_simplified{SUFFIX}')
                plt.savefig(f'plots/performance_{parcellation_type}_{atlas}_{eval_type}_grid_simplified{SUFFIX}',
                            dpi=DPI)
                plt.close('all')


    for atlas_type in ATLAS_TYPES:
        atlases = ATLAS_TYPES[atlas_type]

        # FC atlases

        plt.rcParams["font.size"] = 16
        if parcellation_type in ('nolangloc'):
            plot_x_base = np.arange(len(ALL_REFERENCE))
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)
            x_delta = 0.8 / len(atlases)

            if atlas_type == 'Language':
                # ylim = (-0.1, 2.5)
                ylim = None
            else:
                ylim = None

            source = []
            for a, atlas in enumerate(atlases):
                df = pd.read_csv(CORR_PATH.format(parcellation_type=parcellation_type, atlas=atlas))
                _df = df[df.rtype == 'raw'][ALL_REFERENCE]
                if FISHER:
                    _df = np.arctanh(_df * (1 - EPS))
                source.append(_df.reset_index(drop=True))
                _y_all = _df.values
                _y_all = [_y_all[:, i][~np.isnan(_y_all[:, i])] for i in range(_y_all.shape[1])]
                _y = _df.mean(axis=0).values
                _y_err = _df.sem(axis=0).values * 1.96
                _x = plot_x_base + a * x_delta
                if atlas == 'LANG':
                    _c = (0.5, 0.5, 0.5)
                else:
                    _c = (0.2, 0.2, 0.2)
                linewidth = 2
                label = '%s (FC)' % REFERENCE2NAME[atlas]
                color = _c
                edgecolor = _c
                linestyle = '-'

                plt.bar(
                    _x,
                    _y,
                    yerr=_y_err,
                    label=label,
                    width=x_delta,
                    capsize=0,
                    color=color,
                    edgecolor=edgecolor,
                    ecolor=edgecolor,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    error_kw=dict(linewidth=linewidth),
                    zorder=0
                )

                for i in range(len(_y)):
                    plt.scatter(
                        np.random.random(len(_y_all[i])) * x_delta * 0.5 + _x[i] - x_delta * 0.5 / 2,
                        _y_all[i],
                        s=0.1,
                        alpha=0.1,
                        color=edgecolor,
                        zorder=1
                    )

            source = pd.concat(source, axis=1)

            tick_shift = x_delta * (len(atlases) - 0.5) / 2
            plt.xticks(plot_x_base + tick_shift, [REFERENCE2NAME[x] for x in ALL_REFERENCE], rotation=45, ha='right', rotation_mode='anchor')
            ylabel = 'z(r)'
            plt.ylabel(ylabel)
            if ylim is not None:
                plt.ylim(ylim)
            plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=4, frameon=False, fontsize=14)
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.gcf().set_size_inches(6, 4.5)
            plt.tight_layout()
            plt.savefig(f'plots/performance_{parcellation_type}_{atlas_type}_networksim{SUFFIX}', dpi=DPI)
            plt.close('all')
            source.to_csv(f'plots/performance_{parcellation_type}_{atlas_type}_networksim.csv', index=False)

        # Reference atlases
        plot_x_base = np.arange(len(ALL_REFERENCE))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)
        x_delta = 0.8 / len(atlases)


        if parcellation_type in ('nolangloc'):
            if atlas_type == 'Language':
                # ylim = (-0.08, 0.6)
                ylim = (-0.1, 0.8)
            else:
                ylim = None

            source = []
            for a, atlas in enumerate(atlases):
                df = pd.read_csv(REF_PATH.format(parcellation_type=parcellation_type, atlas=atlas))[ALL_REFERENCE]
                if FISHER:
                    df = np.arctanh(df * (1 - EPS))

                label = '%s (FC)' % REFERENCE2NAME[atlas]
                _df = df.rename(columns={x: '%s to %s' % (label, x) for x in df.columns})

                _x = plot_x_base + a * x_delta
                source.append(_df.reset_index(drop=True))
                _y_all = _df.values
                _y_all = [_y_all[:, i][~np.isnan(_y_all[:, i])] for i in range(_y_all.shape[1])]
                _y = _df.mean(axis=0)
                _y_err = _df.sem(axis=0) * 1.96
                if atlas == 'LANG':
                    _c = (0.5, 0.5, 0.5)
                else:
                    _c = (0.2, 0.2, 0.2)
                linewidth = 2
                color = _c
                edgecolor = _c
                linestyle = '-'

                plt.bar(
                    _x,
                    _y,
                    yerr=_y_err,
                    label=label,
                    width=x_delta,
                    capsize=0,
                    color=color,
                    edgecolor=edgecolor,
                    ecolor=edgecolor,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    error_kw=dict(linewidth=linewidth),
                    zorder=0
                )

                for i in range(len(_y)):
                    plt.scatter(
                        np.random.random(len(_y_all[i])) * x_delta * 0.5 + _x[i] - x_delta * 0.5 / 2,
                        _y_all[i],
                        s=0.1,
                        alpha=0.1,
                        color=edgecolor,
                        zorder=1
                    )

            source = pd.concat(source, axis=1)

            tick_shift = x_delta * (len(atlases) - 0.5) / 2
            plt.xticks(plot_x_base + tick_shift, [REFERENCE2NAME[x] for x in ALL_REFERENCE], rotation=45, ha='right', rotation_mode='anchor')
            ylabel = 'z(r)'
            plt.ylabel(ylabel)
            if ylim is not None:
                plt.ylim(ylim)
            plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=4, frameon=False, fontsize=16)
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.gcf().set_size_inches(6, 4.5)
            plt.tight_layout()
            print(f'plots/performance_{parcellation_type}_{atlas_type}_refsim{SUFFIX}')
            plt.savefig(f'plots/performance_{parcellation_type}_{atlas_type}_refsim{SUFFIX}', dpi=DPI)
            plt.close('all')
            source.to_csv(f'plots/performance_{parcellation_type}_{atlas_type}_refsim.csv', index=False)

        plt.rcParams["font.size"] = FONTSIZE

        # Evaluation atlases
        # Full plots
        plt.rcParams["font.size"] = 22
        x_delta = 0.8 / (len(atlases) * 2)
        for eval_type in EVAL_TYPES:
            source = []

            plot_x_base = np.arange(len(EVALS))
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)

            if eval_type == 'sim':
                if 'residualized' in parcellation_type:
                    ylim = (-0.3, 0.7)
                else:
                    ylim = (-0.3, 0.7)
            else:
                if 'residualized' in parcellation_type:
                    ylim = (-3, 6)
                else:
                    ylim = (-3, 5)

            for a, atlas in enumerate(atlases):
                df = pd.read_csv(
                    EVAL_PATH.format(parcellation_type=parcellation_type, atlas=atlas, eval_type=eval_type)
                )
                ref = 'ref_%s' % atlas
                for k, key in enumerate((ref, 'FC')):
                    _x = plot_x_base + (a * 2 + k) * x_delta
                    _df = df[df.label == key][EVALS].rename(
                        columns={e: '%s (%s) to %s' % (atlas, key, e) for e in EVALS}
                    )
                    if FISHER and eval_type == 'sim':
                        _df = np.arctanh(_df * (1 - EPS))
                    source.append(_df.reset_index(drop=True))
                    _y_all = _df.values
                    _y_all = [_y_all[:, i][~np.isnan(_y_all[:, i])] for i in range(_y_all.shape[1])]
                    _y = _df.mean(axis=0)
                    _y_err = _df.sem(axis=0) * 1.96
                    _c = [tuple(np.array(CLASS2COLOR[EVAL2CLASS[e]]) / 255) for e in EVALS]
                    linewidth = 2
                    if key == 'FC':
                        label = '%s (FC)' % REFERENCE2NAME[atlas]
                        color = _c
                        edgecolor = _c
                        if len(atlases) > 1 and a == 0:
                            # linestyle = 'dotted'
                            linestyle = (0, (2, 1))
                        else:
                            linestyle = '-'
                    else:
                        label = '%s (group)' % REFERENCE2NAME[atlas]
                        color = 'none'
                        edgecolor = _c
                        if len(atlases) > 1 and a == 0:
                            # linestyle = 'dotted'
                            linestyle = (0, (2, 1))
                        else:
                            linestyle = '-'

                    plt.bar(
                        _x,
                        _y,
                        yerr=_y_err,
                        label=label,
                        width=x_delta,
                        capsize=0,
                        color=color,
                        edgecolor=edgecolor,
                        ecolor=edgecolor,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        error_kw=dict(linewidth=linewidth),
                        zorder=0
                    )
                    for i in range(len(_y)):
                        plt.scatter(
                            np.random.random(len(_y_all[i])) * x_delta * 0.5 + _x[i] - x_delta * 0.5 / 2,
                            _y_all[i],
                            s=0.1,
                            alpha=0.1,
                            color=edgecolor[i],
                            zorder=1
                        )
            source = pd.concat(source, axis=1)

            tick_shift = x_delta * (len(atlases) - 0.5)
            tick_colors = [tuple(np.array(CLASS2COLOR[EVAL2CLASS[e]]) / 255) for e in EVALS]
            plt.xticks(plot_x_base + tick_shift, [EVAL2NAME[eval] for eval in EVALS], rotation=60, ha='right', rotation_mode='anchor', fontsize=22)
            for xtick, c_ in zip(plt.gca().get_xticklabels(), tick_colors):
                xtick.set_color(c_)
            if eval_type == 'sim':
                ylabel = 'z(r)'
            else:
                ylabel = '$t$-value'
            plt.ylabel(ylabel)
            if ylim is not None:
                plt.ylim(ylim)
            plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=4, frameon=False, fontsize=LEGEND_FONTSIZE)
            legend = plt.gca().get_legend()
            for i in range(len(legend.legend_handles)):
                facecolor = legend.legend_handles[i].get_facecolor()
                if facecolor[-1] > 0:
                    legend.legend_handles[i].set_facecolor((0.2, 0.2, 0.2))
                legend.legend_handles[i].set_edgecolor((0.2, 0.2, 0.2))
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.gcf().set_size_inches(7, 8)
            plt.tight_layout()
            print(f'plots/performance_{parcellation_type}_{atlas_type}_{eval_type}{SUFFIX}')
            plt.savefig(f'plots/performance_{parcellation_type}_{atlas_type}_{eval_type}{SUFFIX}', dpi=DPI)
            plt.close('all')
            source.to_csv(f'plots/performance_{parcellation_type}_{atlas_type}_{eval_type}.csv', index=False)

        if RUN_SIMPLIFIED:
            # Simplified plots
            plt.rcParams["font.size"] = 22
            for eval_type in ('contrast',):
                if atlas_type == 'Language':
                    _atlases = ['LANA']
                    if eval_type == 'sim':
                        if 'residualized' in parcellation_type:
                            ylim = (-0.15, 0.45)
                        else:
                            ylim = (-0.15, 0.4)
                    else:
                        if 'residualized' in parcellation_type:
                            ylim = (-0.7, 3.9)
                        else:
                            ylim = (-0.7, 3.2)
                else:
                    _atlases = atlases
                    ylim = None

                x_delta = 0.8 / (len(_atlases) * 2)

                # Main plot
                plot_x_base = np.arange(len(EVALS))
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)
                for a, atlas in enumerate(_atlases):
                    df = pd.read_csv(
                        EVAL_PATH.format(parcellation_type=parcellation_type, atlas=atlas, eval_type=eval_type)
                    )
                    ref = 'ref_%s' % atlas
                    for k, key in enumerate((ref, 'FC')):
                        _x = plot_x_base + (a * 2 + k) * x_delta
                        _df = df[df.label == key][EVALS]
                        if FISHER and eval_type == 'sim':
                            _df = np.arctanh(_df * (1 - EPS))
                        _y = _df.mean(axis=0)
                        _y_err = _df.sem(axis=0)
                        _c = [tuple(np.array(CLASS2COLOR[EVAL2CLASS[e]]) / 255) for e in EVALS]
                        linewidth = 2
                        if key == 'FC':
                            label = 'iFC'
                            color = _c
                            edgecolor = _c
                            if len(_atlases) > 1 and a == 0:
                                linestyle = 'dotted'
                            else:
                                linestyle = '-'
                        else:
                            label = 'Atlas'
                            color = 'none'
                            edgecolor = _c
                            if len(_atlases) > 1 and a == 0:
                                linestyle = 'dotted'
                            else:
                                linestyle = '-'

                        plt.bar(
                            _x,
                            _y,
                            yerr=_y_err,
                            label=label,
                            width=x_delta,
                            capsize=0,
                            color=color,
                            edgecolor=edgecolor,
                            ecolor=edgecolor,
                            linestyle=linestyle,
                            linewidth=linewidth,
                            error_kw=dict(linewidth=linewidth),
                            zorder=0
                        )
                tick_shift = x_delta * (len(_atlases) - 0.5)
                tick_colors = [tuple(np.array(CLASS2COLOR[EVAL2CLASS[e]]) / 255) for e in EVALS]
                plt.xticks(plot_x_base + tick_shift, [EVAL2NAME[eval] for eval in EVALS], rotation=60, ha='right', rotation_mode='anchor', fontsize=22)
                for xtick, c_ in zip(plt.gca().get_xticklabels(), tick_colors):
                    xtick.set_color(c_)
                if eval_type == 'sim':
                    ylabel = 'Spatial corr'
                else:
                    ylabel = '$t$-value'
                plt.ylabel(ylabel)
                if ylim is not None:
                    plt.ylim(ylim)
                plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=4, frameon=False, fontsize=LEGEND_FONTSIZE*2)
                legend = plt.gca().get_legend()
                for i in range(len(legend.legend_handles)):
                    facecolor = legend.legend_handles[i].get_facecolor()
                    if facecolor[-1] > 0:
                        legend.legend_handles[i].set_facecolor((0.2, 0.2, 0.2))
                    legend.legend_handles[i].set_edgecolor((0.2, 0.2, 0.2))
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.gcf().set_size_inches(7, 8)
                plt.tight_layout()
                print(f'plots/performance_{parcellation_type}_{atlas_type}_{eval_type}_simplified{SUFFIX}')
                plt.savefig(f'plots/performance_{parcellation_type}_{atlas_type}_{eval_type}_simplified{SUFFIX}', dpi=DPI)
                plt.close('all')

                if atlas_type == 'Language':
                    # Language only
                    for N in (2, 4, 5, 6, 7, 10):
                        plot_x_base = np.arange(len(EVALS))
                        plt.gca().spines['top'].set_visible(False)
                        plt.gca().spines['right'].set_visible(False)
                        plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)
                        for a, atlas in enumerate(_atlases):
                            df = pd.read_csv(
                                EVAL_PATH.format(parcellation_type=parcellation_type, atlas=atlas, eval_type=eval_type)
                            )
                            for k, key in enumerate((None, 'FC')):
                                _x = plot_x_base + (a * 2 + k) * x_delta
                                _df = df[df.label == key][EVALS]
                                if FISHER and eval_type == 'sim':
                                    _df = np.arctanh(_df * (1 - EPS))
                                _y = _df.mean(axis=0)
                                _y_err = _df.sem(axis=0)
                                _c = [tuple(np.array(CLASS2COLOR[EVAL2CLASS[e]]) / 255) if i < N else (1, 1, 1) for i, e in enumerate(EVALS)]
                                linewidth = 2
                                label = 'iFC'
                                edgecolor = _c
                                linestyle = '-'

                                if key is None:
                                    color = 'none'
                                    plt.bar(
                                        _x,
                                        np.zeros_like(_x),
                                        yerr=np.zeros_like(_x),
                                        label='Atlas',
                                        width=x_delta,
                                        capsize=0,
                                        color=color,
                                        edgecolor='white',
                                        ecolor='white',
                                        linestyle=linestyle,
                                        linewidth=linewidth,
                                        error_kw=dict(linewidth=linewidth),
                                        zorder=-2
                                    )
                                else:
                                    color = _c
                                    __y = np.zeros_like(_y)
                                    __y[:N] = _y[:N]
                                    __y_err = np.zeros_like(_y_err)
                                    __y_err[:N] = _y_err[:N]
                                    plt.bar(
                                        _x,
                                        __y,
                                        yerr=__y_err,
                                        label=label,
                                        width=x_delta,
                                        capsize=0,
                                        color=color,
                                        edgecolor=edgecolor,
                                        ecolor=edgecolor,
                                        linestyle=linestyle,
                                        linewidth=linewidth,
                                        error_kw=dict(linewidth=linewidth),
                                        zorder=-2
                                    )
                        tick_shift = x_delta * (len(_atlases) - 0.5)
                        tick_colors = [tuple(np.array(CLASS2COLOR[EVAL2CLASS[e]]) / 255) for e in EVALS]
                        plt.xticks(plot_x_base + tick_shift, [EVAL2NAME[eval] for eval in EVALS], rotation=60, ha='right', rotation_mode='anchor', fontsize=22)
                        for i, (xtick, xtick_line, c_) in enumerate(zip(plt.gca().get_xticklabels(), plt.gca().get_xticklines(), tick_colors)):
                            if i < N:
                                xtick.set_color(c_)
                            else:
                                xtick.set_color('none')
                                # xtick_line.set_markersize(0)
                        if eval_type == 'sim':
                            ylabel = 'Spatial corr'
                        else:
                            ylabel = '$t$-value'
                        plt.ylabel(ylabel)
                        if ylim is not None:
                            plt.ylim(ylim)
                        plt.legend(bbox_to_anchor=(0.5, 1), loc='lower center', ncol=4, frameon=False, fontsize=LEGEND_FONTSIZE*2)
                        legend = plt.gca().get_legend()
                        for i in range(len(legend.legend_handles)):
                            facecolor = legend.legend_handles[i].get_facecolor()
                            label = legend.get_texts()[i]
                            if i < len(legend.legend_handles) - 1:
                                legend.legend_handles[i].set_facecolor((1, 1, 1))
                                legend.legend_handles[i].set_edgecolor((1, 1, 1))
                                label.set_color((1, 1, 1))
                            else:
                                if facecolor[-1] > 0:
                                    legend.legend_handles[i].set_facecolor((0.2, 0.2, 0.2))
                                legend.legend_handles[i].set_edgecolor((0.2, 0.2, 0.2))
                        if not os.path.exists('plots'):
                            os.makedirs('plots')
                        plt.gcf().set_size_inches(7, 8)
                        plt.tight_layout()
                        print(f'plots/performance_{parcellation_type}_{atlas_type}_{eval_type}_simplified_{N}{SUFFIX}')
                        plt.savefig(f'plots/performance_{parcellation_type}_{atlas_type}_{eval_type}_simplified_{N}{SUFFIX}', dpi=DPI)
                        plt.close('all')

            # Very simplified plots
            plt.rcParams["font.size"] = 22
            for eval_type in ('contrast',):
                plot_x_base = np.arange(len(EVALS))
                plt.gca().spines['top'].set_visible(False)
                plt.gca().spines['right'].set_visible(False)
                plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)

                if atlas_type == 'Language':
                    _atlases = ['LANA']
                    if eval_type == 'sim':
                        if 'residualized' in parcellation_type:
                            ylim = (-0.15, 0.45)
                        else:
                            ylim = (-0.15, 0.4)
                    else:
                        if 'residualized' in parcellation_type:
                            ylim = (-0.7, 3.9)
                        else:
                            ylim = (-0.7, 3.2)
                else:
                    _atlases = atlases
                    ylim = None

                x_delta = 0.8 / (len(_atlases))

                for a, atlas in enumerate(_atlases):
                    df = pd.read_csv(
                        EVAL_PATH.format(parcellation_type=parcellation_type, atlas=atlas, eval_type=eval_type)
                    )
                    for k, key in enumerate(('FC',)):
                        _x = plot_x_base + (a * 2 + k) * x_delta
                        _df = df[df.label == key][EVALS]
                        if FISHER and eval_type == 'sim':
                            _df = np.arctanh(_df * (1 - EPS))
                        _y = _df.mean(axis=0)
                        _y_err = _df.sem(axis=0)
                        _c = [tuple(np.array(CLASS2COLOR[EVAL2CLASS[e]]) / 255) for e in EVALS]
                        linewidth = 2
                        label = 'iFC'
                        color = _c
                        edgecolor = _c
                        if len(_atlases) > 1 and a == 0:
                            linestyle = 'dotted'
                        else:
                            linestyle = '-'

                        plt.bar(
                            _x,
                            _y,
                            yerr=_y_err,
                            label=label,
                            width=x_delta,
                            capsize=0,
                            color=color,
                            edgecolor=edgecolor,
                            ecolor=edgecolor,
                            linestyle=linestyle,
                            linewidth=linewidth,
                            error_kw=dict(linewidth=linewidth),
                            zorder=0
                        )
                tick_shift = 0
                tick_colors = [tuple(np.array(CLASS2COLOR[EVAL2CLASS[e]]) / 255) for e in EVALS]
                plt.xticks(plot_x_base + tick_shift, [EVAL2NAME[eval] for eval in EVALS], rotation=60, ha='right', rotation_mode='anchor', fontsize=22)
                for xtick, c_ in zip(plt.gca().get_xticklabels(), tick_colors):
                    xtick.set_color(c_)
                if eval_type == 'sim':
                    ylabel = 'Spatial corr'
                else:
                    ylabel = '$t$-value'
                plt.ylabel(ylabel)
                if ylim is not None:
                    plt.ylim(ylim)
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.gcf().set_size_inches(7, 8)
                plt.tight_layout()
                print(f'plots/performance_{parcellation_type}_{atlas_type}_{eval_type}_verysimplified{SUFFIX}')
                plt.savefig(f'plots/performance_{parcellation_type}_{atlas_type}_{eval_type}_verysimplified{SUFFIX}', dpi=DPI)
                plt.close('all')

            plt.rcParams["font.size"] = FONTSIZE
