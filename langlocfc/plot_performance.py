import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

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
CORR_PATH = f'{PARCELLATE_PATH}/stability_{{parcellation_type}}/between_networks.csv'
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
SUFFIX = '.png'
FISHER = True
EPS = 1e-3
DPI = 200

for parcellation_type in PARCELLATION_TYPES:
    # Stability
    if parcellation_type == 'nolangloc':
        ylim = None
        x_delta = 0.4
        plot_x_base = np.array((0, 2))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)

        within = pd.read_csv(
            os.path.join(STABILITY_DIR.format(parcellation_type=parcellation_type), 'within_subjects_summary.csv')
        )
        between = pd.read_csv(
            os.path.join(STABILITY_DIR.format(parcellation_type=parcellation_type), 'between_subjects.csv')
        )
        for a, mask_type in enumerate(('raw', 'masked6')):
            for k, setname in enumerate(('Lang_S-N', 'LANA')):
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
        plt.savefig(f'plots/stability_within_between{SUFFIX}', dpi=DPI)
        plt.close('all')

    # Grid n voxels
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
                if eval_type.startswith('nvoxels'):
                    ylabel = 'N Voxels'
                else:
                    ylabel = 'FC to S-N z(r)'
                plt.ylabel(ylabel)
                if not os.path.exists('plots'):
                    os.makedirs('plots')
                plt.gcf().set_size_inches(5, 3.75)
                plt.tight_layout()
                plt.savefig(f'plots/performance_{parcellation_type}_{atlas}_{eval_type}_grid{SUFFIX}', dpi=DPI)
                plt.close('all')

    for atlas_type in ATLAS_TYPES:
        atlases = ATLAS_TYPES[atlas_type]

        # FC atlases

        plt.rcParams["font.size"] = 16
        if parcellation_type in ('nolangloc', 'nonlinguistic'):
            plot_x_base = np.arange(len(ALL_REFERENCE))
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)
            x_delta = 0.8 / len(atlases)

            if atlas_type == 'Language':
                ylim = (-0.1, 2.5)
            else:
                ylim = None

            df = pd.read_csv(CORR_PATH.format(parcellation_type=parcellation_type))
            for a, atlas in enumerate(atlases):
                sel = (df.network == atlas) & (df.rtype == 'raw')
                _df = df[sel]
                assert _df.shape[0] == 1, 'Multiple rows for %s' % atlas
                _df = _df.iloc[0]
                _y = _df[ALL_REFERENCE].values
                _y_err = _df[[x + '_sem' for x in ALL_REFERENCE]].values
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

        # Reference atlases
        plot_x_base = np.arange(len(ALL_REFERENCE))
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)
        x_delta = 0.8 / len(atlases)

        if atlas_type == 'Language':
            ylim = (-0.08, 0.6)
        else:
            ylim = None

        for a, atlas in enumerate(atlases):
            df = pd.read_csv(REF_PATH.format(parcellation_type=parcellation_type, atlas=atlas))[ALL_REFERENCE]
            if FISHER:
                df = np.arctanh(df * (1 - EPS))

            _x = plot_x_base + a * x_delta
            _df = df
            _y = _df.mean(axis=0)
            _y_err = _df.sem(axis=0)
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
        plt.savefig(f'plots/performance_{parcellation_type}_{atlas_type}_refsim{SUFFIX}', dpi=DPI)
        plt.close('all')

        plt.rcParams["font.size"] = FONTSIZE


        # Evaluation atlases
        plt.rcParams["font.size"] = 22
        x_delta = 0.8 / (len(atlases) * 2)
        for eval_type in EVAL_TYPES:
            plot_x_base = np.arange(len(EVALS))
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)

            if atlas_type == 'Language':
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
                ylim = None

            for a, atlas in enumerate(atlases):
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
                        label = '%s (FC)' % REFERENCE2NAME[atlas]
                        color = _c
                        edgecolor = _c
                        if len(atlases) > 1 and a == 0:
                            linestyle = 'dotted'
                        else:
                            linestyle = '-'
                    else:
                        label = '%s (group)' % REFERENCE2NAME[atlas]
                        color = 'none'
                        edgecolor = _c
                        if len(atlases) > 1 and a == 0:
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
            plt.savefig(f'plots/performance_{parcellation_type}_{atlas_type}_{eval_type}{SUFFIX}', dpi=DPI)
            plt.close('all')

        plt.rcParams["font.size"] = FONTSIZE
