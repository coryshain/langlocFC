import sys
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14

try:
    with open('data_path.txt', 'r') as f:
        base_path = f.read().strip()
except FileNotFoundError:
    sys.stderr.write('Data path not set. Run `python -m langlocfc.set_data_path` before running any other scripts.\n')
    sys.stderr.flush()
    exit()

PARCELLATE_PATH = os.path.join(base_path, 'derivatives')
REF_PATH = (f'{PARCELLATE_PATH}/{{parcellation_type}}/plots/{{condition}}performance/'
             f'{{atlas}}_sub1_ref_sim.csv')
EVAL_PATH = (f'{PARCELLATE_PATH}/{{parcellation_type}}/plots/{{condition}}performance/'
             f'{{atlas}}_sub1_eval_{{eval_type}}.csv')
STABILITY_DIR = f'{PARCELLATE_PATH}/stability_{{parcellation_type}}/'
PARCELLATION_TYPES = [
    'rest_v_taskrest',
    'rest_v_tasktask',
]
EVAL_TYPES = ['sim', 'contrast']
ATLAS_TYPES = {
    'LanA': ['LANA'],
    'FPN-A': ['FPN_A'],
    'DN-B': ['DN_B'],
    'AUD': ['AUD'],
}
EVALS = [
    'Lang_S-N', 'Lang_I-D', 'ToM_bel-pho', 'ToM_NV_ment-phys', 'SpatWM_H-E', 'Math_H-E', # 'Music_I-B',  # Only 2 people
                                                                                                         # did this,
                                                                                                         # excluded.
    'Vis_Faces-Objects', 'Vis_Bodies-Objects', 'Vis_Scenes-Objects'
]
EVAL_CLASSES = ['Language', 'Theory of Mind', 'Executive', 'Music', 'Vision']
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
    condition = parcellation_type[-4:] + '/'
    parcellation_name = parcellation_type[:-4]
    for atlas_type in ATLAS_TYPES:
        atlases = ATLAS_TYPES[atlas_type]

        # Evaluation atlases
        # x_delta = 0.8 / (len(atlases) * 2)
        x_delta = 0.8
        for eval_type in EVAL_TYPES:
            plot_x_base = np.arange(len(EVALS))
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().axhline(y=0, lw=1, c='k', alpha=1, zorder=-1)

            if eval_type == 'sim':
                if atlas_type == 'LanA':
                    ylim = (-0.15, 0.5)
                elif atlas_type == 'FPN-A':
                    ylim = (-0.2, 0.25)
                elif atlas_type == 'DN-B':
                    ylim = (-0.25, 0.25)
                elif atlas_type == 'AUD':
                    ylim = (-0.11, 0.12)
                else:
                    raise ValueError('Unknown atlas type: %s', atlas_type)
            else:  # eval_type == 'contrast'
                if atlas_type == 'LanA':
                    ylim = (-1, 4)
                elif atlas_type == 'FPN-A':
                    ylim = (-1.6, 2.5)
                elif atlas_type == 'DN-B':
                    ylim = (-2, 1.9)
                elif atlas_type == 'AUD':
                    ylim = (-0.8, 1.2)
                else:
                    raise ValueError('Unknown atlas type: %s', atlas_type)

            for a, atlas in enumerate(atlases):
                df = pd.read_csv(
                    EVAL_PATH.format(parcellation_type=parcellation_name, condition=condition, atlas=atlas, eval_type=eval_type)
                )
                ref = 'ref_%s' % atlas
                for k, key in enumerate(('FC',)):
                    _x = plot_x_base + (a * 2 + k) * x_delta
                    _df = df[df.label == key][EVALS]
                    if FISHER and eval_type == 'sim':
                        _df = np.arctanh(_df * (1 - EPS))
                    _y = _df.mean(axis=0)
                    _y_err = _df.sem(axis=0)
                    _c = [tuple(np.array(CLASS2COLOR[EVAL2CLASS[e]]) / 255) for e in EVALS]
                    linewidth = 1
                    if key == 'FC':
                        label = '%s (FC)' % REFERENCE2NAME[atlas]
                        color = _c
                        edgecolor = _c
                        if atlas == 'LANG':
                            linestyle = 'dotted'
                        else:
                            linestyle = '-'
                    else:
                        label = '%s (group)' % REFERENCE2NAME[atlas]
                        color = 'none'
                        edgecolor = _c
                        if atlas == 'LANG':
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
            tick_labels = [EVAL2NAME[e] for e in EVALS]
            tick_colors = [tuple(np.array(CLASS2COLOR[EVAL2CLASS[e]]) / 255) for e in EVALS]
            plt.xticks(plot_x_base + tick_shift, tick_labels, rotation=45, ha='right', rotation_mode='anchor')
            for xtick, c_ in zip(plt.gca().get_xticklabels(), tick_colors):
                xtick.set_color(c_)
            if eval_type == 'sim':
                ylabel = 'z(r)'
            else:
                ylabel = '$t$-value'
            if ylim is not None:
                plt.ylim(ylim)
            if not os.path.exists('plots'):
                os.makedirs('plots')
            plt.gcf().set_size_inches(4, 4)
            plt.tight_layout()
            plt.savefig(f'plots/performance_{parcellation_name}_{parcellation_type[-4:]}'
                        f'_{atlas_type}_{eval_type}_axis{SUFFIX}', dpi=DPI)
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().spines['left'].set_visible(False)
            plt.tight_layout()
            plt.savefig(f'plots/performance_{parcellation_name}_{parcellation_type[-4:]}'
                        f'_{atlas_type}_{eval_type}{SUFFIX}', dpi=DPI)
            plt.close('all')





