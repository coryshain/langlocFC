import os
import shutil
import subprocess
import yaml
import numpy as np
from nilearn import image, masking
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen
import matplotlib.font_manager as fm
from matplotlib import cm, pyplot as plt
from PIL import Image
import argparse

######################################
#
#  GET BETTER FONT
#
######################################

roboto_url = 'https://github.com/google/fonts/blob/main/ofl/roboto/Roboto%5Bwdth%2Cwght%5D.ttf'
url = roboto_url + '?raw=true'
response = urlopen(url)
f = NamedTemporaryFile(suffix='.ttf')
f.write(response.read())
fm.fontManager.addfont(f.name)
prop = fm.FontProperties(fname=f.name)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()

######################################
#
#  CONSTANTS
#
######################################

SUFFIX2NAME = {
    '_atpgt0.1': 'p > 0.1',
    '_atpgt0.2': 'p > 0.2',
    '_atpgt0.3': 'p > 0.3',
    '_atpgt0.4': 'p > 0.4',
    '_atpgt0.5': 'p > 0.5',
    '_atpgt0.6': 'p > 0.6',
    '_atpgt0.7': 'p > 0.7',
    '_atpgt0.8': 'p > 0.8',
    '_atpgt0.9': 'p > 0.9',
}

COLORS = np.array([
    [255, 0, 0],  # Red
    [0, 0, 255],  # Blue
    [0, 255, 0],  # Green
    [0, 255, 255],  # Cyan
    [255, 0, 255],  # Magenta
    [255, 255, 0],  # Yellow
    [255, 128, 0],  # Orange
    [255, 0, 128],  # Pink
    [128, 255, 0],  # Lime
    [0, 128, 255],  # Aqua
    [128, 0, 255],  # Violet
    [0, 255, 128],  # Teal
    [255, 64, 0],  # Fire
    [255, 0, 64],  # Hot Pink
    [0, 64, 255],  # Ocean
], dtype=int)
RED = COLORS[0]
BLUE = COLORS[1]
GREEN = COLORS[2]
BASE_BRIGHTNESS = 0.  # Value from 0. (black) to 1. (full color)
cmap = cm.get_cmap('gist_rainbow')

SURFICE_SCRIPT = f'''
import sys
import os
import gl

CWD = os.path.normpath(os.path.join('..', '..', '..', os.getcwd()))

def get_path(path):
    if not os.path.isabs(path):
        path = os.path.join(CWD, os.path.normpath(path))
    path = os.path.normpath(path)

    return path

X = 400
Y = 300

plot_sets = [{{plot_set}}]
dark_atlas = False


gl.colorbarvisible(0)
gl.orientcubevisible(0)
gl.cameradistance(0.55)
gl.shadername('Default')
gl.shaderambientocclusion(0.)
if dark_atlas:
    gl.shaderadjust('Ambient', 0.15)
    gl.shaderadjust('Diffuse', 0.5)
    gl.shaderadjust('Specular', 0.35)
    gl.shaderadjust('SpecularRough', 1.)
    gl.shaderadjust('Edge', 1.)
    gl.shaderlightazimuthelevation(0, 0)
gl.overlayadditive(0)

for plot_set in plot_sets:
    for hemi in ('left', 'right'):
        for view in ('lateral', 'medial'):
            if hemi == 'left':
                gl.meshload('BrainMesh_ICBM152.lh.mz3')
                if view == 'lateral':
                    gl.azimuthelevation(-90, 0)
                else:
                    gl.azimuthelevation(90, 0)
            else:
                gl.meshload('BrainMesh_ICBM152.rh.mz3')
                if view == 'lateral':
                    gl.azimuthelevation(90, 0)
                else:
                    gl.azimuthelevation(-90, 0)
            output_path = None
            colors = None

            i = 0
            for atlas_name in plot_set:
                if output_path is None:
                    output_path = get_path(plot_set[atlas_name]['output_path'])
                if colors is None:
                    color = plot_set[atlas_name]['color']
                atlas_path = get_path(plot_set[atlas_name]['path'])
                min_act = plot_set[atlas_name]['min']
                max_act = plot_set[atlas_name]['max']

                if dark_atlas:
                    j_range = range(1, 2)
                else:
                    j_range = range(1, 2)
                for j in j_range:
                    overlay = gl.overlayload(atlas_path)
                    gl.overlaycolor(i + 1, *color)
                    if dark_atlas:
                        gl.overlayextreme(i + 1, 3)
                    else:
                        _opacity = int((j + 1) / 1 * 100)
                        gl.overlayopacity(i + 1, _opacity)
                    if min_act is not None and max_act is not None:
                        if dark_atlas:
                            _min_act, _max_act = min_act, max_act
                        else:
                            _min_act = min_act + (max_act - min_act) * (j + 1) / 1
                            _max_act = _min_act
                        gl.overlayminmax(i + 1, _min_act, _max_act)
                    i += 1

            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            plot_path = output_path % (hemi, view)
            gl.savebmpxy(plot_path, X, Y)
            gl.overlaycloseall()
exit()
'''


def _is_hemi(path):
    if not path.endswith('.png'):
        return False
    path = path[:-4]
    if not path.endswith('_lateral'):
        if not path.endswith('_medial'):
            return False
        path = path[:-7]
    else:
        path = path[:-8]
    if not path.endswith('_right'):
        if not path.endswith('_left'):
            return False
    return True


def sample_color():
    r, g, b = np.random.random(size=3)
    r = round(r * 255)
    g = round(g * 255)
    b = round(b * 255)

    return r, g, b


def expand_color(color, base_brightness=BASE_BRIGHTNESS):
    out = tuple([
        int(round(x * base_brightness)) for x in color
    ]) + tuple(color)

    return out


def plot_sample(config_path, n_samples=3, output_dir=None):
    resource_path = os.path.join('..', '..', 'parcellate', 'parcellate', 'resources')
    binary_path = os.path.join(resource_path, 'surfice', 'Surf_Ice', 'surfice.exe')
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if output_dir is not None:
        output_dir = os.path.join(output_dir, os.path.basename(cfg['output_dir']))
    else:
        output_dir = cfg['output_dir']
    sample_path = os.path.join(output_dir, 'sample', 'main', 'sample.nii.gz')
    sample = image.load_img(sample_path)
    sample_data = image.get_data(sample)
    mask = image.get_data(masking.compute_brain_mask(sample, connected=False, opening=False, mask_type='gm')) > 0.5
    sample_data = sample_data[mask]
    K = sample_data.max() + 1

    with TemporaryDirectory() as tmp_dir_path:
        for s in range(n_samples):
            parcellation_dir = os.path.join(tmp_dir_path, 'plots', 'atlas')
            if not os.path.exists(parcellation_dir):
                os.makedirs(parcellation_dir)
            output_path = os.path.join(parcellation_dir, f'sample_{s}_atlas_%s_%s.png')
            plot_set = {}
            for k in range(K):
                network = sample_data[..., s] == k
                _network = np.zeros(mask.shape)
                _network[mask] = network
                nii = image.new_img_like(sample, _network)

                nii_path = os.path.join(parcellation_dir, 'sample_%d_network_%d.nii.gz' % (s + 1, k + 1))

                nii.to_filename(nii_path)

                color = sample_color()

                plot_set['sample_%d' % (k + 1)] = {
                    'color': expand_color(color),
                    'max': 0.5,
                    'min': 0.5,
                    'name': 'sample_%d' % (k + 1),
                    'output_path': output_path,
                    'path': nii_path
                }

            script = SURFICE_SCRIPT.format(
                plot_set=plot_set
            )

            tmp_path = os.path.join(tmp_dir_path, 'PARCELLATE_SURFICE_SCRIPT_TMP.py')
            with open(tmp_path, 'w') as f:
                f.write(script)

            print('  Generating subplots...')

            subprocess.call([binary_path, '-S', tmp_path])

            print('  Stitching plots...')
            img_prefixes = set()
            for img in [x for x in os.listdir(parcellation_dir) if _is_hemi(x)]:
                img_prefix = '_'.join(img.split('_')[:-2])
                img_prefix = os.path.join(parcellation_dir, img_prefix)
                img_prefixes.add(img_prefix)
            for img_prefix in img_prefixes:
                imgs = []
                img_paths = []
                for hemi in ('left', 'right'):
                    if hemi == 'left':
                        views = ('lateral', 'medial')
                    else:
                        views = ('medial', 'lateral')
                    for view in views:
                        img_path = img_prefix + '_%s_%s.png' % (hemi, view)
                        imgs.append(Image.open(img_path))
                        img_paths.append(img_path)
                widths, heights = zip(*(i.size for i in imgs))
                total_width = sum(widths)
                max_height = max(heights)
                new_im = Image.new('RGB', (total_width, max_height))
                x_offset = 0
                for im in imgs:
                    new_im.paste(im, (x_offset, 0))
                    x_offset += im.size[0]
                new_im.save('%s.png' % img_prefix)
                for img_path in img_paths:
                    os.remove(img_path)

            dest_dir = os.path.join('plots', 'atlas')
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            for x in filter(lambda x: x.endswith('.png'), os.listdir(parcellation_dir)):
                path = os.path.join(parcellation_dir, x)
                subprocess.call(['cp', path, dest_dir])

            # Reset the cache
            shutil.rmtree(tmp_dir_path)
            os.makedirs(tmp_dir_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Plot parcellation sample''')
    argparser.add_argument('config_path', help='Path to config file')
    argparser.add_argument('-n', '--n_samples', default=3, type=int, help='Number of samples to plot')
    argparser.add_argument('-o', '--output_dir', default=None, help='Prefix to use for parcellation output directory')
    args = argparser.parse_args()

    plot_sample(args.config_path, n_samples=args.n_samples, output_dir=args.output_dir)