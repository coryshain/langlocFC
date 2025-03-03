import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

RUN_TYPES = ('nolangloc', 'nonlinguistic')
DEV_TYPES = (
    'DevNobpTimecourse',
    'DevBpTimecourse',
    'DevConnRegions',
    'DevConnRegionsBin',
    'DevConnDownsample',
    'DevConnDownsampleBin'
)
MODEL_TYPES = ('LANG_sub1', 'LANA_sub1')
METRIC_TYPES = ('sim', 'contrast')
results_dir = '../../results/fMRI_parcellation'

if __name__ == '__main__':
    plots = {}
    w = h = None
    for model_type in MODEL_TYPES:
        plots[model_type] = {}
        for run_type in RUN_TYPES:
            plots[model_type][run_type] = {}
            for dev_type in DEV_TYPES:
                _results_dir = os.path.join(results_dir, '%s%s' % (run_type, dev_type), 'plots', 'performance')
                for metric_type in METRIC_TYPES:
                    _plot_path = os.path.join(_results_dir, '%s_eval_%s.png' % (model_type, metric_type))
                    if metric_type not in plots[model_type][run_type]:
                        plots[model_type][run_type][metric_type] = {}
                    plots[model_type][run_type][metric_type][dev_type] = Image.open(_plot_path)
                    if w is None:
                        w, h = plots[model_type][run_type][metric_type][dev_type].size

    pad = 150
    titlepad = 300
    W = w * 8 + pad * 6
    H = h * 6 + pad * 4 + titlepad
    ratio = 2000 / W
    _W, _H = int(W * ratio), int(H * ratio)

    font1 = ImageFont.truetype("Arial", w // 10)
    font2 = ImageFont.truetype("Arial", w // 15)
    font3 = ImageFont.truetype("Arial", w // 20)
    canvas = Image.new('RGB', (W, H), color=(255, 255, 255))
    for i, model_type in enumerate(MODEL_TYPES):
        for j, run_type in enumerate(RUN_TYPES):
            for k, metric_type in enumerate(METRIC_TYPES):
                for l, dev_type in enumerate(DEV_TYPES):
                    if i == 0 and k == 0:
                        # dev_type label
                        txt = Image.new('RGB', (h, pad), color=(255, 255, 255))
                        draw = ImageDraw.Draw(txt)
                        _, _, _w, _h = draw.textbbox((0, 0), dev_type, font=font3)
                        draw.text(
                            ((h - _w) / 2, (pad - _h) / 2),
                            dev_type,
                            (0, 0, 0),
                            font=font3
                        )
                        txt = txt.rotate(90, expand=1)
                        canvas.paste(txt, (pad, l * h + pad + titlepad))
                    if l == 0:
                        # metric_type label
                        txt = Image.new('RGB', (w, pad), color=(255, 255, 255))
                        draw = ImageDraw.Draw(txt)
                        _, _, _w, _h = draw.textbbox((0, 0), metric_type, font=font3)
                        draw.text(
                            ((w - _w) / 2, (pad - _h) / 2),
                            metric_type,
                            (0, 0, 0),
                            font=font3
                        )
                        canvas.paste(txt, ((i * 4 + j * 2 + k) * w + (i * 3 + j + 1) * pad, pad + titlepad))

                        # run_type label
                        if k == 0:
                            txt = Image.new('RGB', (w * 2, pad), color=(255, 255, 255))
                            draw = ImageDraw.Draw(txt)
                            _, _, _w, _h = draw.textbbox((0, 0), run_type, font=font2)
                            draw.text(
                                ((w * 2 - _w) / 2, (pad - _h) / 2),
                                run_type,
                                (0, 0, 0),
                                font=font2
                            )
                            canvas.paste(txt, ((i * 4 + j * 2 + k) * w + (i * 3 + j + 1) * pad, titlepad))

                        # model_type label
                        if k == 0 and j == 0:
                            txt = Image.new('RGB', (w * 4, titlepad), color=(255, 255, 255))
                            draw = ImageDraw.Draw(txt)
                            _, _, _w, _h = draw.textbbox((0, 0), model_type, font=font1)
                            draw.text(
                                ((w * 4 - _w) / 2, (pad - _h) / 2),
                                model_type,
                                (0, 0, 0),
                                font=font1
                            )
                            canvas.paste(txt, ((i * 4 + j * 2 + k) * w + (i * 3 + j + 1) * pad, 0))

                    x = (i * 4 + j * 2 + k) * w + (i * 3 + j + 1) * pad
                    y = l * h + pad * 2 + titlepad
                    canvas.paste(plots[model_type][run_type][metric_type][dev_type], (x, y))

    canvas.resize((_W, _H)).save('plot_dev.png')
