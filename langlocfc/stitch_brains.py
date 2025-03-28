import os
import string
import yaml
import pandas as pd
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import argparse

ALPHA = string.ascii_lowercase

def index_to_char(ix):
    assert ix < 26**2 - 1, 'Index is too big to convert to char'
    head = ix // 26
    tail = ix % 26
    if head > 0:
        head = ALPHA[head]
    else:
        head = ''
    tail = ALPHA[tail]

    return head + tail

def stitch_page(imgs, pad=100):
    col1 = imgs[:len(imgs) // 2:]
    col2 = imgs[len(imgs) // 2:]
    col1_w = max(*[x.size[0] for x in col1])
    col1_h = sum([x.size[1] for x in col1])
    col2_w = max(*[x.size[0] for x in col2])
    col2_h = sum([x.size[1] for x in col1])
    W = col1_w + pad + col2_w
    H = max(col1_h, col2_h)

    canvas = Image.new('RGB', (W, H), color=(255, 255, 255))
    y_offset = 0
    for img in col1:
        canvas.paste(img, (0, y_offset))
        y_offset += img.size[1]
    y_offset = 0
    for img in col2:
        canvas.paste(img, (col1_w + pad, y_offset))
        y_offset += img.size[1]
    return canvas

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Stitch by-participant plots into pages''')
    argparser.add_argument('results_dir', help='Path to results directory')
    argparser.add_argument('-f', '--filename', default='LANA_sub1_vs_Lang_S-N.png', help='Filename for plots to stitch')
    argparser.add_argument('-p', '--parcellation_id', default='main', help='ID of parcellation to use')
    argparser.add_argument('-s', '--start_subject', default=0, type=int, help='Number of first subject to run')
    argparser.add_argument('-S', '--start_index', default='a', help='Alphabetic index of first session to run')
    args = argparser.parse_args()

    index_by_subject = {}
    files = []
    font_main = None
    font_sub = None
    parcel, eval_atlas = args.filename.split('_vs_')
    eval_atlas = eval_atlas.replace('.png', '')

    page = []
    n_per_page = 30
    page_ix = 1
    img_ix = 0

    subject_paths = []
    for x in os.listdir(args.results_dir):
        try:
            int(x.split('_')[0])
            x.split('_')[1]
            subject_paths.append(x)
        except ValueError:
            continue
    subject_paths = sorted(subject_paths, key=lambda x: (int(x.split('_')[0]), x.split('_')[1]))

    if not os.path.exists('stitched'):
        os.makedirs('stitched')

    for subject_path in subject_paths:
        subject = int(subject_path.split('_')[0])
        if subject in index_by_subject:
            index_by_subject[subject] += 1
        else:
            index_by_subject[subject] = 0
        plot_path = os.path.join(
            args.results_dir,
            subject_path,
            'parcellation',
            args.parcellation_id,
            'plots',
            args.filename
        )
        if os.path.exists(plot_path):
            ix = index_to_char(index_by_subject[subject])
            if subject < args.start_subject or ix < args.start_index:
                continue
            cfg_path = os.path.join(
                args.results_dir,
                subject_path,
                'parcellation',
                args.parcellation_id,
                'parcellate_kwargs_optimized.yml'
            )
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
            action_sequence = cfg['action_sequence']

            # Get runs and TRs
            sample_id = None
            k = None
            for action in action_sequence:
                if action['type'] == 'sample':
                    sample_id = action['id']
                    k = action['kwargs']['n_networks']
                    break
            assert sample_id, 'sample_id not found'
            metadata_path = os.path.join(
                args.results_dir,
                subject_path,
                'sample',
                sample_id,
                'metadata.csv'
            )
            metadata = pd.read_csv(metadata_path)
            n_trs = metadata.n_trs.values.squeeze()
            n_runs = metadata.n_runs.values.squeeze()

            # Get performance
            evaluation_id = None
            for action in action_sequence:
                if action['type'] == 'evaluate':
                    evaluation_id = action['id']
                    break
            assert evaluation_id, 'evaluation_id not found'
            evaluation_path = os.path.join(
                args.results_dir,
                subject_path,
                'evaluation',
                evaluation_id,
                'evaluation.csv'
            )
            evaluation = pd.read_csv(evaluation_path)
            sim = evaluation[(evaluation.parcel == parcel)]['eval_%s_score' % eval_atlas].values.squeeze()
            contrast = evaluation[(evaluation.parcel == parcel)]['eval_%s_contrast' % eval_atlas].values.squeeze()

            img = Image.open(plot_path)
            width, height = img.size

            font_main_size = height // 7
            font_sub_size = int(font_main_size * 0.7)
            if font_main is None:
                font_main = ImageFont.truetype("Arial", font_main_size)
            if font_sub is None:
                font_sub = ImageFont.truetype("Arial", font_sub_size)
            W = height
            H = font_main_size
            txt_main = Image.new('RGB', (W, H), color=(255, 255, 255))
            txt_main_text = 'S%s%s' % (subject, ix)
            draw = ImageDraw.Draw(txt_main)
            _, _, w, h = draw.textbbox((0, 0), txt_main_text, font=font_main)
            draw.text(
                ((W-w)/2, (H-h)/2),  # Coordinates
                txt_main_text,  # Text
                (0, 0, 0),  # Color
                font=font_main
            )
            txt_main = txt_main.rotate(90,  expand=1)

            W = height
            H = font_sub_size
            txt_sub = Image.new('RGB', (W, H), color=(255, 255, 255))
            txt_sub_text = 'TRs=%d, runs=%d' % (n_trs, n_runs)
            draw = ImageDraw.Draw(txt_sub)
            _, _, w, h = draw.textbbox((0, 0), txt_sub_text, font=font_sub)
            draw.text(
                ((W-w)/2, (H-h)/2),  # Coordinates
                txt_sub_text,  # Text
                (0, 0, 0),  # Color
                font=font_sub
            )
            txt_sub = txt_sub.rotate(90,  expand=1)

            W = height
            H = font_sub_size
            txt_score = Image.new('RGB', (W, H), color=(255, 255, 255))
            txt_score_text = 'k=%d, sim=%0.2f' % (k, sim)
            draw = ImageDraw.Draw(txt_score)
            _, _, w, h = draw.textbbox((0, 0), txt_score_text, font=font_sub)
            draw.text(
                ((W-w)/2, (H-h)/2),  # Coordinates
                txt_score_text,  # Text
                (0, 0, 0),  # Color
                font=font_sub
            )
            txt_score = txt_score.rotate(90,  expand=1)

            W = height
            H = font_sub_size
            txt_contrast = Image.new('RGB', (W, H), color=(255, 255, 255))
            txt_contrast_text = 'contrast=%0.2f' % contrast
            draw = ImageDraw.Draw(txt_contrast)
            _, _, w, h = draw.textbbox((0, 0), txt_contrast_text, font=font_sub)
            draw.text(
                ((W-w)/2, (H-h)/2),  # Coordinates
                txt_contrast_text,  # Text
                (0, 0, 0),  # Color
                font=font_sub
            )
            txt_contrast = txt_contrast.rotate(90,  expand=1)

            canvas = Image.new('RGB', (width + font_main_size + font_sub_size * 4, height), color=(255, 255, 255))
            canvas.paste(img, (font_main_size + font_sub_size * 4, 0))
            canvas.paste(txt_main, (0, 0))
            canvas.paste(txt_sub, (font_main_size, 0))
            canvas.paste(txt_score, (font_main_size + font_sub_size, 0))
            canvas.paste(txt_contrast, (font_main_size + font_sub_size * 2, 0))

            if img_ix < n_per_page:
                page.append(canvas)
            else:
                page = stitch_page(page)
                page.save('stitched/atlases_%s_%s_page_%d.png' % (args.parcellation_id, args.filename, page_ix))
                page = [canvas]
                img_ix = 0
                page_ix += 1
            img_ix += 1

    if len(page):
        page = stitch_page(page)
        page.save('stitched/atlases_%s_%s_page_%d.png' % (args.parcellation_id, args.filename, page_ix))
