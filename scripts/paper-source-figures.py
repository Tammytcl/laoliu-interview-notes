"""Extract explicitly selected assets from a pinned arXiv LaTeX archive.

Never runs TeX or shell commands from a paper. PDF assets are rendered on their
own canvas, not cropped from a paper page. Review remains a separate manual step.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(paper_id):
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', paper_id):
        raise ValueError('Invalid paper ID')
    config = json.loads((ROOT / 'scripts/figure-crops' / (paper_id + '.json')).read_text())
    cache = ROOT / '.paper-cache' / paper_id
    original = json.loads((cache / 'manifest.json').read_text())
    if original['arxivVersion'] != config['arxivVersion']:
        raise ValueError('Source and plan versions differ')
    source = (cache / 'source').resolve()
    archive_sha = sha(cache / 'source.tar')
    destination = ROOT / 'assets/papers' / paper_id
    manifest_path = destination / 'figures.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['arxivVersion'] != config['arxivVersion']:
        raise ValueError('Existing assets use another version')
    for asset in config.get('sourceAssets', []):
        name = asset['name']
        if not re.fullmatch(r'[a-z0-9-]+-source\.png', name):
            raise ValueError('Invalid output name')
        tex = (source / asset['tex']).resolve()
        paths = [(source / p).resolve() for p in asset.get('paths', [asset.get('path')])]
        if not all(p.is_relative_to(source) for p in paths) or not tex.is_relative_to(source):
            raise ValueError('Source path escapes archive')
        if not tex.is_file() or not asset.get('label'):
            raise ValueError('Record TeX file and figure label')
        panels = []
        for path in paths:
            if path.suffix.lower() == '.pdf':
                doc = pdfium.PdfDocument(path)
                if len(doc) != 1:
                    raise ValueError('Select a single-page figure asset explicitly')
                panel = doc[0].render(scale=3).to_pil().convert('RGB')
            else:
                panel = Image.open(path)
                panel.load()
                if panel.mode in {'RGBA', 'LA'} or 'transparency' in panel.info:
                    rgba = panel.convert('RGBA')
                    panel = Image.new('RGB', rgba.size, 'white')
                    panel.paste(rgba, mask=rgba.getchannel('A'))
            panels.append(panel)
        picture = panels[0]
        if len(panels) > 1:
            # Preserve source panel order; no redraw, data changes or axis edits.
            # Explicit layout is reviewed against the containing TeX figure.
            columns = asset['columns']
            widths = asset.get('panelWidths', [p.width for p in panels])
            panels = [p.resize((w, round(p.height * w / p.width)), Image.Resampling.LANCZOS) for p, w in zip(panels, widths)]
            cell_w, cell_h = max(p.width for p in panels), max(p.height for p in panels)
            picture = Image.new('RGB', (columns * cell_w, ((len(panels) + columns - 1) // columns) * cell_h), 'white')
            for index, panel in enumerate(panels):
                picture.paste(panel, ((index % columns) * cell_w, (index // columns) * cell_h))
        if asset.get('sourcePixelBox'):
            if len(paths) != 1 or not asset.get('trimReason'):
                raise ValueError('Source-image trim needs one asset and the TeX trim justification')
            picture = picture.crop(tuple(asset['sourcePixelBox']))
        picture.save(destination / name)
        entry = {
            'file': name, 'sha256': sha(destination / name),
            'original': 'https://arxiv.org/src/' + config['arxivVersion'],
            'method': 'pinned-latex-asset', 'sourceArchiveSha256': archive_sha,
            'sourcePath': asset.get('path'), 'sourceSha256': sha(paths[0]),
            'sourceFiles': [{'path': str(p.relative_to(source)), 'sha256': sha(p)} for p in paths],
            'texFile': asset['tex'], 'texSha256': sha(tex),
            'figureLabel': asset['label'], 'role': asset['role'],
            'explanation': 'pending', 'visuallyVerified': False,
        }
        for key in ['columns', 'panelWidths', 'sourcePixelBox', 'trimReason']:
            if key in asset:
                entry[key] = asset[key]
        previous = next((e for e in manifest['figures'] if e['file'] == name), {})
        if all(previous.get(k) == v for k, v in entry.items() if k not in {'explanation', 'visuallyVerified'}):
            entry.update({k: previous[k] for k in ['explanation', 'visuallyVerified', 'reportSection', 'reviewedAt', 'reviewScope'] if k in previous})
        manifest['figures'] = [e for e in manifest['figures'] if e['file'] != name] + [entry]
        print(f'{paper_id}/{name} <- {", ".join(str(p.relative_to(source)) for p in paths)}')
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paper_id')
    extract(parser.parse_args().paper_id)
