"""Render manually verified rectangles from pinned PDF; never execute LaTeX."""
import argparse, json, hashlib, re
from pathlib import Path
import pypdfium2 as pdfium
from PIL import ImageStat
ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('paper_id')
parser.add_argument('--refresh', action='store_true')
args = parser.parse_args()
if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', args.paper_id):
    raise ValueError('Invalid paper ID')
config = json.loads((ROOT / 'scripts/figure-crops' / (args.paper_id + '.json')).read_text())
cache = ROOT / '.paper-cache' / args.paper_id
original = json.loads((cache / 'manifest.json').read_text())
if original['arxivVersion'] != config['arxivVersion']:
    raise ValueError('PDF and crop plan versions differ')
source = cache / 'paper.pdf'
pdf_sha = hashlib.sha256(source.read_bytes()).hexdigest()
doc = pdfium.PdfDocument(source)
destination = ROOT / 'assets/papers' / args.paper_id
manifest_path = destination / 'figures.json'
manifest = json.loads(manifest_path.read_text())
if manifest['arxivVersion'] != config['arxivVersion']:
    raise ValueError('Existing assets use another version')
scale = 3
pages = {}
for crop in config['crops']:
    name, number, box = crop['name'], crop['page'], crop['box']
    if not re.fullmatch(r'(?:figure-\d+|table-\d+|tables-\d+-\d+)-pdf\.png', name):
        raise ValueError('Invalid asset name')
    if (destination / name).exists() and not args.refresh:
        raise ValueError('Existing PDF capture: use --refresh after reviewing the crop plan')
    page = doc[number - 1]
    width, height = page.get_size()
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Crop outside PDF page')
    if number not in pages:
        pages[number] = page.render(scale=scale).to_pil().convert('RGB')
    image = pages[number].crop(tuple(round(x * scale) for x in box))
    if max(ImageStat.Stat(image).stddev) < 2:
        raise ValueError('Capture appears blank; review PDF rectangle')
    image.save(destination / name)
    digest = hashlib.sha256((destination / name).read_bytes()).hexdigest()
    previous = next((e for e in manifest['figures'] if e.get('file') == name), None)
    manifest['figures'] = [e for e in manifest['figures'] if e.get('file') != name]
    manifest['figures'].append({
        'file': name, 'sha256': digest,
        'original': f"https://arxiv.org/pdf/{config['arxivVersion']}#page={number}",
        'method': 'pinned-original-pdf-crop', 'pdfPage': number, 'cropBox': box,
        'pdfSha256': pdf_sha, 'scale': scale,
        'explanation': previous.get('explanation', 'pending') if previous else 'pending'
    })
    print(f'{name} <- PDF page {number}')
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
