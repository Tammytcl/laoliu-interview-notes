"""Download a pinned arXiv version for the manual paper-reading workflow."""
import argparse,gzip,hashlib,json,pathlib,re,tarfile,urllib.request,urllib.parse
ROOT=pathlib.Path(__file__).resolve().parents[1]
def download(url, target):
    if target.exists(): return target.read_bytes()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request=urllib.request.Request(url,headers={'User-Agent':'laoliu-paper-reading/2.0 (personal research)'})
    with opener.open(request,timeout=45) as response:
        payload=response.read(150_000_001)
    if len(payload)>150_000_000: raise ValueError('单个资源超过 150 MB，请手动检查')
    target.write_bytes(payload)
    return payload

def fetch(paper_id, version, source_only=False):
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',paper_id): raise ValueError('无效论文 id')
    if not re.fullmatch(r'\d{4}\.\d{4,5}v\d+|[a-z-]+(?:\.[A-Z]{2})?/\d{7}v\d+',version): raise ValueError('请指定带版本的 arXiv id，如 2305.13245v3')
    cache=ROOT/'.paper-cache'/paper_id
    if cache.is_symlink(): raise ValueError('缓存目录不能是符号链接')
    cache.mkdir(parents=True,exist_ok=True)
    previous=cache/'manifest.json'
    if previous.exists() and json.loads(previous.read_text())['arxivVersion']!=version:
        raise ValueError('已有不同版本缓存，请先检查并移走该论文缓存，避免混用来源')
    manifest={'paperId':paper_id,'arxivVersion':version,'originalUrl':'https://arxiv.org/abs/'+version,'files':{},'errors':{}}
    if source_only and previous.exists():
        manifest = json.loads(previous.read_text())
        manifest.setdefault('files', {})
        manifest.setdefault('errors', {})
    for name,kind in ([('source.tar','src')] if source_only else [('source.tar','src'),('paper.pdf','pdf'),('paper.html','html')]):
        url=f'https://arxiv.org/{kind}/{version}'
        try:
            data=download(url,cache/name)
            if name=='paper.pdf' and not data.startswith(b'%PDF'): raise ValueError('响应不是 PDF')
            if name=='paper.html' and b'<html' not in data.lower(): raise ValueError('响应不是 HTML')
            manifest['files'][name]={'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            print(f'✓ {name}: {len(data):,} bytes')
        except Exception as error:
            manifest['errors'][name]=str(error); print(f'未获取 {name}: {error}')
    archive=cache/'source.tar'
    if 'source.tar' in manifest['files']:
        destination=cache/'source'; destination.mkdir(exist_ok=True)
        try:
            with tarfile.open(archive) as source:
                members=source.getmembers()
                if len(members)>10000 or sum(m.size for m in members)>500_000_000: raise ValueError('源码解压规模超过限制')
                for member in members:
                    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()) or not (destination/member.name).resolve().is_relative_to(destination.resolve()): raise ValueError('源码包含越界路径或非普通文件')
                source.extractall(destination,filter='data')
        except tarfile.ReadError:
            try: data=gzip.decompress(archive.read_bytes())
            except OSError: data=archive.read_bytes()
            if len(data)>500_000_000: raise ValueError('源码过大')
            (destination/'main.tex').write_bytes(data)
        except Exception as error: manifest['errors']['extraction']=str(error); print(f'未解压: {error}')
    if not source_only and 'paper.html' in manifest['files']:
        images=cache/'html-images'; images.mkdir(exist_ok=True)
        manifest['files']['html-images']={}
        for source in dict.fromkeys(re.findall(r'<(?:img[^>]+src|object[^>]+data)="([^"]+)"',(cache/'paper.html').read_text())):
            if not source.startswith(version+'/'): continue
            image_url=urllib.parse.urljoin('https://arxiv.org/html/',source)
            name=hashlib.sha256(source.encode()).hexdigest()[:12]+'-'+pathlib.Path(source).name
            try:
                download(image_url,images/name); manifest['files']['html-images'][source]=name
            except Exception as error: manifest['errors'][source]=str(error)
    (cache/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('缓存：'+str(cache))
    if not manifest['files']: raise SystemExit(1)
    return manifest
if __name__=='__main__':
    parser=argparse.ArgumentParser(description='保存 arXiv 源码、PDF、HTML 到本地缓存，不生成精读正文、不执行 LaTeX。')
    parser.add_argument('paper_id'); parser.add_argument('arxiv_version')
    parser.add_argument('--source-only', action='store_true', help='只下载并安全解压固定版本 LaTeX 源码')
    args=parser.parse_args(); fetch(args.paper_id,args.arxiv_version,args.source_only)
