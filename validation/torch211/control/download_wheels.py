import concurrent.futures,hashlib,json,pathlib,urllib.parse,urllib.request,time
root=pathlib.Path('/workspace/wheels');root.mkdir(exist_ok=True)
packages=[('triton','3.6.0'),('nvidia-cudnn-cu12','9.19.0.56'),('nvidia-nccl-cu12','2.28.9'),('nvidia-nvshmem-cu12','3.4.5')]
items=[]
index=urllib.request.urlopen('https://download.pytorch.org/whl/cu128/torch/',timeout=45).read().decode()
import re,html
link=next(html.unescape(x) for x in re.findall(r'href="([^"]+)"',index) if 'torch-2.11.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl' in x)
url,fragment=link.split('#');items.append(dict(url=url,sha256=fragment.split('=',1)[1]))
for name,version in packages:
    j=json.load(urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json'))
    rows=[r for r in j['urls'] if 'x86_64' in r['filename'] and ('cp312' in r['filename'] or 'py3-' in r['filename'])]
    assert len(rows)==1,(name,[r['filename'] for r in rows])
    r=rows[0];items.append(dict(url=r['url'],sha256=r['digests']['sha256']))
def download(item):
    url=item['url'];name=urllib.parse.unquote(url.rsplit('/',1)[-1]);path=root/name
    with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=45) as r:size=int(r.headers['Content-Length'])
    chunk=(size+15)//16
    def part(i):
        lo=i*chunk;hi=min(size,(i+1)*chunk)-1;target=root/f'{name}.part{i}'
        for attempt in range(3):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}'})
                with urllib.request.urlopen(req,timeout=90) as response,target.open('wb') as f:
                    assert response.status==206
                    while block:=response.read(1024*1024):f.write(block)
                assert target.stat().st_size==hi-lo+1
                return target
            except Exception:
                if attempt==2:raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:parts=list(pool.map(part,range(16)))
    with path.open('wb') as out:
        for p in parts:
            with p.open('rb') as stream:
                while b:=stream.read(1024*1024):out.write(b)
            p.unlink()
    digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest();assert digest==item['sha256'],name
    row=dict(filename=name,bytes=size,sha256=digest,url=url);print(json.dumps(row),flush=True);return row
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(download,items))
(root/'verification.json').write_text(json.dumps(results,indent=2))
