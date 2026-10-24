"""Restore verified Release assets with bounded temporary disk use; stdlib only."""
import argparse,hashlib,json,os,shutil,stat,time,urllib.parse,urllib.request,zipfile
from pathlib import Path

def digest(path,offset=0,length=None):
    hashed=hashlib.sha256()
    with Path(path).open('rb') as f:
        f.seek(offset)
        while length is None or length:
            block=f.read(8*1024**2 if length is None else min(length,8*1024**2))
            if not block:
                if length:raise EOFError('Short file during checksum')
                break
            hashed.update(block)
            if length is not None:length-=len(block)
    return hashed.hexdigest()

def safe_name(name):
    if not name or Path(name).name!=name or '/' in name or '\\' in name or name in ('.','..'):raise ValueError('Unsafe model or asset name')
    return name

def download(base,record,cache):
    target=cache/safe_name(record['asset_name']);cache.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.stat().st_size==record['bytes'] and digest(target)==record['sha256']:return target
        raise ValueError('Existing asset has the wrong checksum: '+str(target))
    pending=target.with_name(target.name+'.download')
    if pending.is_symlink():raise ValueError('Refuse symlink temporary asset')
    for attempt in range(4):
        offset=pending.stat().st_size if pending.exists() else 0
        if offset>record['bytes']:raise ValueError('Temporary asset is oversized')
        if offset<record['bytes']:
            headers={'User-Agent':'mayi-competition-model-download'}
            if offset:headers['Range']='bytes='+str(offset)+'-'
            try:
                request=urllib.request.Request(base+'/'+urllib.parse.quote(record['asset_name']),headers=headers)
                with urllib.request.urlopen(request,timeout=120) as response:
                    resumed=offset and response.status==206
                    if resumed and not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-'):raise ValueError('Invalid range response')
                    with pending.open('ab' if resumed else 'wb') as output:
                        shutil.copyfileobj(response,output,8*1024**2);output.flush();os.fsync(output.fileno())
            except (OSError,TimeoutError):
                if attempt==3:raise
                time.sleep(2**attempt);continue
        if pending.stat().st_size!=record['bytes'] or digest(pending)!=record['sha256']:raise ValueError('Downloaded asset checksum mismatch: '+record['asset_name'])
        os.replace(pending,target);return target
    raise RuntimeError('Download did not complete')

def restore_tensor(record,output,base,cache,downloader=download):
    name=safe_name(record['path']);target=output/name;pending=output/(name+'.incomplete')
    if target.is_symlink() or pending.is_symlink():raise ValueError('Refuse symlink tensor')
    if target.exists():
        if target.stat().st_size==record['bytes'] and digest(target)==record['sha256']:return
        raise ValueError('Existing complete model file differs; choose another output directory')
    offset=0
    for part in record['parts']:
        if part['offset']!=offset:raise ValueError('Non-contiguous model parts')
        if pending.exists() and pending.stat().st_size>=offset+part['bytes']:
            if digest(pending,offset,part['bytes'])==part['sha256']:offset+=part['bytes'];continue
        if pending.exists():
            with pending.open('r+b') as f:f.truncate(offset)
        piece=downloader(base,part,cache)
        with pending.open('ab') as f, piece.open('rb') as data:
            shutil.copyfileobj(data,f,8*1024**2);f.flush();os.fsync(f.fileno())
        piece.unlink();offset+=part['bytes']
    if offset!=record['bytes'] or pending.stat().st_size!=record['bytes'] or digest(pending)!=record['sha256']:raise ValueError('Reassembled model checksum mismatch')
    os.replace(pending,target)

def validate(manifest):
    if manifest['format']!='lingbot_hf_release_v1' or manifest['repo']!='chunchuxiazhi/mayi-competition':raise ValueError('Unexpected manifest')
    if manifest['step'] not in (10000,20000) or manifest['release_tag']!='lingbot-step-'+str(manifest['step']):raise ValueError('Wrong model step')
    names=[]
    for row in manifest['files']:
        safe_name(row['path']);offset=0
        for part in row['parts']:
            safe_name(part['asset_name']);names.append(part['asset_name'])
            if part['offset']!=offset or not 0<part['bytes']<2**31:raise ValueError('Invalid asset range')
            offset+=part['bytes']
        if offset!=row['bytes']:raise ValueError('Part size total mismatch')
    if len(names)!=len(set(names)):raise ValueError('Duplicate asset name')
    if sum(row['bytes'] for row in manifest['files'])!=manifest['tensor_bytes']:raise ValueError('Model total mismatch')
    return {'step':manifest['step'],'tensor_files':len(manifest['files']),'tensor_parts':len(names),'tensor_bytes':manifest['tensor_bytes']}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,choices=(10000,20000),required=True)
    parser.add_argument('--output',type=Path);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    manifest=json.loads((Path(__file__).resolve().parents[1]/f'models/step_{args.step}/manifest.json').read_text(encoding='utf-8'))
    report=validate(manifest)
    if args.check_only:print(json.dumps(report));return
    if not args.output:parser.error('--output is required unless --check-only')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True);cache=output/'.release-assets'
    base='https://github.com/'+manifest['repo']+'/releases/download/'+manifest['release_tag']
    metadata=download(base,manifest['metadata'],cache)
    with zipfile.ZipFile(metadata) as archive:
        if sum(item.file_size for item in archive.infolist())>100*1024**2:raise ValueError('Metadata ZIP too large')
        for item in archive.infolist():
            safe_name(item.filename)
            if stat.S_ISLNK(item.external_attr>>16):raise ValueError('Refuse ZIP symlink')
            target=output/item.filename;data=archive.read(item)
            if target.is_symlink():raise ValueError('Refuse metadata symlink')
            if target.exists() and target.read_bytes()!=data:raise ValueError('Existing metadata differs: '+str(target))
            target.write_bytes(data)
    metadata.unlink()
    for row in manifest['files']:
        restore_tensor(row,output,base,cache);print('Verified '+row['path'],flush=True)
    print(json.dumps({'restored':str(output),**report}))

if __name__=='__main__':main()
