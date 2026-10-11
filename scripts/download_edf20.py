"""Resume-friendly official Sleep-EDF-20 downloader with MNE SHA-1 verification."""
from pathlib import Path
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv,hashlib,json,re,time,os,threading,fcntl
import mne
R=Path(__file__).resolve().parents[1]; D=R/'data/raw/sleep-edf-20'; P=R/'data/.downloads/sleep-edf-20'
S=R/'reports/download_edf20_status.json'; BASE='https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/'
D.mkdir(parents=True,exist_ok=True); P.mkdir(parents=True,exist_ok=True)
lockfile=(P/'download.lock').open('w'); fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
manifest=Path(mne.__file__).parent/'datasets/sleep_physionet/age_records.csv'
rows=[r for r in csv.DictReader(manifest.open()) if int(r['subject'])<20]
assert len(rows)==78 and len({r['subject'] for r in rows})==20
items=json.loads((R/'reports/sleep_edf20_manifest.json').read_text())['files']
assert {i['name']:i['sha1'] for i in items}=={r['fname']:r['sha'] for r in rows}
(R/'reports/sleep_edf20_manifest.json').write_text(json.dumps(dict(source=BASE,checksum_source=str(manifest),files=items),indent=2))
total=sum(i['bytes'] for i in items); guard=threading.Lock(); verified=set(); failed=[]; started=time.time()
def sha(p):
    h=hashlib.sha1()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
for i in items:
    f=D/i['name']
    if f.exists() and f.stat().st_size==i['bytes'] and sha(f)==i['sha1']: verified.add(i['name'])
    elif f.exists(): raise RuntimeError('Existing file fails checksum; preserving for inspection: '+str(f))
chunk=2*1024*1024
def partpath(i,start): return P/(i['name']+'.'+str(start)+'.part')
def status(state='running',error=None):
    done=sum(i['bytes'] for i in items if i['name'] in verified)
    partial=sum(p.stat().st_size for p in P.glob('*.part'))
    obj=dict(state=state,verified_files=len(verified),total_files=len(items),total_recordings=39,subjects=20,
             verified_bytes=done,completed_chunk_bytes=partial,total_bytes=total,
             percent=round(min(100,(done+partial)/total*100),2),elapsed_seconds=round(time.time()-started),
             workers=12,destination=str(D),failed=failed,error=error,updated_unix=time.time())
    temp=S.with_suffix('.tmp'); temp.write_text(json.dumps(obj,indent=2)); temp.replace(S)
def assemble(i):
    name=i['name']
    if name in verified: return
    paths=[partpath(i,s) for s in range(0,i['bytes'],chunk)]
    if not all(p.exists() for p in paths): return
    tmp=P/(name+'.assembling')
    with tmp.open('wb') as out:
        for p in paths: out.write(p.read_bytes())
    if sha(tmp)!=i['sha1']: raise RuntimeError('SHA-1 mismatch: '+name)
    tmp.replace(D/name); verified.add(name)
    for p in paths: p.unlink()
    print('VERIFIED',name,flush=True)
def get(i,start):
    end=min(start+chunk,i['bytes'])-1; p=partpath(i,start)
    if p.exists() and p.stat().st_size==end-start+1: return
    for attempt in range(6):
        try:
            req=Request(BASE+i['name'],headers={'Range':f'bytes={start}-{end}'})
            with urlopen(req,timeout=120) as r:
                assert r.status==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')
                data=r.read()
            assert len(data)==end-start+1
            temp=p.with_suffix('.tmp'); temp.write_bytes(data); temp.replace(p); return
        except Exception:
            if attempt==5: raise
            time.sleep(min(30,2**attempt))
status()
print('START',len(verified),'already verified;',total,'total bytes',flush=True)
try:
    with ThreadPoolExecutor(max_workers=12) as pool:
        jobs={pool.submit(get,i,s):(i,s) for i in items if i['name'] not in verified for s in range(0,i['bytes'],chunk)}
        for k,f in enumerate(as_completed(jobs),1):
            i,s=jobs[f]
            try:
                f.result(); assemble(i)
            except Exception as e:
                failed.append(dict(file=i['name'],offset=s,error=str(e))); print('ERROR',failed[-1],flush=True)
            status()
            if k%12==0: print('PROGRESS',json.loads(S.read_text())['percent'],flush=True)
    if len(verified)!=78: raise RuntimeError('Incomplete download; rerun to resume completed chunks')
    (D/'SHA1SUMS').write_text(''.join(i['sha1']+'  '+i['name']+'\n' for i in items))
    status('completed'); print('COMPLETE: all 78 files verified',flush=True)
except Exception as e:
    status('failed',str(e)); raise
