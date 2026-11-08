"""Persistent shared reservation ledger for tonight's explicitly authorized 10 credits.
Every uncached request reserves 0.05 credits before dispatch; failures retain reservation.
This is a conservative reservation, not a provider-enforced billing price guarantee.
No automatic retries. Unknown or unexpectedly high cost blocks subsequent requests.
"""
from pathlib import Path
import fcntl
import json
import os
import time
import uuid
from contextlib import contextmanager

class NightBudget:
    def __init__(self,path,batch,batch_limit,total_limit=10.0,reserve=0.05):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.batch=batch;self.batch_limit=batch_limit;self.total_limit=total_limit;self.reserve=reserve
    @contextmanager
    def edit(self):
        with open(str(self.path)+'.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            d=json.loads(self.path.read_text()) if self.path.exists() else {'authorized_credits':self.total_limit,'calls':{},'batches':{},'blocked':False}
            if d['authorized_credits']!=self.total_limit: raise RuntimeError('ledger authorization mismatch')
            old=d['batches'].setdefault(self.batch,self.batch_limit)
            if old!=self.batch_limit: raise RuntimeError('batch limit cannot silently change')
            try: yield d
            finally:
                tmp=self.path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));os.replace(tmp,self.path)
                fcntl.flock(lock,fcntl.LOCK_UN)
    def reserve_call(self):
        with self.edit() as d:
            total=sum(c['charge'] for c in d['calls'].values())
            batch=sum(c['charge'] for c in d['calls'].values() if c['batch']==self.batch)
            if d['blocked'] or total+self.reserve>self.total_limit or batch+self.reserve>self.batch_limit:
                raise RuntimeError('persistent budget exhausted or blocked; no request sent')
            k=uuid.uuid4().hex;d['calls'][k]={'batch':self.batch,'charge':self.reserve,'state':'reserved','at':time.time()}
            return k
    def settle(self,k,cost):
        with self.edit() as d:
            if cost is None or not isinstance(cost,(int,float)) or not (0<=cost<=self.reserve):
                if isinstance(cost,(int,float)) and cost>self.reserve: d['calls'][k]['charge']=cost
                d['blocked']=True
                raise RuntimeError('unknown or above-reservation cost; future requests blocked')
            d['calls'][k].update(charge=cost,state='settled')
    def block(self,k):
        with self.edit() as d:
            d['blocked']=True;d['calls'][k]['state']='uncertain_retained'
    def summary(self):
        with self.edit() as d:
            return {'authorized':d['authorized_credits'],'charged_or_reserved':sum(c['charge'] for c in d['calls'].values()),
                    'batch':self.batch,'batch_charged_or_reserved':sum(c['charge'] for c in d['calls'].values() if c['batch']==self.batch),
                    'requests':len(d['calls']),'blocked':d['blocked']}


def wrap_http(provider,ledger,max_input_bytes=24000,max_output_tokens=256):
    original=provider.http
    def guarded(payload):
        # Bounded prompts and outputs make the conservative reservation auditable.
        if sum(len(m['content'].encode()) for m in payload['input'])>max_input_bytes or payload['max_output_tokens']>max_output_tokens:
            raise RuntimeError('request exceeds night experiment size limit')
        k=ledger.reserve_call()
        try: result=original(payload)
        except Exception:
            ledger.block(k);raise
        ledger.settle(k,(result.get('usage') or {}).get('cost'))
        return result
    provider.http=guarded
