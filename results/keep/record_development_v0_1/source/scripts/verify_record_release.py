"""Verify hashes, observation/label separation, replay prompts and budget assignments."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys


def rows(path):
    with gzip.open(path,'rt') as f:
        for line in f:yield json.loads(line)


def verify(root):
    manifest=json.loads((root/'manifest.json').read_text())
    for name,want in manifest['files_sha256'].items():
        p=root/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=want:raise ValueError('hash mismatch: '+name)
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='manifest.json'}
    if actual!=set(manifest['files_sha256']):raise ValueError('unmanifested or missing files')
    labels={r['id']:r for r in rows(root/'data/labels.jsonl.gz')}
    records={}
    for r in rows(root/'data/observations.jsonl.gz'):
        if set(r)&{'labels','condition','agent_calls','task_sha256'}:raise ValueError('private fields in observations')
        if not r['id'].startswith('ep_') or len(r['id'])!=23:raise ValueError('nonopaque episode ID')
        if any('hidden' in m for m in r['messages']):raise ValueError('message label leak')
        label=labels[r['id']]
        r.update({key:label[key] for key in ('condition','task_sha256','labels')})
        for m in r['messages']:m['hidden']=label['message_labels'][m['id']]
        records[r['id']]=r
    if set(records)!=set(labels) or len(records)!=manifest['counts']['episodes']:raise ValueError('record count mismatch')
    sys.path.insert(0,str(root/'source/src'));sys.path.insert(0,str(root/'source/scripts'))
    from monitor_record_workflows import view_prompt
    from record_strategies import policy_id
    n=0;seen=set();abstentions=0
    for c in rows(root/'data/judge_views.jsonl.gz'):
        rec=records[c['episode_id']];edges=list(map(tuple,c['edges']))
        if c['prompt']!=view_prompt(rec,edges):raise ValueError('public view does not exactly reproduce recorded prompt')
        key=(rec['id'],tuple(sorted(edges)))
        if key in seen:raise ValueError('duplicate logical view')
        if ('judgment' in c)==('abstention' in c):raise ValueError('ambiguous view disposition')
        seen.add(key);n+=1;abstentions+='abstention' in c
    if n!=manifest['counts']['judge_views']:raise ValueError('view count mismatch')
    assignments=0
    for p in (root/'benchmarks').glob('*/assignments.jsonl.gz'):
        assigned=set();count=0
        for a in rows(p):
            if a['episode_id'] not in records:raise ValueError('unknown assigned episode')
            flattened=[tuple(e) for g in a['groups'] for e in g]
            if len(flattened)!=len(set(flattened)) or len(flattened)>a['k']:raise ValueError('edge budget/reuse violation')
            if any(not 1<=len(g)<=a['s'] for g in a['groups']):raise ValueError('group cap violation')
            if sorted(flattened)!=sorted(map(tuple,a['selected'])):raise ValueError('selection/partition mismatch')
            if any((a['episode_id'],tuple(sorted(map(tuple,g)))) not in seen for g in a['groups']):raise ValueError('assignment references unscored view')
            key=(a['episode_id'],policy_id(a))
            if key in assigned:raise ValueError('duplicate policy assignment')
            assigned.add(key);count+=1
            assignments+=1
        measured=[(r['id'],r['policy']) for r in rows(p.with_name('metrics.jsonl.gz'))]
        if set(measured)!=assigned or len(measured)!=count:raise ValueError('metrics/assignments do not match exactly')
    return {'manifest_files':len(manifest['files_sha256']),'episodes':len(records),'judge_views':n,
            'abstentions':abstentions,'validated_assignments':assignments}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('release');a=p.parse_args();print(json.dumps(verify(Path(a.release)),indent=2))
