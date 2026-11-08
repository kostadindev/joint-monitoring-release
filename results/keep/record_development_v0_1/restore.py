from pathlib import Path
import argparse,gzip,json
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parent;out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
def rows(name):
    with gzip.open(root/'data'/name,'rt') as f:return [json.loads(l) for l in f]
labels={r['id']:r for r in rows('labels.jsonl.gz')}
with (out/'episodes.jsonl').open('w') as f:
    for r in rows('observations.jsonl.gz'):
        label=labels[r['id']]
        for key in ('condition','task_sha256','labels'):r[key]=label[key]
        for m in r['messages']:m['hidden']=label['message_labels'][m['id']]
        f.write(json.dumps(r)+'\n')
with gzip.open(root/'data/judge_views.jsonl.gz','rt') as src,(out/'views.jsonl').open('w') as dst:
    for line in src:dst.write(line)
print('Restored evaluator records and judge views; no model calls.')
