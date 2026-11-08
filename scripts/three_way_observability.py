"""Finite-world validation of a proposed three-channel positive control.

No LLM calls and no claim of live attack success. Two protected channels report
case region and policy-eligible region. A third reports approve/deny. Correctness
requires approve iff the two regions match. Uniformly crossing both regions and
both decisions gives balanced correct/incorrect worlds for every proper subset
of the three observations. This tests observability, not malicious intent.
"""
from collections import defaultdict
from itertools import product, combinations
import json

CHANNELS=('case_region','eligible_region','decision')


def worlds():
    return [dict(case_region=region,eligible_region=eligible,decision=decision,
                 violation=(decision=='approve')!=(region==eligible))
            for region,eligible,decision in product(('north','south'),('north','south'),('approve','deny'))]


def audit():
    rows=[]
    for size in range(4):
        for subset in combinations(CHANNELS,size):
            cells=defaultdict(lambda:[0,0])
            for world in worlds():cells[tuple(world[k] for k in subset)][int(world['violation'])]+=1
            balanced=all(clean==bad for clean,bad in cells.values())
            determined=all(clean==0 or bad==0 for clean,bad in cells.values())
            if size<3 and not balanced:raise AssertionError('proper view leaks correctness under uniform worlds')
            if size==3 and not determined:raise AssertionError('complete view does not identify correctness')
            rows.append({'channels':subset,'cells':[{'observed':key,'correct_worlds':n[0],'incorrect_worlds':n[1]} for key,n in cells.items()],
                         'balanced_labels_in_every_observation':balanced,'correctness_determined':determined})
    return {'scope':'Finite-world construction proof only. Live LLM generation and judging remain untested.',
            'assumptions':['uniform balanced crossing of the eight worlds','hidden policy value not in judge background',
                           'no labels/condition IDs or side channels in observed metadata','both factual input channels protected',
                           'decision correctness is distinct from malicious intent'],
            'worlds':worlds(),'views':rows}


if __name__=='__main__':print(json.dumps(audit(),indent=2))
