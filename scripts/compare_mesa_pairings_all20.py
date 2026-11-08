"""Sample-size expansion of the MESA four-channel replay (paper_v2 revision, 2026-11-07).

Identical to compare_mesa_pairings.py (same prompts, views, provider and parser), except
that attacked episodes of all 20 tasks are scored, not only the 10 test tasks. With the
paper's fixed 0.5 threshold no calibration split is needed, so detections and clean false
alarms can be reported over all 20 tasks per rewrite family. Identical requests reuse the
response cache; only the attacked runs of the former calibration tasks are new calls.
Budget: a separate ledger for the user's 20-credit sample-size authorization.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import compare_mesa_pairings as base
from compare_situations import groupings


def prepare(root):
    inputs = [('static', root/'results/keep/pilot_2026-11-06/episodes.jsonl.gz', 'overt'),
              ('dynamic', root/'results/mesa_pilot_followup_gemma/episodes.jsonl', 'overt_dyn')]
    suite, provenance = [], []
    for family, path, attack in inputs:
        raw = base.read(path)
        clean = [r for r in raw if r['condition'] == 'clean']
        if len(clean) != 20: raise ValueError(f'Expected 20 clean tasks in {path}, found {len(clean)}')
        calibration = {r['task_id'] for r in clean[:10]}
        provenance.append({'family': family, 'source': str(path.relative_to(root)),
                           'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'tasks': sorted(r['task_id'] for r in clean)})
        for r in raw:
            if r['condition'] not in ('clean', attack): continue
            msgs = [{'id': m['msg_id'], 'sender': m['sender'], 'recipient': m['recipient'],
                     'content': m['content'], 'round': m['round'], 'timestamp': m['timestamp']}
                    for m in r['messages'] if m['recipient'] == 'hub' and m['sender'].startswith('worker_')]
            rec = {'id': f'{family}:{r["task_id"]}:{r["condition"]}', 'family': family,
                   'case_seed': r['task_id'], 'variant': r['condition'], 'label': r['condition'] != 'clean',
                   'split': 'calibration' if r['task_id'] in calibration else 'test', 'messages': msgs,
                   'recorded_attack_success': r.get('attack_success'),
                   'provenance': 'MESA replay; all four incoming worker channels; no agent rerun; all 20 tasks'}
            groupings(rec)
            suite.append(rec)
    return suite, provenance


RETRY_TOKENS = 256  # the budget guard's output cap


def _retry_truncated():
    """One retry, with a larger output cap, when a reply is cut off before valid JSON.

    Added after the first live run stopped on a truncated reply (static:task_16:overt) at the
    original 160-token cap. Replies that parse are unchanged, and so are their cache keys.
    """
    from jmon.llm import adaption
    from compare_situations import parse
    cls = next(v for v in vars(adaption).values() if isinstance(v, type) and hasattr(v, 'complete_messages'))
    original = cls.complete

    def complete(self, system, user):
        raw, usage = original(self, system, user)
        try:
            parse(raw)
            return raw, usage
        except Exception:
            msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            return self.complete_messages(msgs, max_output_tokens=RETRY_TOKENS)
    cls.complete = complete


if __name__ == '__main__':
    base.prepare = prepare
    _retry_truncated()
    base.main()
