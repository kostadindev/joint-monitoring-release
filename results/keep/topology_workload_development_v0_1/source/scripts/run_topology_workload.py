"""Frozen topology-only development plans, exact call reuse and bounded dispatch.

No paid inference without --live. Existing role-aware plans are never modified.
Reused calls retain their original cost as replay cost, not new spending.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import json
from pathlib import Path
import random
import shutil
import threading
import time

from record_workflow import canonical
from monitor_record_workflows import SYSTEM, FORMAT_RETRY, view_prompt
from record_workload_benchmark import EVIDENCE, inspect, parse_workload, analyze
from night_budget import NightBudget, wrap_http
from jmon.llm import make_provider

MODEL = 'gemma-4-31b-it'
SYSTEM_TEXT = SYSTEM + EVIDENCE


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_write(path, text):
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise ValueError('frozen artifact changed: ' + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def validate_call(call, key, rec, edges, system=SYSTEM_TEXT):
    """Bind every raw attempt, score and resource total to the frozen request."""
    prompt = view_prompt(rec, edges)
    expected_bytes = len((system + FORMAT_RETRY + prompt).encode())
    expected_key = hashlib.sha256(canonical([system, prompt]).encode()).hexdigest()
    if key != expected_key or call.get('view_key') != key:
        raise ValueError('view identity mismatch')
    attempts = call.get('attempts', [])
    if ('judgment' in call) == ('abstention' in call):
        raise ValueError('expected exactly one disposition')
    if not 1 <= len(attempts) <= 2 or call.get('request_count') != len(attempts):
        raise ValueError('invalid bounded attempt history')
    for index, item in enumerate(attempts):
        actual_system = system + (FORMAT_RETRY if index else '')
        if sorted(map(tuple, item.get('edges', []))) != sorted(map(tuple, edges)):
            raise ValueError('saved acquisition differs from requested edges')
        if (item.get('prompt') != prompt or item.get('system') != actual_system or
            item.get('attempt') != index or item.get('model') != MODEL or
            item.get('temperature') != 0 or item.get('max_output_tokens') != 1024 or
            item.get('planned_input_bytes') != expected_bytes):
            raise ValueError('request or model settings mismatch')
        expected = hashlib.sha256(canonical([actual_system, prompt]).encode()).hexdigest()
        if item.get('prompt_sha256') != expected:
            raise ValueError('attempt hash mismatch')
        try:
            judgment = parse_workload(item['raw'], rec, edges)
        except (ValueError, TypeError, KeyError):
            if index == len(attempts)-1 and not (len(attempts) == 2 and 'abstention' in call):
                raise ValueError('invalid final response lacks bounded abstention')
        else:
            if index != len(attempts)-1 or call.get('judgment') != judgment or 'abstention' in call:
                raise ValueError('resampled valid answer or altered judgment')
    for field in ('prompt', 'system', 'raw', 'attempt', 'model', 'temperature',
                  'max_output_tokens', 'prompt_sha256', 'planned_input_bytes', 'edges'):
        if call.get(field) != attempts[-1].get(field):
            raise ValueError('final attempt differs from call record')
    for field in ('input_tokens', 'output_tokens', 'cost'):
        values = [a['usage'][field] for a in attempts]
        if any(type(v) not in (int, float) or not 0 <= v < float('inf') for v in values):
            raise ValueError('invalid resource usage')
        total = call['usage'][field]
        if type(total) not in (int, float) or not 0 <= total < float('inf'):
            raise ValueError('invalid aggregate resource usage')
        if abs(total - sum(values)) > 1e-9:
            raise ValueError('incorrect aggregate resource usage')
    return call


def reuse_calls(jobs, sources, destination):
    """Import exact matches with a source hash and line; reject conflicting records."""
    calls = {}
    destination = Path(destination)
    if destination.exists():
        with destination.open() as stream:
            for line in stream:
                call = json.loads(line); key = call['view_key']
                if key not in jobs or key in calls:
                    raise ValueError('unknown or duplicate saved view')
                validate_call(call, key, *jobs[key]); calls[key] = call
    imported = []
    for path in map(Path, sources):
        file_hash = sha(path)
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                call = json.loads(line); key = call['view_key']
                if key not in jobs:
                    continue
                validate_call(call, key, *jobs[key])
                if key in calls:
                    old = {k:v for k,v in calls[key].items() if k != 'reuse_provenance'}
                    if old != call:
                        raise ValueError('conflicting raw judgments for one exact view')
                    continue
                call['reuse_provenance'] = {'path': str(path.resolve()), 'file_sha256': file_hash,
                                           'line': number, 'record_sha256': hashlib.sha256(line.encode()).hexdigest()}
                calls[key] = call; imported.append(call)
    # Finish all validation before mutating the new library.
    if imported:
        with destination.open('a') as stream:
            for call in imported:
                stream.write(json.dumps(call) + '\n')
    return calls, len(imported)


def coverage(assignments, expectations, calls, total_views):
    counts = {'scored': 0, 'missing': 0, 'infeasible': 0}
    for row in assignments + expectations:
        status = 'infeasible' if not row['feasible'] else 'scored' if all(k in calls for k in row['view_keys']) else 'missing'
        counts[status] += 1
    return {'unique_planned_views': total_views, 'unique_scored_views': len(calls),
            'reused_views': sum('reuse_provenance' in c for c in calls.values()),
            'reused_replay_credits': sum(c['usage']['cost'] for c in calls.values() if 'reuse_provenance' in c),
            'new_library_replay_credits': sum(c['usage']['cost'] for c in calls.values() if 'reuse_provenance' not in c),
            'cost_note': 'Replay costs; actual new charges and retained uncertainties are in the shared ledger.',
            'abstentions': sum('abstention' in c for c in calls.values()),
            'assignments': len(assignments)+len(expectations), 'by_status': counts,
            'complete': len(calls) == total_views}


def dispatch_order(records, assignments, expectations, jobs):
    """Freeze complete primary/control blocks before broad secondary exploration."""
    records = {r['id']:r for r in records}
    priority = {key:2 for key in jobs}
    for row in assignments + expectations:
        if not row['feasible']:
            continue
        core = ((row['selector'] == 'structural_path_pairs' and row['k'] == 7 and
                 row['grouper'] in ('directed_path','exact_matched_random','full_selected','full_selected_relaxed') and
                 row['s'] in (2,7)) or row['selector'] == 'all_edges' or 'oracle' in row['selector'])
        selection_control = row['k'] == 7 and row['s'] == 2 and row['grouper'] in ('directed_path','learned_pairs')
        tier = (0 if records[row['episode_id']]['workload_size'] == 16 else 1) if core else 1 if selection_control else 2
        for key in row['view_keys']:
            priority[key] = min(priority[key],tier)
    ordered=[]
    for tier in (0,1,2):
        keys=sorted(k for k,p in priority.items() if p==tier)
        random.Random(2026110620+tier).shuffle(keys);ordered.extend(keys)
    return ordered,priority


def uncommitted_attempts(journal,calls):
    """Never resample a raw attempt lost between journaling and view commit."""
    missing=set()
    if Path(journal).exists():
        with Path(journal).open() as stream:
            for line in stream:
                attempt=json.loads(line)
                system=attempt['system']
                if attempt.get('attempt') == 1 and system.endswith(FORMAT_RETRY):
                    system=system[:-len(FORMAT_RETRY)]
                key=hashlib.sha256(canonical([system,attempt['prompt']]).encode()).hexdigest()
                if key not in calls:missing.add(key)
    return sorted(missing)


def main():
    from topology_workload_plan import plan_library
    parser = argparse.ArgumentParser()
    parser.add_argument('--records', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--learned-model'); parser.add_argument('--reuse', action='append', default=[])
    parser.add_argument('--protocol', default='research/topology_workload_protocol.md')
    parser.add_argument('--prompt-decision', default='results/record_audit_diagnostics_v1/adoption_decision.json')
    parser.add_argument('--ledger', default='results/night_2026-11-06_budget.json')
    parser.add_argument('--live', action='store_true'); parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--phase', default='pilot'); parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--new-view-limit', type=int, default=200)
    parser.add_argument('--phase-cap', type=float, default=.6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.new_view_limit < 1 or not 0 < args.phase_cap <= 10:
        raise ValueError('invalid dispatch bounds')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lock = (out/'.run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    records = [json.loads(line) for line in Path(args.records).read_text().splitlines()]
    if not records or any(r['split'] != 'development' or r['workload_size'] not in (4,16) for r in records):
        raise ValueError('expected development workloads at 4 and 16 cases')
    artifact = json.loads(Path(args.learned_model).read_text()) if args.learned_model else None
    model = artifact['model'] if artifact else None
    if artifact and set(artifact['training_cases']) & {r['case_id'] for r in records}:
        raise ValueError('training and workload target cases overlap')
    adoption = json.loads(Path(args.prompt_decision).read_text())
    if adoption['decision'] != 'retain_original_primary':
        raise ValueError('original prompt contradicts diagnostic adoption decision')
    jobs, assignments, expectations = plan_library(records, SYSTEM_TEXT, model)
    order, priority = dispatch_order(records,assignments,expectations,jobs)
    root = Path(__file__).resolve().parents[1]
    names = ['scripts/' + name + '.py' for name in (
        'run_topology_workload','topology_workload_plan','topology_only_policies',
        'network_strategies','record_workload_benchmark','record_workload_exact_random',
        'record_workflow','monitor_record_workflows','record_audit_analysis','record_audit_policies',
        'record_strategies','compare_situations','record_audit_controls',
        'run_record_audit_diagnostics','night_budget')]
    names += ['src/jmon/allocation/mesa_rank.py','src/jmon/llm/adaption.py']
    import jmon
    package_root = Path(jmon.__file__).resolve().parent
    sources = {name: (package_root/Path(name).relative_to('src/jmon')
                     if name.startswith('src/jmon/') else root/name) for name in names}
    plan = {'scope': 'Topology-only allocation; development positive control, no held-out claim.',
        'records_sha256': sha(args.records), 'episodes': len(records),
        'case_clusters': len({r['case_id'] for r in records}),
        'protocol_sha256': sha(args.protocol), 'prompt_adoption_sha256': sha(args.prompt_decision),
        'source_hashes': {name:sha(sources[name]) for name in names},
        'assignments': len(assignments), 'expectations': len(expectations),
        'feasible_assignments': sum(a['feasible'] for a in assignments), 'unique_views': len(jobs),
        'assignments_sha256': hashlib.sha256(canonical(assignments).encode()).hexdigest(),
        'expectations_sha256': hashlib.sha256(canonical(expectations).encode()).hexdigest(),
        'view_keys_sha256': hashlib.sha256(canonical(sorted(jobs)).encode()).hexdigest(),
        'dispatch_order_sha256': hashlib.sha256(canonical(order).encode()).hexdigest(),
        'dispatch_tiers': {str(t):sum(v==t for v in priority.values()) for t in (0,1,2)},
        'system': SYSTEM_TEXT, 'model': MODEL, 'temperature': 0, 'max_output_tokens': 1024,
        'strict_threshold': .5, 'format_retries': 1,
        'learned_model_sha256': sha(args.learned_model) if artifact else None,
        'ledger': str(Path(args.ledger).resolve()),
        'reuse_sources': [{'path':str(Path(p).resolve()),'sha256':sha(p)} for p in args.reuse]}
    frozen_write(out/'plan.json', json.dumps(plan, indent=2))
    for name in names:
        frozen_write(out/'source'/name, sources[name].read_text())
    for name, source in [('protocol.md', args.protocol), ('prompt_adoption.json',args.prompt_decision)]:
        frozen_write(out/name, Path(source).read_text())
    if artifact:
        frozen_write(out/'learned_model.json', Path(args.learned_model).read_text())
    for name, rows in [('episodes',records),('assignments',assignments),('exact_random_plans',expectations)]:
        frozen_write(out/(name+'.jsonl'), ''.join(json.dumps(row)+'\n' for row in rows))
    frozen_write(out/'dispatch_order.json',json.dumps(order))
    calls, imported = reuse_calls(jobs, args.reuse, out/'views.jsonl')
    def checkpoint():
        data = coverage(assignments, expectations, calls, len(jobs))
        (out/'coverage.json').write_text(json.dumps(data, indent=2)); return data
    print(json.dumps({'plan': {k:plan[k] for k in ('episodes','case_clusters','assignments','expectations','unique_views')},
                      'imported_now': imported, 'coverage': checkpoint()}, indent=2), flush=True)
    if args.analyze:
        if len(calls) != len(jobs):
            raise ValueError('complete all planned views before policy analysis')
        analyze(records, assignments, calls, out, expectations)
    if not args.live:
        return
    if uncommitted_attempts(out/'judge_calls.jsonl',calls):
        raise ValueError('uncommitted raw attempts require audit before another live phase; do not resample')
    phase_path = out/('phase_'+args.phase+'.json')
    if phase_path.exists():
        raise ValueError('phase already attempted; preserve it and choose a new phase name')
    ledger = NightBudget(args.ledger, out.name+'_'+args.phase, args.phase_cap)
    before = ledger.summary()
    if before['blocked'] or before['charged_or_reserved'] + args.phase_cap > before['authorized']:
        raise ValueError('aggregate budget blocked or phase cap exceeds remaining authorization')
    pending_keys = [key for key in order if key not in calls]
    todo = pending_keys[:args.new_view_limit]
    if not todo:
        print('All planned views complete; no model calls.', flush=True); return
    phase = {'phase':args.phase, 'status':'running', 'started_at':time.time(), 'view_keys':todo,
             'completed_before':len(calls), 'phase_cap':args.phase_cap, 'workers':args.workers}
    phase_path.write_text(json.dumps(phase, indent=2))
    provider = make_provider({'kind':'adaption','model':MODEL,'temperature':0,'max_output_tokens':1024,
        'cost':{'allow_live':True,'max_calls':2*len(todo),'max_tokens':20000000,'max_credits':args.phase_cap}})
    wrap_http(provider, ledger, max_input_bytes=60000, max_output_tokens=1024)
    write_lock = threading.Lock(); iterator = iter(todo); errors = []
    def run(key):
        rec, edges = jobs[key]
        call = inspect(rec, edges, SYSTEM_TEXT, provider, out/'judge_calls.jsonl', write_lock)
        call['view_key'] = key
        try:validate_call(call,key,rec,edges)
        except Exception:
            with write_lock,(out/'validation_quarantine.jsonl').open('a') as stream:
                stream.write(json.dumps(call)+'\n')
            raise
        return call
    try:
        with (out/'views.jsonl').open('a') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {}
            def submit():
                key = next(iterator,None)
                if key is not None:
                    pending[pool.submit(run,key)] = key
            for _ in range(args.workers): submit()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    key = pending.pop(future)
                    try:
                        call = future.result(); calls[key] = call
                        stream.write(json.dumps(call)+'\n'); stream.flush()
                        if len(calls)%25 == 0:
                            print(f'Completed {len(calls)}/{len(jobs)}', flush=True)
                    except Exception as exc:
                        errors.append({'view_key':key,'type':type(exc).__name__,'error':str(exc)[:180]})
                if not errors:
                    for _ in done: submit()
    except BaseException as exc:
        errors.append({'type':type(exc).__name__,'error':'Interrupted dispatcher; audit raw journal before resuming.'})
        raise
    finally:
        phase.update(status='failed' if errors else 'complete', finished_at=time.time(), errors=errors,
                     completed_after=len(calls), provider=provider.summary(), ledger=ledger.summary())
        phase_path.write_text(json.dumps(phase, indent=2)); checkpoint()
        print(json.dumps({k:v for k,v in phase.items() if k!='view_keys'}, indent=2), flush=True)
    if errors: raise SystemExit(1)


if __name__ == '__main__':
    main()
