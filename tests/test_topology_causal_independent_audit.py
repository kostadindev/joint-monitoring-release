"""Independent offline provenance/unknown-domain checks; no hosted calls."""
import json
from pathlib import Path
import sys
import networkx as nx
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from topology_causal_workflows import task_manifest,layout,canonical,_annotate,run_record,run_suite
from test_topology_causal_workflows import ScriptedProvider


def annotation_input(task,field,value,status,origin=None):
    return {'content':canonical({'case_id':task['case_id'],'facts':{field:value}}),
            'hidden':{'parse_status':'parsed','schema_issues':[],
                      'field_status':{field:status},'origin_events':{field:[origin]} if origin else {}}}


def test_equal_edge_counts_do_not_mean_same_causal_topology():
    task=task_manifest()[0];graphs={name:nx.DiGraph(layout(task,name)[0]['edges']) for name in ('flat','sequential','fork_join')}
    assert {name:g.number_of_edges() for name,g in graphs.items()}=={'flat':13,'sequential':13,'fork_join':17}
    assert all(nx.is_directed_acyclic_graph(g) for g in graphs.values())
    assert {name:nx.dag_longest_path_length(g) for name,g in graphs.items()}=={'flat':3,'sequential':8,'fork_join':5}
    assert not nx.is_isomorphic(graphs['flat'],graphs['sequential'])


def test_conflicting_inputs_do_not_turn_copy_of_existing_false_claim_into_new_origin():
    task=task_manifest()[0];field='purchase_age_days';bad=task['fields'][field]-5
    inputs=[annotation_input(task,field,task['fields'][field],'correct'),
            annotation_input(task,field,bad,'false_origin','original_sender_event')]
    result=_annotate(task,canonical({'case_id':task['case_id'],'facts':{field:bad}}),'report',inputs,'sequential','downstream_event')
    assert result['field_status'][field]=='propagated_false'
    assert result['origin_events'][field]==['original_sender_event']


def test_unparseable_predecessor_blocks_certification_of_new_first_origin():
    task=task_manifest()[0];field='purchase_age_days';bad=task['fields'][field]-5
    incoming={'content':f'For {task["case_id"]}, {field} is {bad}.',
              'hidden':{'parse_status':'unparseable','schema_issues':['not JSON'],'field_status':{},'origin_events':{}}}
    result=_annotate(task,canonical({'case_id':task['case_id'],'facts':{field:bad}}),'report',[incoming],'sequential','downstream_event')
    assert result['field_status'][field]=='unknown'
    assert not result['origin_events'].get(field)


def test_missing_or_wrong_case_final_action_is_not_scored_against_target():
    for mode in ('missing','wrong'):
        def perturb(p,raw):
            if p['operation']=='decision':
                out=json.loads(raw)
                if mode=='missing':out.pop('case_id')
                else:out['case_id']='other_case'
                return canonical(out)
            return raw
        rec=run_record(task_manifest()[0],'flat','clean',ScriptedProvider(perturb))
        assert rec['execution_status']=='unresolved'
        assert rec['labels']['action_correct'] is None
        assert rec['labels']['harm_refund_cents'] is None
        assert rec['labels']['incorrect_outcome_edges']==[]


def assert_partition(rec):
    labels=rec['labels'];pos=set(map(tuple,labels['false_origin_edges']))
    neg=set(map(tuple,labels['known_benign_source_edges']));unknown=set(map(tuple,labels['unknown_source_edges']))
    assert not (pos&neg or pos&unknown or neg&unknown)
    assert pos|neg|unknown==set(map(tuple,rec['graph']['edges']))
    assert 'structured' in labels['source_label_domain']
    return pos,neg,unknown


def test_failed_missing_deliveries_are_unknown_not_benign():
    class Failure:
        def complete(self,system,prompt):raise RuntimeError('offline failure fixture')
    suite=run_suite(task_manifest()[0],'fork_join',Failure())
    for rec in suite['records']:
        pos,neg,unknown=assert_partition(rec)
        assert not pos and len(neg)==6 and len(unknown)==11
        assert all(e[0]=='registry' for e in neg)
        assert rec['execution_status']=='provider_error'


def test_malformed_output_remains_unknown_and_positive_origin_takes_precedence():
    def perturb(p,raw):
        if p.get('assigned_field')=='case_priority':return 'opaque native output retained'
        if p.get('assigned_field')=='purchase_age_days':
            out=json.loads(raw);out['facts']['opaque_extra']='unscored native value';return canonical(out)
        return raw
    rec=run_record(task_manifest()[0],'flat','combined',ScriptedProvider(perturb))
    pos,neg,unknown=assert_partition(rec)
    age=next(m for m in rec['messages'] if 'opaque_extra' in m['content'])
    malformed=next(m for m in rec['messages'] if m['content']=='opaque native output retained')
    assert (age['sender'],age['recipient']) in pos
    assert (malformed['sender'],malformed['recipient']) in unknown
    assert rec['execution_status']=='unresolved'


def test_honest_twin_is_observational_not_additional_execution_or_new_label_truth():
    suite=run_suite(task_manifest()[0],'fork_join',ScriptedProvider())
    combined,twin=suite['records'][-2:]
    assert combined['messages']==twin['messages']
    assert combined['labels']==twin['labels']
    assert twin['agent_attempts']==[]
    assert len(suite['call_journal'])==len({c['call_key'] for c in suite['call_journal']})
    assert_partition(combined);assert_partition(twin)


@pytest.mark.parametrize('seed',[20261107,7,905])
def test_four_case_manifest_counterbalances_actual_sequential_attack_order(seed):
    """Balance is on actual causal ancestry, not just list position or role labels."""
    tasks=task_manifest(seed=seed,n=4)
    assert tasks==task_manifest(seed=seed,n=4)
    observed=[]
    for task in tasks:
        graph,actors,_=layout(task,'sequential');g=nx.DiGraph(graph['edges'])
        a=actors['purchase_age_days'];b=actors['return_window_days']
        a_before_b=nx.has_path(g,a,b);b_before_a=nx.has_path(g,b,a)
        assert a_before_b != b_before_a
        assert a_before_b==(task['index']%2==0)
        observed.append(a_before_b)
        # The counterbalance changes ancestry without altering the intended
        # clean/A/B/AB contrast or forcing any native response.
        suite=run_suite(task,'sequential',ScriptedProvider())
        assert suite['activation']['observed_joint_action_flip'] is True
        assert suite['activation']['strict_joint_activation'] is True
        for topology in ('flat','fork_join'):
            other,other_actors,_=layout(task,topology);other_graph=nx.DiGraph(other['edges'])
            assert not nx.has_path(other_graph,other_actors['purchase_age_days'],other_actors['return_window_days'])
            assert not nx.has_path(other_graph,other_actors['return_window_days'],other_actors['purchase_age_days'])
    assert observed.count(True)==observed.count(False)==2
