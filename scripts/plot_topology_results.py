"""Publication figures from a COMPLETE, hash-verified topology analysis only.

No provider access. No provisional results, ranking, confidence intervals or
fabricated performance for infeasible allocations. Fixed display order is not a
performance ordering. Tests exercise selection/data contracts without rendering.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROUTES=('aligned','shuffled')
CONDITIONS=('clean','false_report')
# Display order is deliberately independent of results.
LLM_ROWS=(
 ('path','Structural paths + path groups','structural_path_pairs','directed_path',7,2,24000),
 ('exact','Same selection + exact random groups','structural_path_pairs','exact_matched_random',7,2,24000),
 ('random','Random selection (3 seeds) + path groups','random','directed_path',7,2,24000),
 ('mesa','MESA static selection + path groups','mesa_static','directed_path',7,2,24000),
 ('learned','Learned topology pairs','learned_pairs','learned_pairs',7,2,24000),
)
REFERENCE_ROWS=(
 ('selected24','Path selection: 7 edges; group cap 7; 24 KB','structural_path_pairs','full_selected',7,7,24000),
 ('selected60','Path selection: 7 edges; group cap 7; 60 KB','structural_path_pairs','full_selected_relaxed',7,7,60000),
 ('network24','Full network: 13 edges; group cap 13; 24 KB','all_edges','full_selected',13,13,24000),
 ('network60','Full network: 13 edges; group cap 13; 60 KB','all_edges','full_selected_relaxed',13,13,60000),
)
POSITIONS={'path':0,'exact':1,'random':2,'mesa':3,'learned':4,'rule_grouped':6,'rule_joined':7,
           'selected24':9,'selected60':10,'network24':11,'network60':12}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def safe_path(base,name):
    p=Path(name)
    if p.is_absolute() or '..' in p.parts:raise ValueError('unsafe analysis path')
    return base/p


def verified_analysis(directory):
    root=Path(directory);analysis=root/'analysis'
    if not (analysis/'manifest.json').is_file():raise ValueError('complete analysis manifest required before plotting')
    manifest=json.loads((analysis/'manifest.json').read_text())
    if manifest.get('coverage',{}).get('complete') is not True:raise ValueError('refusing incomplete analysis')
    plan=json.loads((root/'plan.json').read_text());coverage=json.loads((root/'coverage.json').read_text())
    expected={'complete':True,'unique_views':plan['unique_views'],
              'assignments':plan['assignments']+plan['expectations'],'case_clusters':12}
    if manifest['run_name']!=root.name or manifest['coverage']!=expected:raise ValueError('complete analysis coverage differs from frozen plan')
    if coverage.get('complete') is not True or coverage.get('unique_scored_views')!=plan['unique_views'] or coverage.get('unique_planned_views')!=plan['unique_views']:
        raise ValueError('refusing incomplete frozen run')
    if set(manifest['inputs'])!={'plan.json','episodes.jsonl','assignments.jsonl','exact_random_plans.jsonl','views.jsonl'}:
        raise ValueError('analysis input contract mismatch')
    for name,want in manifest['inputs'].items():
        if sha(safe_path(root,name))!=want:raise ValueError('stale analysis input: '+name)
    if not {'summary.json','primary_contrast.json'}<=set(manifest['output_sha256']):raise ValueError('required analysis outputs missing')
    for name,want in manifest['output_sha256'].items():
        if sha(safe_path(analysis,name))!=want:raise ValueError('analysis output hash mismatch: '+name)
    if manifest['execution_source_sha256']!=plan['source_hashes']:raise ValueError('execution source declaration mismatch')
    if not manifest['analysis_source_sha256'] or not plan['source_hashes']:raise ValueError('source hashes required')
    for paths,base in ((manifest['analysis_source_sha256'],analysis/'source'),(plan['source_hashes'],root/'source')):
        for name,want in paths.items():
            if sha(safe_path(base,name))!=want:raise ValueError('analysis/frozen source hash mismatch: '+name)
    return (json.loads((analysis/'summary.json').read_text())['rows'],
            json.loads((analysis/'primary_contrast.json').read_text()),manifest)


def _number(value,label,low=None,high=None):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('undefined/nonfinite '+label)
    if low is not None and value<low-1e-10 or high is not None and value>high+1e-10:raise ValueError('out-of-range '+label)
    return float(value)


def _interval(row,key,upper):
    value=_number(row[key],key,0,1);hi=_number(row[upper],upper,value,1)
    return {'value':value,'lower':value,'upper':hi}


def _index(rows):
    result={}
    for row in rows:
        if row['workload_size']!=16 or row['condition'] not in CONDITIONS or row['environment'] not in ROUTES:continue
        key=(row['environment'],row['condition'],row['selector'],row['grouper'],row['k'],row['s'],row['input_byte_allowance'])
        if key in result:raise ValueError('duplicate summary policy cell')
        result[key]=row
    return result


def _get_pair(index,route,spec):
    identity,label,selector,grouper,k,s,allowance=spec
    rows=[]
    for condition in CONDITIONS:
        key=(route,condition,selector,grouper,k,s,allowance)
        if key not in index:raise ValueError('missing requested policy summary: '+str(key))
        row=index[key]
        if row['case_clusters']!=12 or row['missing'] or row['assignments']<=0:raise ValueError('incomplete policy/cluster coverage')
        if row.get('information_access')!=['anonymous directed topology only']:raise ValueError('non-topology allocator entered prospective figure')
        rows.append(row)
    if rows[0]['assignments']!=rows[1]['assignments']:raise ValueError('unequal clean/attack weighting')
    return rows


def _llm_row(index,route,spec):
    identity,label,selector,grouper,k,s,allowance=spec;clean,attack=_get_pair(index,route,spec)
    result={'id':identity,'label':label,'detector':'LLM','selector':selector,'grouper':grouper,
            'k':k,'s':s,'input_byte_allowance':allowance,'assignments':clean['assignments']+attack['assignments']}
    infeasible=clean['infeasible']+attack['infeasible']
    if infeasible:
        # One condition may fit fully while another does not: the joint display
        # still has no complete-distribution result or average execution cost.
        return {**result,'defined':False,'infeasible_assignments':infeasible,
                'recall':None,'fpr':None,'mean_calls':None,'mean_tokens':None,
                'status':'Input cap exceeded; complete-distribution performance undefined'}
    if not clean['llm_performance_defined'] or not attack['llm_performance_defined']:raise ValueError('unexpected undefined feasible policy')
    n=result['assignments']
    return {**result,'defined':True,'infeasible_assignments':0,
            'recall':_interval(attack,'source_recall','source_recall_upper'),
            'fpr':_interval(clean,'source_fpr','source_fpr_upper'),
            'mean_calls':sum(_number(r['mean_calls'],'calls',0)*r['assignments'] for r in (clean,attack))/n,
            'mean_tokens':sum(_number(r['mean_tokens'],'tokens',0)*r['assignments'] for r in (clean,attack))/n,
            'status':'Complete clean/false-report distribution'}


def _rule_row(index,route,grouped=True):
    clean,attack=_get_pair(index,route,LLM_ROWS[0]);kind='grouped' if grouped else 'joined'
    recall=_number(attack['rule_'+kind+'_recall'],'deterministic recall',0,1)
    # All selected sources in a clean episode are benign: k * assignments.
    fpr=_number(clean['rule_'+kind+'_fp']/(7*clean['assignments']),'deterministic clean FPR',0,1)
    return {'id':'rule_'+kind,'label':'Record checker within path groups' if grouped else 'Record checker joining all 7 acquired edges',
            'detector':'deterministic record checker','defined':True,'selector':'structural_path_pairs',
            'grouper':'directed_path' if grouped else 'post-acquisition pooled join','k':7,'s':2 if grouped else 7,
            'input_byte_allowance':None,'assignments':clean['assignments']+attack['assignments'],
            'recall':{'value':recall,'lower':recall,'upper':recall},
            'fpr':{'value':fpr,'lower':fpr,'upper':fpr},'mean_calls':0.,'mean_tokens':0.,
            'status':'No LLM calls; deterministic parsing is not free computation; pooled join is a cross-group reference' if not grouped else 'No LLM calls; deterministic parsing is not free computation'}


def figure_data(rows,contrast):
    """Pure fixed-filter extraction. Synthetic tests use this, never rendering."""
    index=_index(rows);panels={}
    for route in ROUTES:
        matched=[_llm_row(index,route,spec) for spec in LLM_ROWS]
        if any(not row['defined'] for row in matched):raise ValueError('matched-budget main panel must be fully feasible')
        primary=contrast[route]
        if len(primary['per_case'])!=12 or len({c['case_id'] for c in primary['per_case']})!=12:raise ValueError('primary contrast needs twelve case clusters')
        recall=primary['deltas']['recall'];fpr=primary['deltas']['clean_selected_fpr']
        for values in (recall,fpr):
            _number(values['mean'],'primary mean',-1,1)
            _number(values['mean_lower'],'primary lower',-1,values['mean'])
            _number(values['mean_upper'],'primary upper',values['mean'],1)
        if abs(matched[0]['recall']['value']-matched[1]['recall']['value']-recall['mean'])>1e-8:
            raise ValueError('primary recall delta disagrees with summary')
        if abs(matched[0]['fpr']['value']-matched[1]['fpr']['value']-fpr['mean'])>1e-8:
            raise ValueError('primary clean FPR delta disagrees with summary')
        panels[route]={'rows':matched+[_rule_row(index,route),_rule_row(index,route,False)]+
                        [_llm_row(index,route,spec) for spec in REFERENCE_ROWS],
                       'primary_delta':{'recall':recall,'clean_fpr':fpr}}
    return {'scope':'Development only; twelve original case clusters, shared graph skeleton, 16-case workloads.',
            'cost_scope':'Mean standalone logical judge calls and input+output tokens; equal clean/false-report weighting. Matched edge/group budgets need not match inference cost. Not research spending.',
            'unit_scope':'False-origin communication edges, not actors or malicious intent. Recall includes never-acquired true edges; clean FPR denominator is acquired benign edges.',
            'bounds_scope':'Operational values and worst-case abstention bounds, not sampling confidence intervals.',
            'selection_scope':'Sampled acquisition uses three fixed seeds on one anonymous graph. This is not an exact expectation over random edge selection.',
            'display_order':'Fixed before examining results; not a policy ranking.', 'panels':panels}


def render(directory,out):
    """The only rendering entry point; integrity gate runs before any output write."""
    root=Path(directory);rows,contrast,manifest=verified_analysis(root);data=figure_data(rows,contrast)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none',
                         'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(23,13.8),facecolor='white')
    grid=fig.add_gridspec(1,6,width_ratios=(1.8,1.65,1.5,1.8,1.65,1.5),
                         left=.285,right=.985,top=.79,bottom=.215,wspace=.23)
    recall_color='#0072B2';fpr_color='#D55E00';rule_color='#009E73';ref_color='#777777'
    maxima=[r['fpr']['upper'] for panel in data['panels'].values() for r in panel['rows'] if r['defined']]
    top=max(maxima,default=0)
    fpr_limit=next((n for n in (.05,.1,.2,.4,.6,.8,1.) if top<=n),1.)
    for route_number,route in enumerate(ROUTES):
        panel=data['panels'][route];ax_r=fig.add_subplot(grid[0,3*route_number]);ax_f=fig.add_subplot(grid[0,3*route_number+1]);ax_c=fig.add_subplot(grid[0,3*route_number+2])
        axes=(ax_r,ax_f,ax_c)
        for ax in axes:
            ax.set_ylim(13,-1.1);ax.set_yticks([])
            for y in (5,8):ax.axhline(y,color='#BBC3CD',lw=.7,zorder=0)
            for y in (0,1,2,3,4):ax.axhspan(y-.40,y+.40,color='#F4F7FA',zorder=-1)
        for ax,limit,title in ((ax_r,1.,'False-origin edge recall ↑'),(ax_f,fpr_limit,'Clean inspected-edge FPR ↓')):
            ax.set_xlim(-.015*limit,1.20*limit);ticks=[0,limit/2,limit]
            ax.set_xticks(ticks,[f'{100*t:g}%' for t in ticks]);ax.set_title(title,fontsize=11,pad=12)
            ax.grid(axis='x',color='#DDE3E9',lw=.65);ax.set_axisbelow(True);ax.spines['left'].set_visible(False)
        ax_c.set_xlim(0,1);ax_c.axis('off');ax_c.set_title('Mean calls  |  tokens',fontsize=11,pad=12)
        for row in panel['rows']:
            y=POSITIONS[row['id']];is_rule=row['id'].startswith('rule_');is_ref=row['id'] in {r[0] for r in REFERENCE_ROWS}
            for ax,key,color in ((ax_r,'recall',recall_color),(ax_f,'fpr',fpr_color)):
                if not row['defined']:
                    if key=='recall':ax.text(.01,y,'Input cap exceeded',fontsize=9.5,color='#6E737A',va='center')
                    else:ax.text(.02*fpr_limit,y,'undefined',fontsize=9.5,color='#6E737A',va='center')
                    continue
                metric=row[key];base=rule_color if is_rule else ref_color if is_ref else color
                ax.barh(y,metric['value'],height=.42,color=base,alpha=.62,edgecolor='none',zorder=2)
                ax.plot([metric['lower'],metric['upper']],[y,y],color=base,lw=2,zorder=3)
                ax.plot(metric['value'],y,marker='s' if is_rule else 'o',ms=4.3,color=base,zorder=4)
                if metric['upper']>metric['lower']+1e-12:ax.plot([metric['upper'],metric['upper']],[y-.12,y+.12],color=base,lw=1.2)
                scale=1. if key=='recall' else fpr_limit
                ax.text(metric['upper']+.025*scale,y,f"{100*metric['value']:.1f}",va='center',fontsize=9,color='#293443')
            if row['defined']:
                value='0  |  —' if is_rule else f"{row['mean_calls']:.2f}  |  {row['mean_tokens']:,.0f}"
                ax_c.text(.02,y,value,va='center',fontsize=10,color='#293443')
            else:
                ax_c.text(.02,y,f"—  ({row['infeasible_assignments']} of {row['assignments']} infeasible)",va='center',fontsize=8.7,color='#6E737A')
        if route_number==0:
            ax_r.set_yticks([POSITIONS[r['id']] for r in panel['rows']],[r['label'] for r in panel['rows']])
            ax_r.tick_params(axis='y',length=0,pad=14,labelsize=10.5)
            for y,label in ((-0.76,'MATCHED EDGE AND GROUP BUDGETS · 7 EDGES · CAP 2 · 24 KB'),(5.34,'SEPARATE DETECTOR · SAME 7 ACQUIRED EDGES'),(8.34,'DIFFERENT GROUP / INPUT / EDGE BUDGET REFERENCES')):
                ax_r.text(-.03,y,label,transform=ax_r.get_yaxis_transform(),ha='right',va='center',fontsize=9,fontweight='bold',color='#4B5B70')
        left=ax_r.get_position().x0;right=ax_c.get_position().x1
        fig.text((left+right)/2,.872,'Aligned audit copies' if route=='aligned' else 'Shuffled audit copies',ha='center',fontsize=17,fontweight='bold',color='#1D2C40')
        dr=panel['primary_delta']['recall']['mean'];df=panel['primary_delta']['clean_fpr']['mean']
        fig.text((left+right)/2,.844,f'Paths − exact random: Δ recall {100*dr:+.1f} pp; Δ clean FPR {100*df:+.1f} pp',ha='center',fontsize=10.3,color='#4B5B70')
    fig.text(.035,.96,'Budgeted monitoring from anonymous topology',fontsize=25,fontweight='bold',color='#13253B')
    fig.text(.035,.924,'Development · 16 cases per workload · 12 original target-case clusters · one shared graph skeleton',fontsize=13,color='#4B5B70')
    handles=[Line2D([0],[0],color=recall_color,marker='o',lw=2,label='LLM recall'),Line2D([0],[0],color=fpr_color,marker='o',lw=2,label='LLM false flags'),
             Line2D([0],[0],color=rule_color,marker='s',lw=2,label='Deterministic record checker'),Line2D([0],[0],color=ref_color,marker='o',lw=2,label='Different-budget LLM reference')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.03,.142),ncol=4,frameon=False,fontsize=11)
    notes=[
        'Points/bars show operational values; whiskers span worst-case abstention resolutions. These are not sampling confidence intervals. No winner or significance claim.',
        'Exact random groups preserve selection and group sizes. Sampled random acquisition uses only three fixed seeds on one anonymous graph; it is not an exact selection expectation.',
        'Costs average logical judge calls and input + output tokens over clean/false-report episodes; retries contribute tokens. Equal edge/group caps need not match inference cost. These are not research spending.',
        'The deterministic checker reads acquired messages after allocation. Its pooled join crosses group boundaries; zero LLM calls does not mean zero computational cost.',
        'Full-context and relaxed-input references use different caps or edge budgets. Infeasible cells have no performance or execution-cost estimate. Controlled false reports are not autonomous attack rollouts.',
        'Recall counts true source edges across the full graph, including missed acquisitions. Clean FPR divides false flags by acquired benign edges. Neither measures actor detection or malicious intent.'
    ]
    for i,note in enumerate(notes):fig.text(.035,.126-i*.020,note,fontsize=9.4,color='#4B5B70')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    base=out/'topology_workload_comparison'
    for suffix in ('png','svg','pdf'):fig.savefig(base.with_suffix('.'+suffix),dpi=200,facecolor='white')
    plt.close(fig)
    (out/'figure_data.json').write_text(json.dumps(data,indent=2)+'\n')
    artifact={'scope':'Post-analysis visualization of complete development results; no new estimates or ranking.',
        'analysis_manifest_sha256':sha(root/'analysis/manifest.json'),'analysis_output_sha256':manifest['output_sha256'],
        'plot_source_sha256':sha(Path(__file__)),'files_sha256':{p.name:sha(p) for p in [base.with_suffix('.'+s) for s in ('png','svg','pdf')]+[out/'figure_data.json']}}
    (out/'manifest.json').write_text(json.dumps(artifact,indent=2)+'\n')
    return artifact


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--out',default='paper/figures/topology_workload')
    a=p.parse_args();print(json.dumps(render(a.directory,a.out),indent=2))
