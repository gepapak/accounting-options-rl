"""HELD bounded frozen-policy gate. --prepare is source/metadata-only.

--run requires a NEW, exact root GO file. The V1 hand GO does not authorize it.
No checkpoint, market file, external module body or native function is opened
by preparation. This is an optional audit feasibility screen, not new training.
"""
from __future__ import annotations
import argparse,ast,hashlib,importlib.util,json,sys,traceback
from fractions import Fraction
from pathlib import Path
import floor_inverse

HERE=Path(__file__).resolve().parent;PAPER=HERE.parents[3]
BASE=PAPER/'research_ai_2026-10-29/future_value_learning_001/generic_ppo_001'
OLD=BASE/'FROZEN_PPO_PROTOCOL_V1.json'
BUNDLE=BASE/'runs/train_v1/TRAINED_BUNDLE_FROZEN_BEFORE_ANY_EVALUATION.json'
PROTOCOL=HERE/'FROZEN_POLICY_EVENT_PROTOCOL_V1.json'
AUTH=HERE/'ROOT_POLICY_EVENT_AUTHORIZATION_V1.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def serial(value):
    if isinstance(value,Fraction):return str(value)
    if isinstance(value,floor_inverse.Interval):return value.record()
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if hasattr(value,'tolist'):return serial(value.tolist())
    if hasattr(value,'item'):return value.item()
    return value
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(serial(value),f,indent=2,allow_nan=False);f.write('\n')
def digest(value):return hashlib.sha256(json.dumps(serial(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def prepare():
    p=read(OLD);bundle=read(BUNDLE)
    selected=[r for r in bundle['models'] if (r['arm'],r['seed'])==('plain_ppo',4102)]
    if len(selected)!=1:raise AssertionError('Pinned existing frozen model metadata missing')
    if p['validation_contexts'][1]['episode']!=15:raise AssertionError('Pinned already-used context metadata changed')
    own=('floor_inverse.py','known_sum_v2.py','known_sum_v3.py','policy_event_observer_v1.py',
         'run_policy_event_gate_v1.py','check_policy_event_observer_v1.py')
    hashes=dict(p['source_sha256'])
    for name in own:
        path=HERE/name;ast.parse(path.read_text(encoding='utf-8'))
        hashes[path.relative_to(PAPER).as_posix()]=sha(path)
    for path in (OLD,BUNDLE,HERE/'POLICY_EVENT_OBSERVER_PURE_CHECKS_V2.json',HERE/'KNOWN_SUM_PURE_CHECKS_V3.json'):
        hashes[path.relative_to(PAPER).as_posix()]=sha(path)
    # Source hashes are checked without importing/executing original modules.
    for rel,h in hashes.items():
        if sha(PAPER/rel)!=h:raise AssertionError('Pinned source/metadata changed: '+rel)
    save(PROTOCOL,dict(status='HELD_SOURCE_ONLY_NOT_AUTHORIZED_TO_EXECUTE',version=1,
      purpose='Instrumented settlement-event audit on a reused frozen learned-policy development path; no policy improvement inference.',
      existing_ppo_protocol_sha256=sha(OLD),selected_model=selected[0],
      selected_context=p['validation_contexts'][1],arm='plain_ppo',training_seed=4102,
      deterministic=True,context_index=1,evaluation_seed=910201,native_rng_seed=7,
      public_reflection_center=500.,no_center_tuning_or_fallback=True,
      source_metadata_sha256=hashes,
      budgets=dict(episodes=2,full_Env_instances=2,native_steps=2016,macro_steps=336,
        selected_PPO_predict_calls=336,selected_PPO_weight_loads=2,new_training=0,
        due_events_max=166,extra_original_settler_calls_max=332,
        extra_original_body_calls_max=10000,additional_public_price_reader_queries_max=166,full_shadow_Env_steps=0),
      before_any_market_or_model_access=['Exact NEW root GO and protocol SHA',
        'All source/metadata hashes unchanged','Pure observer and inverse receipts PASS',
        'Existing public-vs-Prototype settlement-kernel AST parity receipt retained'],
      runtime_parity=['Fresh reference and fresh observed episodes share context/model/seeds',
        'All 1008 native-row records, 168 actor inputs/actions/rewards and terminal financial totals match exactly',
        'Each due event: unchanged-source unflipped minimal proxy matches real cash/remaining queue/gross/per-asset result',
        'Real environment cash/queue remain unchanged by proxy calls'],
      restricted_api=['event pre-cash','event real post-cash','event reflected-shadow post-cash',
        'accepted own executed quantities/entry prices/delivery keys','fixed public center','public price support'],
      gold_excluded=['actual settlement prices','source G','source booked reward','conditional moments',
        'ledger-derived credits/prefixes','future market rows','additional policy observations'],
      mandatory_baselines=['Real-only cash inverse + SAME grouped-delivery own-contract price support',
        'Direct source settlement ledger at the native immediate release',
        'Own-contract arithmetic using paid settlement price (full-information reference)',
        'Same arithmetic at the existing public reader latency, delivery+7 vs source settlement delivery+6',
        'Local TIR(C=0) wherever exact known-K=0 interface applies; general known K is separately labeled'],
      reports=['All due events and unsupported reasons, including both-censored intervals',
        'Real-only/paired widths and signed midpoint errors; deterministic true booked containment',
        'Strict paired interval improvement count vs equally informed real-only accounting',
        'No exact IEEE singleton claim from a conservative interval',
        'All call budgets, complete/no-truncation episodes and zero fitting'],
      interpretation_limits=['Retrospective reused2022 development context selected after earlier outcomes were known',
        'Original actor observations/training already contain ledger information',
        'Public engine exposes G/S: this is an artificial restricted auditor interface',
        'Event snapshot is after cost commit; no whole-step unknown-suffix invariance is asserted',
        'No policy selection or learned-policy improvement and no new independent evaluation',
        'Full native execution remains on hold pending cheaper saved-cohort audit']))
    print(json.dumps(dict(status='PREPARED_HELD_NO_NATIVE_MODEL_MARKET',protocol_sha256=sha(PROTOCOL))))


def run():
    p=read(PROTOCOL);a=read(AUTH)
    if a.get('status')!='ROOT_GO_POLICY_EVENT_V1' or a.get('protocol_sha256')!=sha(PROTOCOL):
        raise PermissionError('A separate exact root GO for this policy event protocol is required')
    for rel,h in p['source_metadata_sha256'].items():
        if sha(PAPER/rel)!=h:raise AssertionError('Frozen source/metadata changed: '+rel)
    # Every external import/model/market operation is below the NEW root GO.
    out=HERE/'runs/policy_event_v1';out.mkdir(parents=True,exist_ok=False)
    save(out/'STARTED.json',dict(protocol_sha256=sha(PROTOCOL),authorization=a))
    env=None;rng=None
    try:
        sys.path.insert(0,str(BASE))
        spec=importlib.util.spec_from_file_location('_audit_existing_ppo_driver',BASE/'train_generic_ppo.py')
        driver=importlib.util.module_from_spec(spec);sys.modules[spec.name]=driver;spec.loader.exec_module(driver)
        old=read(OLD);driver.verify(old,all_inputs=False)
        for rel,h in old['artifact_sha256'].items():
            if rel.endswith('/scenario_015.csv') and sha(PAPER/rel)!=h:
                raise AssertionError('Pinned selected2022 market input changed: '+rel)
        import numpy as np,torch
        from stable_baselines3 import PPO
        from native_rng_stream import NativeCPUStream
        from policy_event_observer_v1 import SettlementObserver
        modelpath=BASE/'runs/train_v1'/p['selected_model']['relative_directory']/'FINAL_MODEL.zip'
        if sha(modelpath)!=p['selected_model']['model_sha256']:raise AssertionError('Pinned policy weights changed')
        records=[];observer=None;released=None
        for mode in ('reference','observed'):
            native=driver.load_wrapper(old);model=PPO.load(modelpath,device='cpu')
            if model.num_timesteps!=4704 or model.observation_space.shape!=(91,):raise AssertionError('Pinned PPO task metadata')
            driver.seed_all(p['evaluation_seed'],torch,np)
            env=native.NativeDecisionEnv(old['validation_contexts'],arm=p['arm'],output_dir=out/mode)
            rng=NativeCPUStream(p['native_rng_seed'])
            obs,_=rng.reset_call(env.reset,seed=7,options={'context_index':p['context_index']})
            if mode=='observed':observer=SettlementObserver(env.backend.env,center=p['public_reflection_center'])
            trace=[];done=False
            for k in range(168):
                before=np.asarray(obs).copy();action,_=model.predict(obs,deterministic=True)
                if observer is None:obs,reward,done,truncated,info=rng.call(env.step,action,operation='step')
                else:
                    with observer:obs,reward,done,truncated,info=rng.call(env.step,action,operation='step')
                if truncated or done!=(k==167):raise AssertionError('No partial episode or early terminal')
                trace.append(dict(observation=before,action=action,reward=reward,info=info))
            if env.lifetime_native_steps!=1008 or env.lifetime_macro_steps!=168 or not done:
                raise AssertionError('Exact full episode budget')
            row=dict(mode=mode,native_rows_digest=digest(env.backend.episode_rows),
                actor_trace_digest=digest(trace),result=env.completed_episodes[-1],rng=rng.record())
            save(out/(mode+'_TRACE_DIGEST.json'),row);records.append(row)
            if observer is not None:
                released=observer.released_price_baseline(env.backend.reader,now=1007,max_queries=166)
            rng.call(env.close,operation='close');env=None
        if any(records[0][key]!=records[1][key] for key in ('native_rows_digest','actor_trace_digest','result','rng')):
            raise AssertionError('Instrumented/reference full-path parity failed')
        if len(observer.steps)!=1008:raise AssertionError('Observer missed source steps')
        save(out/'RESTRICTED_API_ONLY.json',[r['api'] for r in observer.events if 'api' in r])
        save(out/'GOLD_AND_EVALUATION.json',dict(events=observer.events,steps=observer.steps))
        save(out/'RELEASED_PRICE_BASELINE.json',released)
        save(out/'RESULTS.json',dict(status='COMPLETE_FROZEN_POLICY_EVENT_FEASIBILITY_ONLY',
            full_path_parity=True,protocol_sha256=sha(PROTOCOL),audit=observer.summary(),
            budgets=p['budgets'],new_training=0,learner_updates=0,
            interpretation=p['interpretation_limits']))
    except BaseException as error:
        save(out/'FAILED.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise
    finally:
        if env is not None:
            if rng is None:env.close()
            else:rng.call(env.close,operation='close')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true');group.add_argument('--run',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else run()
