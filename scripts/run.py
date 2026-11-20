#!/usr/bin/env python3
"""Launch LIBERO training and evaluation."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')

def paths(config):
    values = json.loads((ROOT / 'configs/paths.example.json').read_text())
    selected = Path(config) if config else ROOT / 'configs/paths.local.json'
    if config and not selected.is_absolute():
        selected = ROOT / selected
    if selected.exists():
        override = json.loads(selected.read_text())
        unknown = set(override) - set(values)
        if unknown:
            raise ValueError('Unknown path keys: ' + ', '.join(sorted(unknown)))
        values.update(override)
    elif config:
        raise FileNotFoundError(selected)
    for key, value in values.items():
        if key == 'cudnn_preload' and value in ('auto', ''):
            continue
        value = Path(os.path.expandvars(os.path.expanduser(value)))

        values[key] = os.path.abspath(value if value.is_absolute() else ROOT / value)
    return values

def environment(p, dry=False):
    env = os.environ.copy()
    env.pop('LEROBOT_HOME', None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        OPENPI_DATA_HOME=p['openpi_data_home'], HF_HOME=p['hf_home'],
        HF_LEROBOT_HOME=p['lerobot_home'], FPF_ASSETS_DIR=p['assets_dir'],
        SERVER_PYTHON=p['server_python'], CLIENT_PYTHON=p['client_python'],
        OUTPUT_ROOT=p['output_dir'], LIBERO_CONFIG_PATH=str(Path(p['output_dir'])/'libero_config'),
        PYTHONPATH=os.pathsep.join(str(ROOT/x) for x in (
            'src','vendor/openpi/src','vendor/openpi/packages/openpi-client/src')))
    preload = p['cudnn_preload']
    if preload == 'auto' and not dry:
        purelib = subprocess.check_output([p['server_python'],'-c',
            'import sysconfig; print(sysconfig.get_paths()["purelib"])'],text=True).strip()
        candidate = Path(purelib)/'nvidia/cudnn/lib/libcudnn.so.9'
        if not candidate.is_file():
            raise FileNotFoundError('cuDNN preload not found; configure cudnn_preload: '+str(candidate))
        preload = str(candidate)
    env['FPF_CUDNN_PRELOAD'] = preload
    if not dry:
        config = Path(env['LIBERO_CONFIG_PATH'])
        config.mkdir(parents=True, exist_ok=True)
        lib = ROOT/'vendor/libero/libero/libero'
        (config/'config.yaml').write_text(json.dumps({
            'benchmark_root':str(lib), 'bddl_files':str(lib/'bddl_files'),
            'init_states':str(lib/'init_files'), 'assets':str(lib/'assets'),
            'datasets':p['expert_data']},indent=2)+'\n')
    return env

def train_command(p, exp):
    args = json.loads((ROOT/'configs/train_args.json').read_text())
    return [p['server_python'],str(ROOT/'scripts/libero/train.py'),
        '--exp_name',exp,'--checkpoint_base_dir',str(Path(p['output_dir'])/'checkpoints'),
        '--norm_stats_dir',p['norm_stats_dir']] + [a.format(**p) for a in args]

def eval_commands(p, a):
    result = Path(p['output_dir'])/'eval_results'/a.run_name
    commands=[]
    for i in range(8):
        start=(i%2)*5
        commands.append(['bash',str(ROOT/'scripts/libero/eval_shard.sh'),
            p['checkpoint'], str(a.step), a.run_name, str(result), SUITES[i//2],
            str(start), str(start+5), str(a.trials), a.gpus[i], str(a.port_base+i), '8'])
    return commands, result

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--paths',help='JSON path overrides, relative to release root')
    sub=ap.add_subparsers(dest='command',required=True)
    train=sub.add_parser('train')
    train.add_argument('--exp-name',default='libero_fresh_seed42')
    train.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    train.add_argument('--dry-run',action='store_true')
    ev=sub.add_parser('eval')
    ev.add_argument('--checkpoint',help='Checkpoint step folder, containing params/')
    ev.add_argument('--step',type=int,default=10000)
    ev.add_argument('--seed',type=int,choices=(7,17,27,37),default=7)
    ev.add_argument('--mode',choices=('random','argmax'),default='argmax')
    ev.add_argument('--trials',type=int,default=20)
    ev.add_argument('--gpus',default='0,1,2,3,4,5,6,7')
    ev.add_argument('--port-base',type=int,default=9410)
    ev.add_argument('--run-name',default=None)
    ev.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    a.gpus=a.gpus.split(',')
    if len(a.gpus)!=8 or len(set(a.gpus))!=8 or not all(g.isdigit() for g in a.gpus):
        ap.error('Training and evaluation require eight distinct numeric GPU IDs.')
    p=paths(a.paths)
    if a.command=='eval' and a.checkpoint:
        ckpt=Path(a.checkpoint).expanduser()
        p['checkpoint']=str(ckpt if ckpt.is_absolute() else ROOT/ckpt)
    name=a.exp_name if a.command=='train' else (a.run_name or 'step{}_seed{}_{}'.format(a.step,a.seed,a.mode))
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',name) or name in ('.','..'):
        ap.error('Use a simple run name without path separators.')
    if a.command=='eval' and (a.trials<1 or not 1<=a.port_base<=65528):
        ap.error('trials must be positive and ports must fit 1..65535.')
    env=environment(p,a.dry_run)
    if a.command=='train':
        cmd=train_command(p,name)
        env.update(CUDA_VISIBLE_DEVICES=','.join(a.gpus),
            XLA_PYTHON_CLIENT_PREALLOCATE='false',XLA_PYTHON_CLIENT_MEM_FRACTION='0.90',
            LD_PRELOAD=env['FPF_CUDNN_PRELOAD'])
        print(shlex.join(cmd),flush=True)
        if a.dry_run: return 0
        for key in ('rollout_data','expert_data','init_checkpoint','norm_stats_dir'):
            if not Path(p[key]).is_dir(): raise FileNotFoundError(p[key])
        if (Path(p['output_dir'])/'checkpoints'/name).exists():
            raise FileExistsError('Use a fresh experiment name; params-only resume is not supported.')
        for key, rel in {'JAX_COMPILATION_CACHE_DIR':'cache/jax', 'TMPDIR':'tmp',
                         'WANDB_DIR':'wandb_runs', 'WANDB_CACHE_DIR':'cache/wandb'}.items():
            directory=Path(p['output_dir'])/rel
            directory.mkdir(parents=True, exist_ok=True)
            env[key]=str(directory)
        return subprocess.call(cmd,cwd=ROOT,env=env)
    a.run_name=name
    commands,result=eval_commands(p,a)
    env.update(EVAL_SEED=str(a.seed),POLICY_SEED=str(310000+a.seed),
        NUM_CANDIDATES='5',NUM_FLOW_STEPS='10',SCORE_MODE='shared_raw',
        SELECTION_MODE=a.mode,TACTILE_SUBGEOM='1',MEM_FRACTION='0.82',
        MUJOCO_GL='egl',SERVER_START_TIMEOUT_SECONDS='900',
        JAX_CACHE_DIR=str(Path(p['output_dir'])/'cache/jax'),
        EVAL_TMPDIR=str(Path(p['output_dir'])/'tmp'),
        ROLLOUTS_DIR_ROOT=str(result/'rollouts'))
    print(json.dumps({k:env[k] for k in ('EVAL_SEED','POLICY_SEED','NUM_CANDIDATES',
        'NUM_FLOW_STEPS','SCORE_MODE','SELECTION_MODE','TACTILE_SUBGEOM')},indent=2))
    for cmd in commands: print(shlex.join(cmd),flush=True)
    if a.dry_run: return 0
    if not (Path(p['checkpoint'])/'params').is_dir(): raise FileNotFoundError(p['checkpoint'])
    if result.exists(): raise FileExistsError('Use a fresh --run-name to avoid mixing results: '+str(result))
    for command in ('bash','jq','ss','flock'):
        import shutil
        if not shutil.which(command): raise RuntimeError('Missing executable: '+command)
    result.mkdir(parents=True)
    children=[]
    try:
        for cmd in commands:
            children.append(subprocess.Popen(cmd,cwd=ROOT,env=env,start_new_session=True))
            time.sleep(3)
        codes=[child.wait() for child in children]
        if any(codes): raise RuntimeError('Evaluation shard failed: '+str(codes))
    finally:
        for child in children:
            if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
        for child in children:
            if child.poll() is None:
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
    files=sorted((result/'shards'/name).glob('*.json'))
    if len(files)!=8: raise RuntimeError('Expected exactly eight result shards.')
    for f in files:
        d=json.loads(f.read_text())
        if d['rc']!=0 or d['total_episodes']!=5*a.trials or not 0<=d['total_successes']<=d['total_episodes']:
            raise RuntimeError('Incomplete shard: '+str(f))
    subprocess.run([p['server_python'],str(ROOT/'scripts/libero/merge_shards.py'),
        '--shards-dir',str(result/'shards'/name),'--results-dir',str(result),
        '--ckpt-step',str(a.step),'--run-label',name,'--expected','8',
        '--ckpt-dir',p['checkpoint']],cwd=ROOT,env=env,check=True)
    print('Validated {} episodes: {}'.format(40*a.trials,result/'ckpt_{}.json'.format(a.step)))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
