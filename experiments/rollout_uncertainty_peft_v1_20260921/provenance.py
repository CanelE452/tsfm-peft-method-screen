import importlib.metadata as md
import inspect
import platform
import shutil
import sys
import urllib.request
from common import *

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return {'url': url, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size}

def prepare():
    import torch, chronos, peft, transformers, psutil
    from huggingface_hub import HfApi, snapshot_download
    src = CACHE / 'upstream'
    prior = 'b60bfd92ff836525773c71039a83ba2fe3d123bf'
    manifest = {'master_sha256': sha(EXP/'contract/MASTER_CLI.txt'), 'repo_initial_head': 'e84ef88579382574d9c82985c2c0c4672e262b21', 'initial_dirty_files': [], 'prior_revision': prior, 'sources': [], 'models': {}}
    for f in ['README.md', 'src/README.md', 'src/forecaster.py', 'src/metrics.py']:
        manifest['sources'].append(fetch(f'https://raw.githubusercontent.com/Coaster41/Beyond-Accuracy-TSFM-Calibration/{prior}/{f}', src/'prior'/f))
    upstream = __import__('json').loads(urllib.request.urlopen('https://api.github.com/repos/amazon-science/chronos-forecasting/commits/main').read())['sha']
    manifest['chronos_upstream_commit'] = upstream
    manifest['sources'].append(fetch(f'https://raw.githubusercontent.com/amazon-science/chronos-forecasting/{upstream}/src/chronos/chronos_bolt.py',src/'chronos_bolt.py'))
    import chronos.chronos_bolt as bolt
    manifest['installed_bolt_source'] = {'path': inspect.getfile(bolt), 'sha256': sha(inspect.getfile(bolt)), 'matches_current_upstream': sha(inspect.getfile(bolt)) == sha(src/'chronos_bolt.py')}
    for model_id in ['amazon/chronos-bolt-small', 'amazon/chronos-2']:
        info = HfApi().model_info(model_id)
        event('model_download', model=model_id, revision=info.sha)
        path = Path(snapshot_download(model_id, revision=info.sha, cache_dir=str(CACHE/'huggingface'), allow_patterns=['*.json','*.safetensors']))
        manifest['models'][model_id] = {'revision': info.sha, 'path': str(path), 'files': [{'name':f.name,'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(path.glob('*')) if f.is_file()], 'config':read_json(path/'config.json')}
        save_json(RESULTS/'SOURCE_AND_MODEL_MANIFEST.json',manifest)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('BLOCKED_RESOURCE: CUDA unavailable')
    x = torch.randn(128,128,device='cuda')
    assert torch.isfinite(x @ x).all()
    env = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(), 'packages':{n:md.version(n) for n in ['torch','chronos-forecasting','peft','transformers','huggingface_hub','numpy','pandas','scipy','safetensors','psutil','pyyaml','pytest','matplotlib','requests']},'torch_cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'vram':torch.cuda.get_device_properties(0).total_memory,'nvidia_smi':command(['nvidia-smi']),'pip_check':command([sys.executable,'-m','pip','check']),'ram_bytes':psutil.virtual_memory().total,'disk_free_bytes':shutil.disk_usage(ROOT).free,'environment_policy':'Existing isolated Python 3.11 reused read-only; no old learned weights or experiment modules imported','gpu_tensor_check':True}
    save_json(RESULTS/'ENVIRONMENT.json',env)
    lock = command([sys.executable,'-m','pip','freeze'])
    assert lock['returncode']==0 and env['pip_check']['returncode']==0
    (RESULTS/'requirements-lock.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cu128\n'+lock['stdout'],encoding='utf-8')
    event('provenance_complete')

if __name__=='__main__': prepare()
