from common import *
import importlib.metadata as md
import subprocess
import requests
import torch
import bitsandbytes as bnb
from huggingface_hub import HfApi, snapshot_download


def main():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    x = torch.randn(8, 64, device='cuda', dtype=torch.bfloat16)
    layer = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.bfloat16,
                            compress_statistics=True, quant_type='nf4').to('cuda')
    y = layer(x)
    assert torch.isfinite(y).all() and layer.weight.dtype == torch.uint8
    env = {'python': sys.version, 'executable': sys.executable, 'torch_cuda': torch.version.cuda,
           'gpu': torch.cuda.get_device_name(), 'capability': torch.cuda.get_device_capability(),
           'bf16_actual_forward': True, 'nf4_actual_forward': True,
           'packed_dtype': str(layer.weight.dtype), 'nested': layer.weight.quant_state.nested,
           'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
           'packages': {p: md.version(p) for p in ['torch','chronos-forecasting','peft','transformers','bitsandbytes','numpy','safetensors','huggingface-hub']},
           'bnb_isolated_overlay': str(CACHE / 'python_packages')}
    write(RESULTS / 'ENVIRONMENT.json', env)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (RESULTS / 'requirements-lock.txt').write_text(freeze + '\n# task-local overlay\nbitsandbytes==0.50.2\n', encoding='utf-8')
    pins = {}
    for size in ['small', 'base']:
        model_id = 'amazon/chronos-bolt-' + size
        revision = HfApi().model_info(model_id).sha
        path = snapshot_download(model_id, revision=revision, allow_patterns=['config.json','model.safetensors'],
                                 cache_dir=str(CACHE / 'hf'))
        pins[size] = {'model_id': model_id, 'revision': revision, 'path': path,
                      'files': {p.name: {'sha256': sha(p), 'bytes': p.stat().st_size} for p in Path(path).iterdir() if p.is_file()}}
        event('model_downloaded', size=size, revision=revision)
    source_dir = CACHE / 'upstream'
    source_dir.mkdir(exist_ok=True)
    revision = 'bd7fc86a2e44d41f95b9b0421f27f5624dd37064'
    provenance = []
    for source in ['src/qera/fine_tuning/initialization.py','src/qera/statistic_profiler/scale.py','src/qera/approximate.py','LICENSE']:
        url = f'https://raw.githubusercontent.com/ChengZhang-98/QERA/{revision}/{source}'
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        path = source_dir / Path(source).name
        path.write_bytes(r.content)
        provenance.append({'url':url,'revision':revision,'sha256':sha(path),'bytes':len(r.content)})
    write(RESULTS / 'UPSTREAM.json', {'models': pins, 'qera': provenance,
          'contract_sha256': sha(EXP/'contract/MASTER_PLAN.txt'),
          'wheel': {p.name:sha(p) for p in (CACHE/'wheels').glob('*.whl')}})
    event('bootstrap_complete')


if __name__ == '__main__':
    main()
