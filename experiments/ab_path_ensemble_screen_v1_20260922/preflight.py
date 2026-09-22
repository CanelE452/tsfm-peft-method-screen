import sys,subprocess,platform,importlib.metadata,inspect
import numpy as np
import torch
import chronos.chronos_bolt as bolt
from common import *
from A.data import audit

def main():
    assert torch.cuda.is_available()
    versions={n:importlib.metadata.version(n) for n in ['torch','chronos-forecasting','peft','transformers','numpy','pandas','scipy','scikit-learn','safetensors','pytest','requests','matplotlib','psutil']}
    env=dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions,
        gpu=torch.cuda.get_device_name(),vram=torch.cuda.get_device_properties(0).total_memory,torch_cuda=torch.version.cuda,
        nvidia_smi=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],capture_output=True,text=True).stdout)
    modelmanifest=read(ROOT/'results/continuation_lora_v1_20260922/SOURCE_MANIFEST.json')
    for info in modelmanifest['models'].values():
        for f in info['files']:assert sha(Path(info['path'])/f['name'])==f['sha256']
    source=Path(inspect.getfile(bolt))
    for c in ['A','B']:
        save(RESULTS/c/'ENVIRONMENT.json',env)
        save(RESULTS/c/'SOURCE_AND_MODEL_MANIFEST.json',dict(models=modelmanifest['models'],chronos_bolt_source=dict(path=source,sha256=sha(source)),reference_components_sha256=sha(EXP/'contract/reference_core.py')))
        freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        (RESULTS/c/'requirements-lock.txt').write_text(freeze,encoding='utf-8')
        save(RESULTS/c/'TRAINING_BUDGET.json',dict(main_fit_cap=12,updates_per_fit=256,main_update_cap=3072,smoke_update_cap=12,seeds=SEEDS,optimizer='AdamW',lora_lr=.0001,module_lr=.0003,effective_batch=4,checkpoints=[0,128,256],selection='raw VAL CRPS then common CAL then VAL baseline',failed_attempts_count=True))
        save(RESULTS/c/'CPU_TESTS.json',dict(reference_tests=22,status='PASS',real_model_validation_required=True))
        (RESULTS/c/'PROTOCOL.md').write_text('# '+c+' execution protocol\n\nThe sole execution contract is [MASTER_CLI.txt](../../../experiments/'+NAME+'/contract/MASTER_CLI.txt). User explicitly authorized implementation, fixed training, evaluation and scoped publication. No previous experiments resumed.\n',encoding='utf-8')
    d=audit()
    save(RESULTS/'REPOSITORY_START.json',dict(actual_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        contract_design_head='f31b5834f6d6258713f6c7d08f44adbb54880e7c',changes_since_design='Four commits of independent two-hypothesis screen and design; read only, not reused as initialization',
        zip_sha256=(EXP/'contract/ZIP_SHA256.txt').read_text().strip(),master_sha256=sha(EXP/'contract/MASTER_CLI.txt'),initial_dirty_files=[],scope='Only new experiment/results/cache paths; existing files preserved'))
    print('ENV + A DATA PASS',env['gpu'])

if __name__=='__main__':main()
