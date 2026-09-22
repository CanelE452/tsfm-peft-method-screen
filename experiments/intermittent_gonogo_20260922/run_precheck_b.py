"""사전확인 (b): G3 배선 smoke (10 step). seed / 체크포인트 / 시간·VRAM 외삽."""
import sys, time, json, hashlib
from pathlib import Path
import numpy as np, torch

M5=Path("/home/minjae/Documents/github/m5dataset"); sys.path.insert(0,str(M5))
from common.data_split import get_split
OUT=Path(__file__).resolve().parent
SPLIT_END=1885; H=28; BS=32; STEPS=10; NSERIES=2000
def log(m): print(m, flush=True)

log("[STEP 1] 학습 데이터 (d_1885 이하만 — 누수 차단)")
ids,train,test=get_split()
elig=(train[:,:SPLIT_END]>0).sum(1)>=20
sel=np.where(elig)[0][:NSERIES]
inputs=[torch.tensor(train[i,:SPLIT_END],dtype=torch.float32) for i in sel]
log(f"  계열 {len(inputs)}개, 각 길이 {inputs[0].shape[0]} (= d_1..d_{SPLIT_END})")

from chronos import Chronos2Pipeline
base=Chronos2Pipeline.from_pretrained("amazon/chronos-2",device_map="cuda",dtype=torch.bfloat16)
log(f"  모델 기본 context_length = {base.model.chronos_config.context_length}")

def adapter_hash(d):
    f=list(Path(d).rglob("adapter_model.safetensors"))
    if not f: return None,None
    h=hashlib.sha256(f[0].read_bytes()).hexdigest()[:16]
    from safetensors.torch import load_file
    sd=load_file(str(f[0])); keys=sorted(sd.keys())
    a=[k for k in keys if "lora_A" in k]
    return h,(len(keys),a[:3])

results={}
for seed in [0,1]:
    log(f"\n[STEP 2] fit seed={seed}  (10 step, batch {BS}, lora 기본설정)")
    od=OUT/f"fit_seed{seed}"
    torch.manual_seed(seed); np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    t0=time.time()
    ft=base.fit(inputs, prediction_length=H, finetune_mode="lora",
                learning_rate=1e-4, num_steps=STEPS, batch_size=BS,
                output_dir=od, remove_printer_callback=True,
                seed=seed, save_strategy="steps", save_steps=5, save_total_limit=5)
    dt=time.time()-t0; peak=torch.cuda.max_memory_allocated()/1e9
    ckpts=sorted([p.name for p in od.glob("checkpoint-*")])
    h,info=adapter_hash(od)
    log(f"  소요 {dt:.1f}s  peak {peak:.2f}GB")
    log(f"  중간 체크포인트: {ckpts if ckpts else '없음'}")
    log(f"  adapter sha256[:16] = {h}   (텐서 {info[0] if info else '?'}개)")
    if info: log(f"  lora_A 예시: {info[1]}")
    results[f"seed{seed}"]={"sec":dt,"peak_gb":peak,"ckpts":ckpts,"adapter_hash":h,
                            "n_tensors":info[0] if info else None}
    del ft; torch.cuda.empty_cache()

log("\n[STEP 3] 판정")
h0,h1=results["seed0"]["adapter_hash"],results["seed1"]["adapter_hash"]
seed_ok = (h0 is not None and h0!=h1)
ck_ok   = len(results["seed0"]["ckpts"])>0
per_step=np.mean([results[f"seed{s}"]["sec"] for s in [0,1]])/STEPS
g3_hours=per_step*1000*16/3600
log(f"  seed 배선 유효(해시 다름) : {'YES' if seed_ok else 'NO'}  ({h0} vs {h1})")
log(f"  중간 체크포인트 저장       : {'YES' if ck_ok else 'NO'}")
log(f"  step당 {per_step:.2f}s -> G3 16 fits x 1000 step = {g3_hours:.1f}시간  (지시문 예산 3시간)")
log(f"  peak VRAM {max(results[f'seed{s}']['peak_gb'] for s in [0,1]):.2f} GB / 10.0 GB (RTX 3080)")
json.dump({"results":results,"seed_wiring_ok":seed_ok,"checkpoint_ok":ck_ok,
           "sec_per_step":per_step,"g3_projected_hours":g3_hours,
           "batch_size":BS,"n_series":len(inputs)}, open(OUT/"precheck_b.json","w"), indent=2)
log(f"[DONE] {OUT/'precheck_b.json'}")
