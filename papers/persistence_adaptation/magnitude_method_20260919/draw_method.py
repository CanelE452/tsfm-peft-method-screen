"""Source-grounded MAG diagram. Generates figures only; no model inference."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Circle,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(15,7.1));ax.set_xlim(0,14.8);ax.set_ylim(0,6.9);ax.axis('off')
blue='#e8f0fa';orange='#fff0d7';purple='#eee8f6';gray='#f1f3f5';ink='#26374a'
def box(x,y,w,h,label,color=gray,size=10):
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.06,rounding_size=0.07',facecolor=color,edgecolor=ink,linewidth=1.1));ax.text(x,y,label,ha='center',va='center',fontsize=size,color=ink)
def arrow(points,dashed=False,color=ink):
 for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>' if i==len(points)-2 else '-',mutation_scale=11,lw=1.2,color=color,linestyle='--' if dashed else '-'))
def node(x,y,label):
 ax.add_patch(Circle((x,y),.16,edgecolor=ink,facecolor='white',lw=1.1));ax.text(x,y,label,ha='center',va='center',fontsize=14,color=ink)
ax.text(.15,6.66,'MAG_ONLY: unchanged observations, gated patch residuals',fontsize=17,weight='bold',color=ink)
ax.text(.15,6.25,'Fixed implementation; empirical evidence and novelty limitations are reported separately.',fontsize=10,color='#586779')
box(1,5.55,1.35,.85,'Observed x\n512 values')
box(3.5,5.55,2.55,.85,'B0 normalization\n+ patch embedding',blue)
node(5.1,5.55,'h');node(9.6,5.55,'+')
box(11.25,5.55,2.3,.85,'Frozen encoder / decoder\n+ output head / inverse norm',blue,9)
box(13.65,5.55,1.1,.85,'Forecast\n9 × 64',blue)
arrow([(1.735,5.55),(2.165,5.55)]);arrow([(4.835,5.55),(4.91,5.55)])
arrow([(5.29,5.55),(9.4,5.55)]);ax.text(7.25,5.73,'identity path',ha='center',fontsize=10,color='#586779')
arrow([(9.8,5.55),(10.04,5.55)]);arrow([(12.46,5.55),(13.02,5.55)])
box(7.2,4.33,3.3,.95,'Trainable residual: 512 → 8 → 512\nGELU, then cap(h) × tanh',orange)
arrow([(5.1,5.36),(5.1,4.33),(5.49,4.33)])
node(9.6,4.33,'×');arrow([(8.91,4.33),(9.4,4.33)]);arrow([(9.6,4.52),(9.6,5.35)])
box(3.8,3.1,4.0,1.03,'Fixed magnitude gate g(x)\nrobust center / scale over 512 values\n1 − extreme-value fraction in each patch',purple,10)
arrow([(1,5.06),(1,3.1),(1.74,3.1)]);arrow([(5.86,3.1),(9.6,3.1),(9.6,4.14)])
box(1.0,2.0,1.5,.60,'TRAIN sigma',gray,9);arrow([(1.81,2.0),(2.2,2.0),(2.2,2.52)])
ax.text(5.08,4.78,'32 patches\n512-d embeddings',ha='right',fontsize=8.5,color='#586779')
box(11.2,4.05,2.0,.95,'Training loss\nnormalized 2-pinball',gray,9.5)
box(13.65,3.25,1.1,.65,'Target y\ntrain only',gray,9)
arrow([(13.65,5.06),(13.65,4.22),(12.26,4.22)],True)
arrow([(13.65,3.635),(13.65,3.87),(12.26,3.87)],True)
ax.text(11.2,3.30,'Backpropagation through frozen B0;\nonly residual parameters are updated.',ha='center',va='top',fontsize=9,color='#586779')
ax.text(6.65,2.52,'cap(h): detached median patch RMS, floor 1e−6\nZero-initialized up projection: initial forecast = B0\nResidual parameters: 8,712; B0 stays frozen',ha='left',va='top',fontsize=9,color=ink)
ax.plot([.15,14.45],[1.39,1.39],color='#cbd2d9',lw=1)
ax.text(.15,1.11,'Controls at the same residual location',fontsize=11,weight='bold',color=ink)
ax.text(.15,.72,'PLAIN: g = 1      |      C3: persistence rule      |      TOKEN_GATE: sigmoid(Wg h + bg), +513 parameters',fontsize=10,color=ink)
ax.text(.15,.39,'TOKEN_GATE_ENTROPY: same learned gate + fixed normalized Bernoulli-entropy penalty (λ = 0.01).',fontsize=10,color=ink)
ax.text(.15,.04,'Gate inputs exclude future targets, clean inputs, fault masks and generator state. No clipping of the observed model input.',fontsize=9,color='#586779')
fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99)
for suffix in ['pdf','svg','png']:fig.savefig(OUT/f'method.{suffix}',dpi=180)
plt.close(fig)
p=OUT/'method.svg';p.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
paths=['experiments/c3_weakness_controls_20260918/model.py','experiments/additive_persistence_validation_v1_20260917/model.py','experiments/outlier_signal_peft_v1_20260917/model.py','experiments/learned_gate_comparison_20260919/model.py',str(Path(__file__).relative_to(ROOT))]
hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
manifest=dict(status='GENERATED_FROM_CURRENT_FROZEN_IMPLEMENTATION',source_hashes=hashes,artifact_hashes={f'method.{suffix}':hashlib.sha256((OUT/f'method.{suffix}').read_bytes()).hexdigest() for suffix in ['pdf','svg','png']},model_inferences=0,optimizer_updates=0,empirical_superiority_claimed=False,novelty_claimed=False)
(OUT/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('FIGURES_GENERATED')
