"""Method schematic only: contains no performance evidence or evaluation data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT=Path(__file__).resolve().parent
fig,ax=plt.subplots(figsize=(12,7));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,text,color='#e9eef4',fs=10):
 ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.009',linewidth=1,edgecolor='#53657a',facecolor=color))
 ax.text(x,y,text,ha='center',va='center',fontsize=fs)
def arrow(a,b,label=None):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color='#46566d',linewidth=1.25))
 if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.02,label,ha='center',fontsize=8)
ax.text(.02,.98,'Temporal response preservation for additional PEFT',fontsize=16,weight='bold',va='top')
ax.text(.02,.91,'Training: preserve the frozen model\'s response to a suffix-level probe',fontsize=11,color='#33465a')
box(.105,.65,.16,.16,'Observed past x\nProbe T(x)\nNo hidden state',fs=10)
box(.37,.79,.23,.12,'Frozen trained B0\nadapter disabled')
box(.37,.52,.23,.12,'Same B0 + adapter θ\nonly θ is trainable',color='#e1f0e7')
box(.64,.79,.22,.12,'Teacher response\nΔf₀ = f₀(Tx) − f₀(x)')
box(.64,.52,.22,.12,'Adapted response\nΔfθ = fθ(Tx) − fθ(x)',color='#e1f0e7')
box(.875,.66,.18,.14,'Response loss\nmean |Δfθ − Δf₀| / σ',color='#fff0da',fs=9)
box(.37,.29,.23,.11,'Original train label y\nUsed only for pinball')
box(.64,.29,.22,.11,'Forecast loss\n2-pinball(fθ(x), y)',color='#e1f0e7')
box(.875,.29,.18,.11,'Sum of losses\nλ = 1, fixed',color='#fff0da')
for a,b in [((.185,.69),(.246,.79)),((.185,.61),(.246,.52)),((.494,.79),(.521,.79)),((.494,.52),(.521,.52)),((.759,.79),(.776,.70)),((.759,.52),(.776,.62)),((.494,.29),(.521,.29)),((.48,.455),(.56,.355)),((.759,.29),(.776,.29)),((.875,.581),(.875,.354))]:arrow(a,b)
ax.text(.50,.385,'fθ(x)',fontsize=8,color='#53657a')
ax.text(.63,.881,'teacher outputs are detached',fontsize=8,color='#53657a',ha='center')
ax.axhline(.19,color='#c6cbd0',linewidth=1)
ax.text(.02,.16,'Inference: one ordinary adapter forward; no teacher probe',fontsize=11,color='#33465a')
box(.13,.065,.17,.075,'Observed input',fs=10);box(.45,.065,.30,.075,'Frozen B0 + trained θ',color='#e1f0e7',fs=10);box(.81,.065,.24,.075,'9-quantile forecast',fs=10)
arrow((.224,.065),(.291,.065));arrow((.609,.065),(.681,.065))
fig.text(.5,.006,'Method under evaluation. This schematic does not establish accuracy gains or novelty.',ha='center',fontsize=9,color='#555')
fig.tight_layout(rect=(0,.02,1,1))
for ext in ['svg','pdf','png']:fig.savefig(OUT/f'method.{ext}',dpi=170,bbox_inches='tight')
plt.close(fig)

p=OUT/"method.svg"
p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n")
