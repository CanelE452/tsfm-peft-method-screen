"""Visualize post-hoc input-only gate audit; no inference."""
from .figures import *
def main():
    f=pd.read_csv(OUT/'GATE_IDENTIFIABILITY.csv');f=f[f.kind=='standard'];fig,axes=plt.subplots(1,2,figsize=(12,6))
    for ax,col,title in zip(axes,['equal_context_fraction','C3_all_one_context_fraction'],['C3 and RECENCY: identical masks (%)','C3: all-one masks (%)']):
        z=f.pivot(index='condition',columns='panel',values=col).reindex(index=STATES,columns=PANELS)*100
        ax.imshow(z,cmap='Blues',vmin=0,vmax=100,aspect='auto');ax.set_xticks(range(5),[LABELS[p] for p in PANELS],rotation=65);ax.set_yticks(range(10),STATES);ax.set_title(title)
        for i in range(10):
            for j in range(5):ax.text(j,i,f'{z.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if z.iloc[i,j]>65 else 'black')
    fig.suptitle('Post-hoc input-only audit: exact equality across all 32 patch gates');plt.tight_layout()
    for suffix in ['png','pdf','svg']:plt.savefig(PAPER/'figures'/('F9_gate_identifiability.'+suffix),dpi=180,bbox_inches='tight')
    svg=PAPER/"figures/F9_gate_identifiability.svg";svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    plt.close();m=read(PAPER/'FIGURE_MANIFEST.json');m['figures']=sorted(p.name for p in (PAPER/'figures').glob('*.png'));m['tables']=sorted(p.name for p in (PAPER/'tables').glob('*.csv'));m['mask_audit']='post-hoc CPU descriptive audit; no new prediction or tuning';save(PAPER/'FIGURE_MANIFEST.json',m)
if __name__=='__main__':main()
