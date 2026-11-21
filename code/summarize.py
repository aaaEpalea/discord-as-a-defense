# Author: Aliyu Agboola Alege
# Developed with the support of Claude Sonnet 5.5 (Anthropic) as a pair-programming assistant.
# License: MIT (see ../LICENSE)
import pandas as pd, numpy as np, sys
pd.set_option('display.width',250); pd.set_option('display.max_rows',300); pd.set_option('display.max_columns',30)
f=sys.argv[1]
d=pd.read_csv(f)
def ms(g,c):
    return g.groupby('model')[c].agg(lambda v: f"{np.mean(v)*100:.1f}±{np.std(v)*100:.1f}")
s=d[d.kind=='sweep']
for dr in ['w','v','both_indep','both_coord']:
    t=s[(s.direction==dr)]
    out={}
    for r in [0.0,0.3,0.45]:
        if r==0 and dr!='w': continue
        g=t[t.rate==r]
        out[f'overall@{r}']=ms(g,'overall')
        if r>0: out[f'corrupted@{r}']=ms(g,'corrupted'); out[f'fire_clean@{r}']=ms(g,'fire_clean'); out[f'fire_corr@{r}']=ms(g,'fire_corrupt')
    print('== direction',dr); print(pd.DataFrame(out).reindex(["PoE","SheafAlign","DGSA-hard(val30)","DGSA-hard(clean95)","DGSA-soft","Entropy","V-only","W-only"]))
a=d[d.kind=='attack']
print('== attacks (corrupted-subset acc @30%)'); print(a.groupby(['model','attack']).corrupted.agg(lambda v:f"{np.mean(v)*100:.1f}±{np.std(v)*100:.1f}"))
t=d[d.kind=='detect']
print('== detection @30% (clean-95th-pct threshold)')
for dr in ['w','v','both_indep','both_coord']:
    g=t[t.direction==dr].groupby('model')[['auroc','recall','false_rej','coverage','sel_acc']].mean()
    print(dr);print(g.round(3))
al=d[d.kind=='soft_alpha']; print('soft alpha mean/clean/corr', al[['overall','clean','corrupted']].mean().round(3).to_dict(), 'midpoint,temp', al[['fire_clean','fire_corrupt']].mean().round(2).to_dict())
