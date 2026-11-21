# Author: Aliyu Agboola Alege
# Developed with the support of Claude Sonnet 5.5 (Anthropic) as a pair-programming assistant.
# License: MIT (see ../LICENSE)
import pandas as pd, numpy as np, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
R3=R2=os.path.join(HERE,"..","results")
OUT=os.environ.get("OUT_DIR",os.path.join(HERE,"..","tables_and_figures"))   # tables/ and figures/ are written here
OT=os.path.join(OUT,"tables"); OF=os.path.join(OUT,"figures")
os.makedirs(OT,exist_ok=True); os.makedirs(OF,exist_ok=True)
SET=[("digits_word","paper","digits-word"),("mnist_views","paper","mnist-views (2k)"),("halves","paper","halves (2k)"),("mnist_views","full","mnist-views (full)")]
def L2(ds,sc):
    for d in (R3,R2):
        f=f"{d}/v2_{ds}_{sc}.csv"
        if os.path.exists(f): return pd.read_csv(f)
def L3(ds): 
    f=f"{R3}/v3_{ds}_paper.csv"; return pd.read_csv(f) if os.path.exists(f) else None
def ms(v,p=1): v=np.asarray(v,float); v=v[~np.isnan(v)]; return f"{np.mean(v)*100:.{p}f} ({np.std(v)*100:.{p}f})" if len(v) else "--"
def tab(path,header,rows,colspec):
    open(path,"w").write("\\small\n\\begin{tabular}{"+colspec+"}\n\\toprule\n"+header+" \\\\\n\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")
def blocks(rows_by_set):
    out=[]
    for k,(lab,rows) in enumerate(rows_by_set):
        for i,r in enumerate(rows):
            out.append((f"\\multirow{{{len(rows)}}}{{*}}{{{lab}}}" if i==0 else "")+" & "+r)
        if k<len(rows_by_set)-1: out[-1]+="\n\\midrule"
    return out
core=["V-only","PoE","SheafAlign","DGSA-hard(val30)","DGSA-hard(clean95)","DGSA-soft"]
nseeds={}
# ---------------- detect
rs=[]
for ds,sc,lab in SET:
    d=L2(ds,sc); 
    if d is None: continue
    nseeds[lab]=d.seed.nunique(); dt=d[d.kind=="detect"]; rows=[]
    for m in ["discord","PoE-variance","PoE-entropy","Sheaf-entropy"]:
        v=[dt[(dt.direction==dr)&(dt.model==m)].auroc.mean() for dr in ["w","v","both_indep","both_coord"]]
        rows.append(f"{m} & "+" & ".join(f"{x:.2f}" for x in v)+" \\\\")
    rs.append((lab,rows))
tab(f"{OT}/detect.tex","Setting & Detector & $w$ & $v$ & both-ind. & both-coord.",blocks(rs),"llcccc")
# ---------------- accuracy (core) 
rs=[]
for ds,sc,lab in SET:
    d=L2(ds,sc)
    if d is None: continue
    s=d[(d.kind=="sweep")&(d.direction=="w")]; rows=[]
    for m in core:
        g=s[s.model==m]; rows.append(f"{m} & {ms(g[g.rate==0].overall)} & {ms(g[g.rate==0.3].overall)} & {ms(g[g.rate==0.3].corrupted)} \\\\")
    rs.append((lab,rows))
tab(f"{OT}/accuracy.tex","Setting & Model & overall @0\\% & overall @30\\% & corrupted subset @30\\%",blocks(rs),"llccc")
# ---------------- gate
rows=[]
for ds,sc,lab in SET:
    d=L2(ds,sc)
    if d is None: continue
    s=d[(d.kind=="sweep")&(d.direction=="w")&(d.rate==0.3)]
    for m in ["DGSA-hard(val30)","DGSA-hard(clean95)"]:
        g=s[s.model==m]; rows.append(f"{lab} & {m} & {ms(g.fire_clean)} & {ms(g.fire_corrupt)} \\\\")
    sa=d[d.kind=="soft_alpha"]; rows.append(f"{lab} & DGSA-soft (mean $\\alpha$) & {sa.clean.mean():.2f} & {sa.corrupted.mean():.2f} \\\\")
    d3=L3(ds)
    if d3 is not None:
        for m in ["DGSA-soft-init","DGSA-soft-init+corr"]:
            sa=d3[(d3.kind=="soft_alpha")&(d3.model==m)]; rows.append(f"{lab} & {m} (mean $\\alpha$) & {sa.clean.mean():.2f} & {sa.corrupted.mean():.2f} \\\\")
tab(f"{OT}/gate.tex","Setting & Gate & clean & corrupted",rows,"llcc")
# ---------------- direction (core subset)
rs=[]
for ds,sc,lab in SET:
    d=L2(ds,sc)
    if d is None: continue
    s=d[(d.kind=="sweep")&(d.rate==0.3)]; rows=[]
    for m in ["V-only","PoE","SheafAlign","DGSA-hard(clean95)","DGSA-soft"]:
        rows.append(f"{m} & "+" & ".join(f"{np.mean(s[(s.model==m)&(s.direction==dr)].overall)*100:.1f}" for dr in ["w","v","both_indep","both_coord"])+" \\\\")
    rs.append((lab,rows))
tab(f"{OT}/direction.tex","Setting & Model & $w$ & $v$ & both-ind. & both-coord.",blocks(rs),"llcccc")
# ---------------- attacks
pri=["worst-of-9","joint(worst-of-lams)","task-only(lam=0)","any"]
def strongest(d,m):
    if d is None: return "--"
    g=d[(d.kind=="attack")&(d.model==m)]
    for p in pri:
        h=g[g.attack==p]
        if len(h): return ms(h.corrupted)
    return "--"
cols=[("digits_word","paper"),("digits_word","paper_adv"),("mnist_views","paper"),("mnist_views","paper_adv"),("halves","paper"),("halves","paper_adv"),("mnist_views","full"),("mnist_views","full_adv")]
rows=[]
for m in core:
    rows.append(f"{m} & "+" & ".join(strongest(L2(ds,sc),m) for ds,sc in cols)+" \\\\")
tab(f"{OT}/attacks.tex","Model & dw & dw adv. & mv2k & mv2k adv. & hv & hv adv. & mvF & mvF adv.",rows,"lcccccccc")
# ---------------- baselines table (paper-scale settings)
new=["DGSA-soft-init","DGSA-soft-init+corr","PDF","Kim-gated(clean)","Kim-gated(aug)","OOO(Yang-style)"]
rs=[]
for ds,sc,lab in SET[:3]:
    d2=L2(ds,sc); d3=L3(ds)
    if d2 is None or d3 is None: continue
    rows=[]
    for m in ["V-only","PoE","DGSA-soft"]+new:
        src=d2 if m in core else d3
        s=src[src.kind=="sweep"]
        o0=s[(s.model==m)&(s.direction=="w")&(s.rate==0)].overall
        ow=s[(s.model==m)&(s.direction=="w")&(s.rate==0.3)].overall
        ov=s[(s.model==m)&(s.direction=="v")&(s.rate==0.3)].overall
        rows.append(f"{m} & {ms(o0)} & {ms(ow)} & {ms(ov)} & {strongest(src,m)} \\\\")
    rs.append((lab,rows))
tab(f"{OT}/baselines.tex","Setting & Model & clean & $w$ corr. & $v$ corr. & $w$ attack",blocks(rs),"llcccc")
# ---------------- figures
# A accuracy vs corruption
fig,ax=plt.subplots(1,4,figsize=(16,3.4))
for a,(ds,sc,lab) in zip(ax,SET):
    d=L2(ds,sc)
    if d is None: a.set_title(lab+" (n/a)"); continue
    s=d[(d.kind=="sweep")&(d.direction=="w")]
    for m,st in [("PoE","o-"),("SheafAlign","s-"),("DGSA-hard(clean95)","^-"),("DGSA-soft","v-"),("V-only","k--")]:
        g=s[s.model==m].groupby("rate").overall.agg(["mean","std"])
        a.errorbar(g.index*100,g["mean"]*100,yerr=g["std"]*100,fmt=st,label=m,capsize=2,ms=4)
    a.set_title(lab); a.set_xlabel("corruption rate of w (%)"); a.grid(alpha=.3)
ax[0].set_ylabel("overall accuracy (%)"); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OF}/figA_accuracy.png",dpi=200); plt.close()
# B AUROC
fig,ax=plt.subplots(1,3,figsize=(14,3.2))
for a,(ds,sc,lab) in zip(ax,SET[:3]):
    d=L2(ds,sc)
    if d is None: continue
    dt=d[d.kind=="detect"]; drs=["w","v","both_indep","both_coord"]; x=np.arange(4)
    for i,m in enumerate(["discord","PoE-variance","PoE-entropy","Sheaf-entropy"]):
        a.bar(x+i*.2,[dt[(dt.direction==dr)&(dt.model==m)].auroc.mean() for dr in drs],.2,yerr=[dt[(dt.direction==dr)&(dt.model==m)].auroc.std() for dr in drs],label=m,capsize=2)
    a.axhline(.5,color="k",ls=":",lw=1); a.set_xticks(x+.3); a.set_xticklabels(["w","v","both ind.","both coord."],fontsize=8); a.set_ylim(.3,1.02); a.set_title(lab)
ax[0].set_ylabel("detection AUROC"); ax[0].legend(fontsize=6,loc="lower left"); plt.tight_layout(); plt.savefig(f"{OF}/figB_detection.png",dpi=200); plt.close()
# D risk coverage
fig,ax=plt.subplots(2,3,figsize=(13,6))
for j,(ds,sc,lab) in enumerate(SET[:3]):
    d=L2(ds,sc)
    if d is None or "riskcov" not in set(d.kind): continue
    for i,dr in enumerate(["w","v"]):
        a=ax[i,j]; r=d[(d.kind=="riskcov")&(d.direction==dr)]
        for m in ["discord","PoE-variance","PoE-entropy","Sheaf-entropy"]:
            g=r[r.model==m].groupby("rate").sel_acc.agg(["mean","std"])
            a.errorbar(g.index,g["mean"]*100,yerr=g["std"]*100,label=m,capsize=2,ms=3,marker="o")
        a.set_title(f"{lab}: {dr} corrupted",fontsize=9); a.grid(alpha=.3); a.set_xlabel("coverage (fraction accepted)")
        if j==0: a.set_ylabel("accuracy on accepted (%)")
ax[0,0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OF}/figD_riskcoverage.png",dpi=200); plt.close()
# ---------------- loss-term ablation (Table: ablation, Fig. 4); raw data from ablation_loss_terms.py
AB=f"{R3}/ablation_loss_terms.csv"
if os.path.exists(AB):
    ab=pd.read_csv(AB)
    order=[("full (task+align+recon)","task + align + recon (full, published)","Full\n(task+align+recon)","tab:red"),
           ("task+align only (no recon)","task + alignment only","task+align\nonly","tab:orange"),
           ("task+recon only (no align)","task + reconstruction only","task+recon\nonly","tab:purple"),
           ("task only (no align, no recon)","task only","task\nonly","tab:green")]
    stats=[(ab[ab.variant==v].clean_acc.mean()*100, ab[ab.variant==v].corrupted_acc.mean()*100, ab[ab.variant==v].corrupted_acc.std(ddof=0)*100) for v,_,_,_ in order]
    best=int(np.argmax([x[1] for x in stats])); rows=[]
    for i,((v,lab,_,_),(c,m,sd)) in enumerate(zip(order,stats)):
        cm=f"\\textbf{{{m:.1f}\\%}}" if i==best else f"{m:.1f}\\%"
        rows.append(f"{lab} & {c:.1f}\\% & {cm} ({sd:.1f}) \\\\")
    open(f"{OT}/ablation.tex","w").write("\\begin{tabular}{lcc}\n\\toprule\n\\textbf{Loss variant} & \\textbf{Clean acc.} & \\textbf{Corrupted-subset acc.} \\\\\n\\midrule\n"+"\n".join(rows)+"\n\\bottomrule\n\\end{tabular}\n")
    fig,a=plt.subplots(figsize=(7.5,4.4))
    a.bar(range(4),[x[1] for x in stats],yerr=[x[2] for x in stats],color=[o[3] for o in order],capsize=4,ecolor="black")
    a.set_xticks(range(4)); a.set_xticklabels([o[2] for o in order])
    a.set_ylabel("Corrupted-subset accuracy (%)"); a.set_title("Loss-term ablation: digits-word (which term causes the collapse?)")
    a.grid(axis="y",alpha=.3); a.set_axisbelow(True)
    plt.tight_layout(); plt.savefig(f"{OF}/fig4_loss_ablation.png",dpi=140); plt.close()
print("nseeds",nseeds); print(os.listdir(OT),os.listdir(OF))
