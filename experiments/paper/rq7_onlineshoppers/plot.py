from pathlib import Path
import os
import sys
import json, hashlib, re, math
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator
import networkx as nx
from PIL import Image

PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE/'scripts'))
from isolated_worker import required_output
SOURCE_ROOT=Path(os.environ.get('FAST30_SOURCE_ROOT','/root/autodl-tmp/fast30_final')).resolve()
ROOT=required_output('RQ7_OUTPUT_ROOT',forbidden=(SOURCE_ROOT/'results',))
SOURCE=ROOT/"evidence.json"
source_hash=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
e=json.loads(SOURCE.read_text())
BLUE="#2878B5"; ORANGE="#FA8926"; GREY="#A7A7A7"; DARK="#28343F"
FONT="STIXGeneral"; DPI=600
plt.rcParams.update({
    "font.family": FONT, "mathtext.fontset": "stix",
    "font.size":8.5, "axes.labelsize":9, "xtick.labelsize":8.5,
    "ytick.labelsize":8.5, "legend.fontsize":8.2,
    "axes.linewidth":.65, "savefig.facecolor":"white",
})
core=set(e["p2"]["core_features"])
recovered=set(e["p3"]["recovered_features"])
excluded=set(e["final"]["excluded_features"])
all_features=e["p1"]["feature_names"]
scores=e["p1"]["contribution_scores"]
threshold=e["p1"]["high_contrib_threshold"]
assert len(all_features)==17 and len(core)==11 and len(recovered)==4 and len(excluded)==2
assert not(core & recovered or core & excluded or recovered & excluded)
assert core | recovered | excluded == set(all_features)
assert all(scores[n]>0 for n in all_features)
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE]}
meta={"input_evidence_sha256":source_hash,"font_family":FONT,"font_description":"Times-style STIX serif","minimum_font_size_pt":8.2,"dpi":DPI}

# Figure 3: original contribution scores only; a log axis keeps low scores legible.
order=sorted(all_features,key=lambda n:scores[n],reverse=True)
values=np.array([scores[n] for n in order])
colours=[BLUE if n in core else ORANGE if n in recovered else GREY for n in order]
fig,ax=plt.subplots(figsize=(3.35,4.2))
fig.subplots_adjust(left=.435,right=.985,bottom=.13,top=.88)
lower=1e-7
ax.barh(np.arange(len(order)),values-lower,left=lower,height=.66,color=colours,edgecolor="none",zorder=3)
ax.set_xscale("log")
ax.set_xlim(lower,.35)
ax.set_ylim(-.8,len(order)-.2)
ax.invert_yaxis()
ax.set_yticks(np.arange(len(order)),[n.replace("_"," ") for n in order])
ax.tick_params(axis="y",length=0,pad=4)
ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4))
ax.set_xticks([1e-7,1e-5,1e-3,1e-1])
ax.xaxis.set_major_formatter(LogFormatterMathtext())
ax.xaxis.set_minor_locator(NullLocator())
ax.tick_params(axis="x",width=.6,length=3,pad=3)
ax.grid(axis="x",color="#E8E8E8",linewidth=.5,zorder=0)
ax.axvline(threshold,color=DARK,linewidth=1.05,linestyle=(0,(3,2)),zorder=5)
ax.set_xlabel(r"Contribution score $\mathrm{Con}_i$ (log scale)",labelpad=6)
for spine in ("top","right","left"):ax.spines[spine].set_visible(False)
handles=[Patch(facecolor=BLUE,label="P2 core"),Patch(facecolor=ORANGE,label="P3 recovered"),Patch(facecolor=GREY,label="Excluded"),Line2D([0],[0],color=DARK,lw=1.05,ls=(0,(3,2)),label="P1 threshold")]
fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.52,.99),frameon=False,ncol=2,columnspacing=1.7,handlelength=1.4,handletextpad=.5,labelspacing=.65)
p3=ROOT/"fig3_onlineshoppers_s01_contribution.png"
fig.savefig(p3,dpi=DPI)
fig.savefig(ROOT/"fig3_onlineshoppers_s01_contribution_preview.png",dpi=150)
plt.close(fig)
meta["fig3"]={"file":str(p3),"physical_inches":[3.35,4.2],"features":len(order),"group_counts":{"P2 core":len(core),"P3 recovered":len(recovered),"Excluded":len(excluded)},"xscale":"log","original_contribution_scores_only":True,"P1_threshold":threshold,"feature_order":order}

# Figure 4: use the exact undirected union of edges appearing in retained paths.
target=e["p2"]["target_name"]
paths=e["p2"]["retained_paths"]
edge_counts=Counter()
for path in paths:
    for a,b in zip(path[:-1],path[1:]):edge_counts[tuple(sorted((a,b)))]+=1
stored={tuple(sorted(pair)) for pair in e["p2"]["retained_edges"]}
assert set(edge_counts)==stored and len(paths)==51 and len(stored)==27
assert set(n for p in paths for n in p)==core|{target}
g=nx.Graph()
g.add_nodes_from(sorted(core)+[target])
g.add_edges_from(sorted(stored))
labels={n:re.sub(r"(?<=[a-z])(?=[A-Z])","\n",n) for n in g}
# Score deterministic spring embeddings for label separation and edge crossings.
def cross(a,b,c,d):
    def orient(p,q,r):return np.cross(q-p,r-p)
    return orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0
def layout_cost(pos):
    nodes=list(g)
    overlap=0.
    for i,u in enumerate(nodes):
        for v in nodes[i+1:]:
            dx=abs(pos[u][0]-pos[v][0]);dy=abs(pos[u][1]-pos[v][1])
            if dx<.44 and dy<.23:overlap+=((.44-dx)/.44)*((.23-dy)/.23)*100
    edges=list(g.edges)
    crossings=sum(cross(pos[a],pos[b],pos[c],pos[d]) for i,(a,b) in enumerate(edges) for c,d in edges[i+1:] if len({a,b,c,d})==4)
    return overlap+crossings
best=None
for seed in range(80):
    pos=nx.spring_layout(g,seed=seed,k=.75,iterations=450)
    arr=np.array(list(pos.values()))
    span=np.ptp(arr,axis=0)
    pos={n:(np.array(v)-arr.min(axis=0))/span*2-1 for n,v in pos.items()}
    cost=layout_cost(pos)
    if best is None or cost<best[0]:best=(cost,seed,pos)
_,chosen_seed,pos=best
fig,ax=plt.subplots(figsize=(3.35,3.45))
fig.subplots_adjust(left=.025,right=.975,bottom=.025,top=.91)
nx.draw_networkx_edges(g,pos,ax=ax,edge_color="#7F8E99",width=.8,alpha=.75)
for n,(x,y) in pos.items():
    ax.text(x,y,labels[n],ha="center",va="center",fontsize=8.5,fontfamily=FONT,
        fontweight="bold" if n==target else "normal",
        color="white" if n==target else "#163448",
        bbox=dict(boxstyle="round,pad=.32",facecolor=DARK if n==target else "#E1EFF9",edgecolor=DARK if n==target else BLUE,linewidth=.85),
        zorder=5,linespacing=1.0)
ax.set_xlim(-1.23,1.23);ax.set_ylim(-1.18,1.18);ax.axis("off")
fig.legend(handles=[Patch(facecolor="#E1EFF9",edgecolor=BLUE,label="P2 core"),Patch(facecolor=DARK,label="Target")],loc="upper center",bbox_to_anchor=(.5,1),frameon=False,ncol=2,columnspacing=2,handlelength=1.4,handletextpad=.5)
p4=ROOT/"fig4_onlineshoppers_s01_dependency.png"
fig.savefig(p4,dpi=DPI)
fig.savefig(ROOT/"fig4_onlineshoppers_s01_dependency_preview.png",dpi=150)
plt.close(fig)
meta["fig4"]={"file":str(p4),"physical_inches":[3.35,3.45],"core_features":len(core),"target":target,"node_count":len(g),"edge_count":len(g.edges),"retained_path_count":len(paths),"undirected":True,"edge_counts_shown":False,"layout_seed":chosen_seed,"layout_cost":float(best[0]),"edge_path_counts":[{"u":a,"v":b,"count":v} for (a,b),v in sorted(edge_counts.items())]}
for key in ["fig3","fig4"]:
    p=Path(meta[key]["file"])
    with Image.open(p) as im:meta[key].update(pixels=list(im.size),stored_dpi=list(im.info.get("dpi",[])))
    meta[key]["sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE]}
(ROOT/"figure_metadata.json").write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))

