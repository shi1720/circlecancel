"""Reproduce experiments, certificates, tables and figures (seed fixed)."""
import os,time,json,csv,warnings,platform
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import scipy,cvxpy,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import AAA
from fractions import Fraction as F
from pathlib import Path
from circlecancel import *
from verify_certificate import verify
warnings.filterwarnings('ignore',category=RuntimeWarning)
ROOT=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','pdf.fonttype':42})
COL=['#176c8c','#d3752a','#485b38','#9b466c']
rng=np.random.default_rng(1720406)
certs=[]; rows=[]

def qs(z):return [str(F(float(complex(z).real))),str(F(float(complex(z).imag)))]
def cert_case(name,aa,bb,cs,M,gains,edges,budget):
 return {'name':name,'zeros':[qs(z) for z in aa],'poles':[qs(z) for z in bb],
 'circles':[{'center':qs(c.center),'radius':str(F(float(c.radius)))} for c in cs],
 'selected':[{'i':i,'j':j,'gain':[str(t) for t in gains[i,j]],'epsilon':str(edges[i,j])} for i,j in sorted(M)],'budget':str(F(budget))}

def all_edges(aa,bb,cs,maxbudget):
 edges={}; gains={}; unit={}; ug={}
 for i,a in enumerate(aa):
  for j,b in enumerate(bb):
   ds=[image_disk(a,b,c) for c in cs]
   lower=max(s/abs(m) if abs(m)>s else 1 for m,s in ds)
   if lower>maxbudget:continue
   ce=certified_edge(a,b,cs)
   if ce and ce[1]<1:
    gains[i,j],edges[i,j]=ce
   ce=certified_edge(a,b,cs,gain=1+0j)
   if ce and ce[1]<1:ug[i,j],unit[i,j]=ce
 return edges,gains,unit,ug

def greedy(aa,bb,edges,budget,metric):
 M=set();p=F(1)
 order=sorted(edges,key=lambda ij: abs(aa[ij[0]]-bb[ij[1]]) if metric=='distance' else edges[ij])
 for i,j in order:
  if any(i==x or j==y for x,y in M):continue
  if p*(1+edges[i,j])<=1+F(budget):
   M.add((i,j));p*=1+edges[i,j]
 return p,M

def raw_nearest(aa,bb,k):
 M=set()
 for i,j in sorted([(i,j) for i in range(len(aa)) for j in range(len(bb))],key=lambda ij:abs(aa[ij[0]]-bb[ij[1]])):
  if any(i==x or j==y for x,y in M):continue
  M.add((i,j))
  if len(M)>=k:break
 return M if k else set()

def sample_error(aa,bb,cs,M,gains):
 return max(float(np.max(abs(ratio(c.center+c.radius*np.exp(2j*np.pi*(np.arange(8192)+.371)/8192),aa,bb,M,gains)-1))) for c in cs)

start=time.perf_counter()
# Exact one-factor sharpness and circle image checks.
sharp=[]
for _ in range(400):
 c=Circle(complex(*rng.normal(size=2)),rng.uniform(.3,2))
 r1,r2=rng.choice([.2,.7,1.2,2.5],2)
 a=c.center+c.radius*r1*np.exp(1j*rng.uniform(0,2*np.pi))
 b=c.center+c.radius*r2*np.exp(1j*rng.uniform(0,2*np.pi))
 g,e,ds=optimal_gain(a,b,[c])
 if not g:continue
 z=c.center+c.radius*np.exp(2j*np.pi*np.arange(16384)/16384)
 actual=max(abs(g*(z-b)/(z-a)-1))
 # For a single circle centered at 1 after gain, every point has this error.
 sharp.append(abs(actual-e)/(1+e))
# Unit-gain small distance failure.
etas=np.logspace(-7,-1,100)
fig,ax=plt.subplots(1,2,figsize=(6.8,2.7))
ax[0].semilogx(etas,[optimal_gain(1-t,1-2*t,[Circle(0,1)])[1] for t in etas],label='optimized gain',color=COL[0])
ax[0].semilogx(etas,np.ones(100),label='unit gain',color=COL[1])
ax[0].set(xlabel=r'Euclidean separation $\eta$',ylabel='worst relative change',title=r'Same side: $a=1-\eta$, $b=1-2\eta$')
ax[0].legend(frameon=False)
# Circle-image obstruction.
csobs=[Circle(-.1,.01),Circle(.1,.01)]
obs=[]
for c,co in zip(csobs,COL):
 m,s=image_disk(0,1,c);th=np.linspace(0,2*np.pi,400)
 ax[1].fill(m.real+s*np.cos(th),m.imag+s*np.sin(th),alpha=.2,color=co)
 ax[1].plot(m.real+s*np.cos(th),m.imag+s*np.sin(th),color=co)
 obs.append(optimal_gain(0,1,[c])[1])
ax[1].axhline(0,color='.7',lw=.6);ax[1].plot(0,0,'k+',ms=9)
ax[1].set(xlabel=r'$\Re h$',ylabel=r'$\Im h$',title='Two image disks: no common gain')
fig.tight_layout();fig.savefig(ROOT/'figures/geometry.pdf');fig.savefig(ROOT/'figures/geometry.png',dpi=180);plt.close(fig)
print('sharpness',len(sharp),max(sharp),flush=True)
# Random multi-contour pairs: individual-circle lower bound versus common gain.
joint=[]
for case in range(120):
 cs=[Circle(0,1),Circle(.35,.45)] if case<60 else [Circle(-.8,.3),Circle(.8,.3)]
 a=complex(*rng.uniform(-1.5,1.5,2));b=a+complex(*rng.normal(size=2))*.12
 ds=[image_disk(a,b,c) for c in cs]
 low=max(min(1,s/abs(m)) if abs(m) else 1 for m,s in ds)
 g,e,_=optimal_gain(a,b,cs)
 joint.append({'lower':low,'joint':min(e,1.),'unit':error_bound(1,ds)})
# Manufactured banks, three budgets and four geometry classes.
for regime in ['interior','boundary','crowded','two_circles']:
 for seed in range(12):
  rr=np.random.default_rng(10000+100*['interior','boundary','crowded','two_circles'].index(regime)+seed)
  n=12
  if regime=='interior':
   bb=.7*np.sqrt(rr.random(n))*np.exp(2j*np.pi*rr.random(n));aa=bb+.002*(rr.normal(size=n)+1j*rr.normal(size=n));cs=[Circle(0,1)]
  elif regime=='boundary':
   bb=(1+rr.uniform(-.02,.02,n))*np.exp(2j*np.pi*rr.random(n));aa=bb+.002*(rr.normal(size=n)+1j*rr.normal(size=n));cs=[Circle(0,1)]
  elif regime=='crowded':
   bb=.3+.008*(rr.normal(size=n)+1j*rr.normal(size=n));aa=.3+.008*(rr.normal(size=n)+1j*rr.normal(size=n));cs=[Circle(0,1)]
  else:
   bb=.8*(rr.normal(size=n)+1j*rr.normal(size=n));aa=bb+.002*(rr.normal(size=n)+1j*rr.normal(size=n));cs=[Circle(0,1),Circle(.35,.45)]
  # Four deliberately non-doublet features; all roots are retained as nominal exact dyadics.
  aa=np.r_[aa,[-.4-.3j,-.4+.3j,.5+.4j,.5-.4j]]
  bb=np.r_[bb,[-1.8-.4j,-1.8+.4j,1.7+.6j,1.7-.6j]]
  t0=time.perf_counter();edges,gains,unit,ug=all_edges(aa,bb,cs,.1)
  path=optimal_path(len(aa),len(bb),edges);elapsed=time.perf_counter()-t0
  for budget in [F(1,1000),F(1,100),F(1,10)]:
   p,M=[(p,M) for p,M in path if p<=1+budget][-1]
   gp,GM=greedy(aa,bb,edges,budget,'cost')
   up,UM=greedy(aa,bb,unit,budget,'distance')
   NM=raw_nearest(aa,bb,len(M));ng={ij:1+0j for ij in NM}
   er=sample_error(aa,bb,cs,M,gains)
   raw=sample_error(aa,bb,cs,NM,ng)
   assert er<=float(p-1)+1e-12
   certs.append(cert_case(f'{regime}-{seed}-{budget}',aa,bb,cs,M,gains,edges,budget))
   rows.append({'regime':regime,'seed':seed,'budget':float(budget),'optimal_k':len(M),'greedy_k':len(GM),'unit_k':len(UM),'bound':float(p-1),'measured':er,'raw_nearest_error':raw,'edge_and_path_s':elapsed})
 print('banks',regime,'done',flush=True)
# AAA-generated rational inputs, three analytic functions, three noise levels, four seeds.
aaarows=[]
funcs={'exp':np.exp,'log':lambda z:np.log(2+z),'tan':lambda z:np.tan(.8*z)}
cs=[Circle(0,1),Circle(0,1.1)]
x=np.exp(2j*np.pi*np.arange(512)/512)
for name,fun in funcs.items():
 for noise in [1e-7,1e-10,1e-13]:
  for seed in range(4):
   rr=np.random.default_rng(20000+seed)
   y=fun(x)+noise*(rr.normal(size=len(x))+1j*rr.normal(size=len(x)))/np.sqrt(2)
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');r=AAA(x,y,rtol=1e-14,max_terms=28,clean_up=False)
   aa=r.roots();bb=r.poles()
   edges,gains,unit,ug=all_edges(aa,bb,cs,.01)
   p,M=select(len(aa),len(bb),edges,F(1,100))
   z=np.concatenate([c.center+c.radius*np.exp(2j*np.pi*(np.arange(8192)+.371)/8192) for c in cs])
   anchor=1.234+.234j
   # Factored nominal rational input uses a barycentric anchor only to fix gain.
   nominal=np.full(len(z),r(anchor),dtype=complex)
   for a in aa:nominal*=(z-a)/(anchor-a)
   for b in bb:nominal*=(anchor-b)/(z-b)
   factor_agreement=float(max(abs(nominal-r(z)))/max(abs(r(z))))
   reduced=nominal*ratio(z,aa,bb,M,gains)
   nr=max(abs(nominal-fun(z)));red=max(abs(reduced-fun(z)))
   # Standard SciPy support-deletion/refit cleanup is an empirical comparator.
   with warnings.catch_warnings():
    warnings.simplefilter('ignore');rc=AAA(x,y,rtol=1e-14,max_terms=28,clean_up=True,clean_up_tol=1e-7)
   se=float(max(abs(rc(z)-r(z)))/max(abs(r(z))))
   aaarows.append({'function':name,'noise':noise,'seed':seed,'poles':len(bb),'removed':len(M),'bound':float(p-1),'measured':sample_error(aa,bb,cs,M,gains),'factor_agreement':factor_agreement,'nominal_truth_abs':float(nr),'reduced_truth_abs':float(red),'scipy_poles':len(rc.poles()),'scipy_change_normalized':se})
   certs.append(cert_case(f'AAA-{name}-{noise}-{seed}',aa,bb,cs,M,gains,edges,F(1,100)))
 print('AAA',name,'done',flush=True)

# Timing larger exact matching problems, independent of geometry.
timings=[]
for n in [16,32,64]:
 for rep in range(3):
  rr=np.random.default_rng(30000+n+rep)
  ed={(i,j):F(int(rr.integers(1,10000)),10**7) for i in range(n) for j in range(n) if rr.random()<.15 or i==j}
  t=time.perf_counter();path=optimal_path(n,n,ed);dt=time.perf_counter()-t
  timings.append({'n':n,'rep':rep,'edges':len(ed),'seconds':dt,'full_k':len(path)-1})
print('verify',verify(certs),flush=True)
(ROOT/'results/certificates.json').write_text(json.dumps(certs,indent=2))
for fname,arr in [('banks',rows),('aaa',aaarows),('joint',joint),('timings',timings)]:
 with open(ROOT/f'results/{fname}.csv','w') as f:
  w=csv.DictWriter(f,fieldnames=arr[0]);w.writeheader();w.writerows(arr)
summary={'sharpness_cases':len(sharp),'sharpness_max_normalized_residual':max(sharp),'obstruction_individual':obs,'obstruction_joint':optimal_gain(0,1,csobs)[1], 'joint_gain_gap_count':sum(d['joint']>d['lower']+1e-6 for d in joint),'joint_max_gap':max(d['joint']-d['lower'] for d in joint),'bank_cases':len(rows),'aaa_cases':len(aaarows),'certificates':verify(certs),'elapsed_s':time.perf_counter()-start,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'cvxpy':cvxpy.__version__,'platform':platform.platform()}
(ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()))
# Figures from saved outcomes.
fig,ax=plt.subplots(1,2,figsize=(6.8,2.8))
for k,regime in enumerate(['interior','boundary','crowded','two_circles']):
 sub=[r for r in rows if r['regime']==regime and r['budget']==.01]
 xx=np.array([r['bound'] for r in sub]);yy=np.array([r['measured'] for r in sub]);mask=xx>0
 ax[0].scatter(xx[mask],yy[mask],s=18,label=regime.replace('_',' '),color=COL[k])
 sub=[r for r in rows if r['regime']==regime and r['budget']==.01]
 vals=[np.mean([r[key] for r in sub]) for key in ['unit_k','greedy_k','optimal_k']]
 ax[1].bar(np.arange(3)+(k-1.5)*.18,vals,width=.17,color=COL[k])
ax[0].plot([1e-6,.01],[1e-6,.01],'k--',lw=.8);ax[0].set(xscale='log',yscale='log',xlabel='continuous bound',ylabel='sampled relative change');ax[0].legend(fontsize=7,frameon=False)
ax[1].set(xticks=np.arange(3),xticklabels=['distance / unit','error greedy','exact matching'],ylabel='mean pairs removed',ylim=(0,13))
fig.tight_layout();fig.savefig(ROOT/'figures/banks.pdf');fig.savefig(ROOT/'figures/banks.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(6.8,2.7))
for k,name in enumerate(funcs):
 for xoff,noise in enumerate([1e-7,1e-10,1e-13]):
  sub=[r for r in aaarows if r['function']==name and r['noise']==noise]
  ax[0].scatter([xoff+(k-1)*.18]*len(sub),[r['removed'] for r in sub],s=18,color=COL[k],label=name if xoff==0 else None)
  ax[1].scatter([r['nominal_truth_abs'] for r in sub],[r['reduced_truth_abs'] for r in sub],s=18,color=COL[k])
ax[0].set(xticks=[0,1,2],xticklabels=['$10^{-7}$','$10^{-10}$','$10^{-13}$'],xlabel='sample noise scale',ylabel='pairs removed');ax[0].legend(frameon=False)
ax[1].plot([1e-15,1e-2],[1e-15,1e-2],'k--',lw=.8);ax[1].set(xscale='log',yscale='log',xlabel='nominal error to analytic function',ylabel='reduced error to analytic function')
fig.tight_layout();fig.savefig(ROOT/'figures/aaa.pdf');fig.savefig(ROOT/'figures/aaa.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2,default=lambda x:x.item()),flush=True)
