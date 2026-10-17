"""Continuous-circle cancellation bounds and exact multiplicative matching.

A factor (z-a)/(z-b) is removed by multiplying the remaining rational
function by gamma. The relative change is gamma*(z-b)/(z-a)-1.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import numpy as np

@dataclass(frozen=True)
class Circle:
    center: complex
    radius: float


def image_disk(a, b, circle):
    """Image circle center m and radius s for h(z)=(z-b)/(z-a)."""
    A, B, R = a-circle.center, b-circle.center, circle.radius
    D = R*R-abs(A)**2
    if D == 0:
        raise ValueError('zero on protected circle')
    return (R*R-B*np.conj(A))/D, R*abs(a-b)/abs(D)


def error_bound(gain, disks):
    return max(abs(gain*m-1)+abs(gain)*s for m,s in disks)


def optimal_gain(a,b,circles):
    """SOCP proposal. Exact post-verification is provided separately."""
    disks=[image_disk(a,b,c) for c in circles]
    if len(disks)==1:
        m,s=disks[0]
        if abs(m)<=s:
            return 0j,1.,disks
        g=1/m
        return g,error_bound(g,disks),disks
    import cvxpy as cp
    g=cp.Variable(2); t=cp.Variable()
    constraints=[]
    for m,s in disks:
        mat=np.array([[m.real,-m.imag],[m.imag,m.real]])
        constraints.append(cp.norm(mat@g-np.array([1.,0.]))+s*cp.norm(g)<=t)
    problem=cp.Problem(cp.Minimize(t),constraints)
    problem.solve(solver='CLARABEL',tol_gap_abs=1e-10,tol_feas=1e-10,tol_gap_rel=1e-10)
    if g.value is None:
        raise RuntimeError(problem.status)
    z=complex(*g.value)
    return z,error_bound(z,disks),disks

# Complex rational arithmetic is deliberately independent of numpy/CVXPY.
def qcomplex(z):
    if isinstance(z,tuple): return z
    return F(float(complex(z).real)),F(float(complex(z).imag))
def add(x,y): return x[0]+y[0],x[1]+y[1]
def sub(x,y): return x[0]-y[0],x[1]-y[1]
def mul(x,y): return x[0]*y[0]-x[1]*y[1],x[0]*y[1]+x[1]*y[0]
def norm2(x): return x[0]**2+x[1]**2
def divreal(x,d): return x[0]/d,x[1]/d

def exact_image(a,b,circle):
    A=sub(qcomplex(a),qcomplex(circle.center)); B=sub(qcomplex(b),qcomplex(circle.center))
    rr=F(float(circle.radius))**2; D=rr-norm2(A)
    if not D: raise ValueError('zero on circle')
    m=divreal(sub((rr,F(0)),mul(B,(A[0],-A[1]))),D)
    s2=rr*norm2(sub(A,B))/D**2
    return m,s2


def verify_pair(a,b,circles,gain,epsilon):
    """Exact proof for the nominal dyadic root and circle inputs.

    sqrt(x)+sqrt(y)<=e iff e>=0, e²-x-y>=0 and
    (e²-x-y)²>=4xy. No floating square roots enter acceptance.
    """
    g=qcomplex(gain); e=F(epsilon)
    if e<0 or norm2(g)==0: return False
    for c in circles:
        m,s2=exact_image(a,b,c)
        x=norm2(sub(mul(g,m),(F(1),F(0))))
        y=norm2(g)*s2
        t=e*e-x-y
        if t<0 or t*t<4*x*y: return False
    return True


def certified_edge(a,b,circles,gain=None,denom=10**10):
    """Rational gain and outward rational error, verified exactly."""
    if gain is None: gain,_,_=optimal_gain(a,b,circles)
    g=(F(round(gain.real*10**12),10**12),F(round(gain.imag*10**12),10**12))
    if norm2(g)==0: return None
    gf=complex(float(g[0]),float(g[1]))
    disks=[image_disk(a,b,c) for c in circles]
    approx=error_bound(gf,disks)
    e=F(math.ceil(approx*denom)+2,denom)
    for _ in range(30):
        if verify_pair(a,b,circles,g,e): return g,e
        e+=max(F(1,denom),e/F(10**8))
    raise ArithmeticError('Unable to verify proposed edge')


def optimal_path(n,m,edges):
    """Exact minimum-product matching at every possible cardinality.

    edges[(i,j)] is a rational per-factor error epsilon >= 0.
    Successive shortest augmenting paths use rational products, equivalent
    to logarithmic costs but without rounding or a transcendental comparison.
    """
    factors={ij:1+F(e) for ij,e in edges.items()}
    if any(t<1 for t in factors.values()): raise ValueError('negative error')
    chosen=set(); out=[(F(1),set())]; source=n+m; sink=source+1; N=sink+1
    while True:
        iz={i for i,j in chosen}; jp={j for i,j in chosen}
        arcs=[(source,i,F(1),None) for i in range(n) if i not in iz]
        arcs += [(n+j,sink,F(1),None) for j in range(m) if j not in jp]
        for (i,j),v in factors.items():
            arcs.append((n+j,i,1/v,(i,j)) if (i,j) in chosen else (i,n+j,v,(i,j)))
        dist={source:F(1)}; prev={}
        for _ in range(N-1):
            changed=False
            for u,v,w,ij in arcs:
                if u in dist and (v not in dist or dist[u]*w<dist[v]):
                    dist[v]=dist[u]*w; prev[v]=(u,ij); changed=True
            if not changed: break
        if sink not in dist: break
        v=sink; toggle=[]; seen=set()
        while v!=source:
            if v in seen: raise RuntimeError('residual negative cycle')
            seen.add(v); u,ij=prev[v]
            if ij is not None: toggle.append(ij)
            v=u
        for ij in toggle:
            if ij in chosen: chosen.remove(ij)
            else: chosen.add(ij)
        cost=math.prod(factors[ij] for ij in chosen)
        assert len(chosen)==len(out)
        assert cost>=out[-1][0]
        out.append((cost,chosen.copy()))
    return out


def select(n,m,edges,budget):
    feasible=[(p,M) for p,M in optimal_path(n,m,edges) if p<=1+F(budget)]
    return feasible[-1]


def ratio(z,zeros,poles,matching,gains):
    ans=np.ones_like(np.asarray(z),dtype=complex)
    for i,j in matching:
        gain=gains[(i,j)]
        if isinstance(gain,tuple): gain=complex(float(gain[0]),float(gain[1]))
        ans*=gain*(z-poles[j])/(z-zeros[i])
    return ans


def reduced_factors(zeros,poles,matching,gains,kappa=1+0j):
    """Return retained factors and replacement gain; no refitting."""
    usedz={i for i,j in matching}; usedp={j for i,j in matching}
    if len(usedz)!=len(matching) or len(usedp)!=len(matching):
        raise ValueError('selected edges must form a matching')
    for i,j in matching:
        g=gains[i,j]
        if isinstance(g,tuple):g=complex(float(g[0]),float(g[1]))
        kappa*=g
    return ([a for i,a in enumerate(zeros) if i not in usedz],
            [b for j,b in enumerate(poles) if j not in usedp],kappa)
