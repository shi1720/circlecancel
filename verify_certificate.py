"""Independent exact verifier; Python standard library only.

Uses the degree-one trigonometric nonnegativity criterion, rather than
circle-image centers, optimizer output, or sampled boundary points.
"""
import json,sys
from fractions import Fraction as F

def z(x):
    if not isinstance(x, (list, tuple)) or len(x)!=2:
        raise ValueError('complex coordinate must have two rational components')
    return tuple(F(t) for t in x)
def plus(x,y):return x[0]+y[0],x[1]+y[1]
def minus(x,y):return x[0]-y[0],x[1]-y[1]
def times(x,y):return x[0]*y[0]-x[1]*y[1],x[0]*y[1]+x[1]*y[0]
def conj(x):return x[0],-x[1]
def norm(x):return x[0]**2+x[1]**2
def scale(x,k):return x[0]*k,x[1]*k

class CertificateError(ValueError):
    """A certificate does not meet the declared cancellation contract."""

def require(condition, message):
    # These acceptance checks must remain active under python -O.
    if not condition:
        raise CertificateError(message)

def check_edge(a,b,c,R,g,e):
    v=minus(g,(F(1),F(0)))
    d=plus(times(v,c),minus(a,times(g,b)))
    A=minus(c,a)
    q=e*e*(R*R+norm(A))-R*R*norm(v)-norm(d)
    l=minus(scale(conj(A),e*e),times(v,conj(d)))
    return q>=0 and q*q>=4*R*R*norm(l)

def verify(data):
    require(isinstance(data, list), 'certificate collection must be a list')
    total=0
    for case in data:
        require(isinstance(case, dict), 'case must be an object')
        require(isinstance(case['circles'], list) and len(case['circles'])>0,
                'at least one protected circle is required')
        require(isinstance(case['zeros'], list) and isinstance(case['poles'], list),
                'root collections must be lists')
        require(isinstance(case['selected'], list), 'selected edges must be a list')
        require(F(case['budget'])>=0, 'budget must be nonnegative')
        iz=set(); jp=set(); product=F(1)
        for circ in case['circles']:
            c,R=z(circ['center']),F(circ['radius'])
            require(R>0, 'circle radius must be positive')
            for root in case['zeros']+case['poles']:
                require(norm(minus(z(root),c))!=R*R, 'root on protected circle')
        for edge in case['selected']:
            i,j=edge['i'],edge['j']
            require(type(i) is int and 0<=i<len(case['zeros']), 'invalid zero index')
            require(type(j) is int and 0<=j<len(case['poles']), 'invalid pole index')
            require(i not in iz and j not in jp, 'not a matching')
            iz.add(i); jp.add(j)
            a,b=z(case['zeros'][i]),z(case['poles'][j])
            g,e=z(edge['gain']),F(edge['epsilon'])
            require(norm(g)>0 and 0<=e<1, 'invalid gain or edge error')
            for circ in case['circles']:
                c,R=z(circ['center']),F(circ['radius'])
                require(R>0, 'circle radius must be positive')
                require(norm(minus(a,c))!=R*R and norm(minus(b,c))!=R*R,
                        'root on protected circle')
                require(check_edge(a,b,c,R,g,e), 'circle inequality failed')
            product*=1+e
            total+=1
        require(product-1<=F(case['budget']), 'budget exceeded')
    return len(data),total

if __name__=='__main__':
    with open(sys.argv[1] if len(sys.argv)>1 else 'results/certificates.json') as f: data=json.load(f)
    n,k=verify(data)
    print(f'PASS: {n} cases, {k} selected cancellations; exact continuous-circle inequalities and total budgets verified.')
